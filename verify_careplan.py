"""
Care plan verification script — Layer 1 (code only, no AI).

Checks each CarePlan row against the structured fields stored in the same row:
  - Numeric consistency : doses / ICD codes / MRN / CPSO extracted from `plan`
                          are compared to the DB columns.
  - Drug name audit     : every drug name found in `plan` is checked against
                          medication_name + medication_history + patient_records.

Usage:
    python verify_careplan.py              # verify all care plans
    python verify_careplan.py --id 42      # verify one care plan by id
    python verify_careplan.py --id 42 --verbose
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from typing import Literal

from database import get_db
from models import CarePlan

# ── Types ─────────────────────────────────────────────────────────────────────

Status = Literal["MATCH", "MISMATCH", "NOT_IN_RECORD"]

@dataclass
class Finding:
    check: str          # what was checked
    status: Status
    expected: str = ""
    found: str = ""
    detail: str = ""

@dataclass
class Report:
    careplan_id: int
    patient: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(f.status == "MATCH" for f in self.findings)

    def summary(self) -> str:
        counts = {"MATCH": 0, "MISMATCH": 0, "NOT_IN_RECORD": 0}
        for f in self.findings:
            counts[f.status] += 1
        return (f"CarePlan #{self.careplan_id}  [{self.patient}]  "
                f"✓ {counts['MATCH']}  ✗ {counts['MISMATCH']}  "
                f"? {counts['NOT_IN_RECORD']}")


# ── Helpers ───────────────────────────────────────────────────────────────────

# Known drug name patterns (extend as needed)
_DRUG_PATTERN = re.compile(
    r'\b('
    r'metformin|insulin|lisinopril|atorvastatin|amlodipine|ramipril|'
    r'rosuvastatin|simvastatin|glipizide|glyburide|sitagliptin|empagliflozin|'
    r'dapagliflozin|canagliflozin|liraglutide|semaglutide|jardiance|ozempic|'
    r'victoza|januvia|glucophage|lipitor|crestor|norvasc|altace|'
    r'warfarin|aspirin|clopidogrel|metoprolol|atenolol|bisoprolol|'
    r'hydrochlorothiazide|furosemide|spironolactone|omeprazole|pantoprazole|'
    r'[a-z]+[- ]?\d+\s?mg'   # anything like "drug 500mg" or "drug500mg"
    r')\b',
    re.IGNORECASE,
)

# Numeric value with optional unit — captures "1000 mg", "6.5%", "130/80"
_NUM_PATTERN = re.compile(r'\b(\d+(?:\.\d+)?)\s*(mg|mcg|mmol|kg|%|mmhg)?\b', re.IGNORECASE)

# ICD-10 codes
_ICD_PATTERN = re.compile(r'\b([A-Z]\d{2}(?:\.\d{1,3})?)\b')

# Dosage string like "500mg", "1000 mg BID"
_DOSE_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*(mg|mcg|g)\b', re.IGNORECASE)


def _normalise(text: str) -> str:
    return text.lower().strip()


def _extract_doses(text: str) -> list[tuple[str, str]]:
    """Return list of (value, unit) pairs found in text."""
    return [(m.group(1), m.group(2).lower()) for m in _DOSE_PATTERN.finditer(text)]


def _extract_drugs(text: str) -> set[str]:
    return {m.group(0).lower() for m in _DRUG_PATTERN.finditer(text)}


def _extract_icd(text: str) -> set[str]:
    return {m.group(1).upper() for m in _ICD_PATTERN.finditer(text)}


def _known_sources(cp: CarePlan) -> str:
    """All text the pharmacist actually supplied — the ground truth."""
    parts = [
        cp.medication_name or "",
        cp.medication_history or "",
        cp.patient_records or "",
        cp.primary_diagnosis or "",
        cp.additional_diagnoses or "",
    ]
    return " ".join(parts)


# ── Checks ────────────────────────────────────────────────────────────────────

def check_icd_codes(cp: CarePlan) -> list[Finding]:
    """ICD-10 codes in care plan should match those in the DB."""
    findings = []
    db_codes = _extract_icd(
        (cp.primary_diagnosis or "") + " " + (cp.additional_diagnoses or "")
    )
    plan_codes = _extract_icd(cp.plan or "")

    for code in plan_codes:
        if code in db_codes:
            findings.append(Finding(
                check=f"ICD-10 code {code}",
                status="MATCH",
                expected=code,
                found=code,
            ))
        else:
            findings.append(Finding(
                check=f"ICD-10 code {code}",
                status="NOT_IN_RECORD",
                expected="(not in DB diagnoses)",
                found=code,
                detail="Code appears in care plan but not in primary or additional diagnoses.",
            ))

    return findings


def check_current_medication_dose(cp: CarePlan) -> list[Finding]:
    """Doses stated in the care plan for the current medication should match the DB."""
    findings = []
    if not cp.medication_name:
        return findings

    db_doses = _extract_doses(cp.medication_name)
    plan_doses = _extract_doses(cp.plan or "")

    if not db_doses:
        return findings  # no numeric dose recorded — nothing to verify

    db_dose_strings = {f"{v}{u}" for v, u in db_doses}

    for value, unit in plan_doses:
        dose_str = f"{value}{unit}"
        # Only flag doses that look like they relate to the current medication
        med_name_root = _normalise(cp.medication_name.split()[0])
        # Find if this dose appears near the medication name in the plan
        context_pattern = re.compile(
            rf'{re.escape(med_name_root)}.{{0,60}}{re.escape(value)}\s*{re.escape(unit)}',
            re.IGNORECASE,
        )
        if context_pattern.search(cp.plan or ""):
            status: Status = "MATCH" if dose_str in db_dose_strings else "MISMATCH"
            findings.append(Finding(
                check=f"Dose for {cp.medication_name.split()[0]}",
                status=status,
                expected=", ".join(db_dose_strings),
                found=dose_str,
                detail="" if status == "MATCH" else "Care plan states a different dose than DB.",
            ))

    return findings


def check_drug_names(cp: CarePlan) -> list[Finding]:
    """
    Every drug name found in the care plan should appear somewhere in the
    patient-supplied data (medication_name, medication_history, patient_records).
    Drugs only in guidelines/recommendations are expected to be tagged [INFERRED]
    or [GUIDELINE] — flag them as NOT_IN_RECORD so the reviewer can decide.
    """
    findings = []
    sources = _known_sources(cp)
    plan_drugs = _extract_drugs(cp.plan or "")
    source_drugs = _extract_drugs(sources)

    for drug in sorted(plan_drugs):
        # Skip dose-only matches like "500mg" which aren't drug names
        if re.fullmatch(r'\d+(\.\d+)?\s*(mg|mcg|g)', drug, re.IGNORECASE):
            continue

        if any(drug in sd or sd in drug for sd in source_drugs):
            findings.append(Finding(
                check=f"Drug: {drug}",
                status="MATCH",
                found=drug,
            ))
        else:
            findings.append(Finding(
                check=f"Drug: {drug}",
                status="NOT_IN_RECORD",
                found=drug,
                detail="Mentioned in care plan but not in medication_name, "
                       "medication_history, or patient_records.",
            ))

    return findings


def check_mrn_present(cp: CarePlan) -> list[Finding]:
    """MRN in care plan should match DB."""
    if not cp.patient_mrn:
        return []
    mentioned = cp.patient_mrn in (cp.plan or "")
    # MRN is optional to include, so only flag a mismatch if a *different* number appears
    pattern = re.compile(r'\bMRN[:\s#]*(\d{6})\b', re.IGNORECASE)
    matches = pattern.findall(cp.plan or "")
    findings = []
    for m in matches:
        status: Status = "MATCH" if m == cp.patient_mrn else "MISMATCH"
        findings.append(Finding(
            check="MRN",
            status=status,
            expected=cp.patient_mrn,
            found=m,
            detail="" if status == "MATCH" else "Care plan contains a different MRN.",
        ))
    return findings


# ── Main verification ──────────────────────────────────────────────────────────

def verify(cp: CarePlan) -> Report:
    report = Report(
        careplan_id=cp.id,
        patient=f"{cp.patient_first_name} {cp.patient_last_name}",
    )
    report.findings += check_icd_codes(cp)
    report.findings += check_current_medication_dose(cp)
    report.findings += check_drug_names(cp)
    report.findings += check_mrn_present(cp)
    return report


def print_report(report: Report, verbose: bool = False) -> None:
    print(report.summary())
    if verbose or not report.passed:
        for f in report.findings:
            icon = {"MATCH": "✓", "MISMATCH": "✗", "NOT_IN_RECORD": "?"}[f.status]
            line = f"  {icon} [{f.status:<14}] {f.check}"
            if f.status != "MATCH":
                if f.expected:
                    line += f"  |  expected: {f.expected}"
                if f.found:
                    line += f"  |  found: {f.found}"
                if f.detail:
                    line += f"\n              → {f.detail}"
            print(line)
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Verify care plan consistency.")
    parser.add_argument("--id", type=int, help="Care plan ID to verify (default: all)")
    parser.add_argument("--verbose", action="store_true", help="Show all findings including MATCH")
    args = parser.parse_args()

    db = next(get_db())
    try:
        if args.id:
            cp = db.query(CarePlan).filter(CarePlan.id == args.id).first()
            if not cp:
                print(f"Care plan #{args.id} not found.")
                sys.exit(1)
            care_plans = [cp]
        else:
            care_plans = db.query(CarePlan).order_by(CarePlan.id).all()

        if not care_plans:
            print("No care plans found in database.")
            return

        total = len(care_plans)
        failed = 0
        for cp in care_plans:
            report = verify(cp)
            print_report(report, verbose=args.verbose)
            if not report.passed:
                failed += 1

        print(f"── Summary: {total - failed}/{total} care plans fully matched ──")
    finally:
        db.close()


if __name__ == "__main__":
    main()
