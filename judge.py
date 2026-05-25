"""
LLM-as-Judge: verifies every claim in a generated care plan.

Verdict per claim:
  VERIFIED      — directly traceable to the provided patient data
  UNVERIFIED    — plausible but cannot be confirmed from the input
  HALLUCINATION — contradicts the input, or fabricates patient-specific facts

Metrics (stored in the report):
  Precision = VERIFIED / (VERIFIED + HALLUCINATION)
  Recall    = ground-truth facts covered by VERIFIED claims / total ground-truth facts
  F1        = 2 * P * R / (P + R)
"""

import json
import os
import re
from dataclasses import dataclass

import anthropic

JUDGE_PROMPT = """\
You are a clinical accuracy auditor reviewing an AI-generated medication care plan.
Your role is to verify every factual claim against the provided ground-truth patient data.
You are NOT the author — you are an independent reviewer checking for errors and hallucinations.

## Verdict definitions (use exactly these labels):
- VERIFIED      : The claim is directly and explicitly supported by the patient data below.
- UNVERIFIED    : The claim is clinically reasonable but cannot be confirmed or denied from the data.
- HALLUCINATION : The claim contradicts the patient data, or asserts a patient-specific fact
                  (name, dose, diagnosis, medication, MRN, provider) that is not in the data.

## Rules:
1. Extract every distinct factual claim from the care plan (one claim per bullet or sentence).
2. Generic clinical statements that apply to any patient (e.g., "Metformin should be taken with food")
   are UNVERIFIED, not HALLUCINATION — they are not patient-specific.
3. Only mark HALLUCINATION when the claim is patient-specific AND wrong or fabricated.
4. Be conservative: when uncertain between UNVERIFIED and HALLUCINATION, choose UNVERIFIED.
5. Your reason must cite the specific field (e.g., "medication_name field states 1000mg, plan states 500mg").

## Ground-truth patient data:
{patient_data}

## Care plan to audit:
{care_plan}

## Output — respond with ONLY valid JSON, no markdown fences, no prose:
{{
  "claims": [
    {{
      "text": "<exact quote or close paraphrase of the claim>",
      "verdict": "VERIFIED | UNVERIFIED | HALLUCINATION",
      "reason": "<one sentence citing the specific evidence or contradiction>"
    }}
  ]
}}
"""


@dataclass
class JudgeReport:
    claims: list[dict]
    verified: int
    unverified: int
    hallucination: int
    precision: float
    recall: float
    f1: float
    has_hallucination: bool

    def to_dict(self) -> dict:
        return {
            "claims": self.claims,
            "metrics": {
                "verified": self.verified,
                "unverified": self.unverified,
                "hallucination": self.hallucination,
                "precision": round(self.precision, 4),
                "recall": round(self.recall, 4),
                "f1": round(self.f1, 4),
            },
        }


def _build_patient_data(cp) -> str:
    """Serialise the structured DB fields into a readable ground-truth block."""
    lines = [
        f"patient_name: {cp.patient_first_name} {cp.patient_last_name}",
        f"patient_mrn: {cp.patient_mrn}",
        f"referring_provider: {cp.referring_provider}",
        f"referring_provider_cpso: {cp.referring_provider_npi}",
        f"primary_diagnosis: {cp.primary_diagnosis}",
        f"additional_diagnoses: {cp.additional_diagnoses or 'None'}",
        f"medication_name: {cp.medication_name}",
        f"medication_history: {cp.medication_history or 'None'}",
        f"patient_records: {cp.patient_records or 'None provided'}",
    ]
    return "\n".join(lines)


def _ground_truth_facts(cp) -> list[str]:
    """
    Explicit facts the care plan should cover.
    Used as the denominator for Recall.
    """
    facts = [
        cp.patient_first_name + " " + cp.patient_last_name,
        cp.patient_mrn,
        cp.primary_diagnosis,
        cp.medication_name,
    ]
    if cp.additional_diagnoses:
        facts += [d.strip() for d in cp.additional_diagnoses.splitlines() if d.strip()]
    if cp.medication_history:
        facts += [m.strip() for m in cp.medication_history.splitlines() if m.strip()]
    return [f for f in facts if f and f != "None"]


def _compute_recall(verified_claims: list[str], ground_truth: list[str]) -> float:
    """
    A ground-truth fact is 'covered' if any VERIFIED claim text contains it
    (case-insensitive substring match).
    """
    if not ground_truth:
        return 1.0
    covered = sum(
        1
        for fact in ground_truth
        if any(fact.lower() in claim.lower() for claim in verified_claims)
    )
    return covered / len(ground_truth)


def run_judge(cp, client: anthropic.Anthropic) -> JudgeReport:
    """Call the LLM judge and return a structured JudgeReport."""
    patient_data = _build_patient_data(cp)
    prompt = JUDGE_PROMPT.format(patient_data=patient_data, care_plan=cp.plan)

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text.strip()

    # Strip accidental markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    data = json.loads(raw)
    claims = data.get("claims", [])

    counts = {"VERIFIED": 0, "UNVERIFIED": 0, "HALLUCINATION": 0}
    for c in claims:
        v = c.get("verdict", "UNVERIFIED").upper()
        if v in counts:
            counts[v] += 1
        else:
            counts["UNVERIFIED"] += 1
            c["verdict"] = "UNVERIFIED"

    verified = counts["VERIFIED"]
    hallucination = counts["HALLUCINATION"]
    denominator = verified + hallucination
    precision = verified / denominator if denominator > 0 else 1.0

    verified_texts = [c["text"] for c in claims if c.get("verdict") == "VERIFIED"]
    ground_truth = _ground_truth_facts(cp)
    recall = _compute_recall(verified_texts, ground_truth)

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return JudgeReport(
        claims=claims,
        verified=verified,
        unverified=counts["UNVERIFIED"],
        hallucination=hallucination,
        precision=precision,
        recall=recall,
        f1=f1,
        has_hallucination=hallucination > 0,
    )
