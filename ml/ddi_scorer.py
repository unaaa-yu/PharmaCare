"""
Drug-Drug Interaction scorer.
Uses TF-IDF character n-gram similarity to fuzzy-match patient medication
strings against a curated knowledge base, then checks all matched drug pairs
for known interactions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class DDIResult:
    drug_a: str
    drug_b: str
    risk: str          # HIGH | MODERATE | LOW
    mechanism: str
    confidence: float  # min cosine sim of the two matched drugs


# 36-pair curated knowledge base
_KB = [
    ("warfarin",      "aspirin",        "HIGH",     "Additive anticoagulation — major bleeding risk"),
    ("warfarin",      "ibuprofen",      "HIGH",     "NSAIDs displace warfarin; potentiate GI and systemic bleeding"),
    ("warfarin",      "naproxen",       "HIGH",     "NSAID-warfarin synergy markedly raises GI bleed risk"),
    ("warfarin",      "amoxicillin",    "MODERATE", "Antibiotics reduce gut-flora vitamin K → enhanced anticoagulation"),
    ("warfarin",      "phenytoin",      "HIGH",     "Complex bidirectional CYP interaction; narrow TI for both drugs"),
    ("warfarin",      "carbamazepine",  "HIGH",     "CYP induction by carbamazepine reduces warfarin levels unpredictably"),
    ("metformin",     "furosemide",     "MODERATE", "Furosemide raises metformin plasma levels; lactic acidosis risk"),
    ("metformin",     "ciprofloxacin",  "LOW",      "Fluoroquinolones can disrupt glucose homeostasis"),
    ("atorvastatin",  "clarithromycin", "HIGH",     "CYP3A4 inhibition → statin overexposure → myopathy / rhabdomyolysis"),
    ("simvastatin",   "clarithromycin", "HIGH",     "CYP3A4 inhibition → severe myopathy risk"),
    ("simvastatin",   "amlodipine",     "MODERATE", "Amlodipine raises simvastatin AUC ~77 %; cap dose at 20 mg"),
    ("lisinopril",    "spironolactone", "HIGH",     "ACE inhibitor + K⁺-sparing diuretic → life-threatening hyperkalemia"),
    ("lisinopril",    "potassium",      "MODERATE", "ACE inhibitor + K⁺ supplement → hyperkalemia"),
    ("metoprolol",    "verapamil",      "HIGH",     "Additive AV block and profound bradycardia"),
    ("metoprolol",    "diltiazem",      "MODERATE", "Synergistic negative chronotropy and dromotropy"),
    ("metoprolol",    "insulin",        "MODERATE", "Beta-blockers mask tachycardia that signals hypoglycemia"),
    ("citalopram",    "tramadol",       "HIGH",     "Serotonin syndrome risk"),
    ("sertraline",    "tramadol",       "HIGH",     "Serotonin syndrome"),
    ("fluoxetine",    "tramadol",       "HIGH",     "Serotonin syndrome"),
    ("fluoxetine",    "tamoxifen",      "HIGH",     "CYP2D6 inhibition reduces tamoxifen → endoxifen conversion"),
    ("paroxetine",    "tamoxifen",      "HIGH",     "Strong CYP2D6 inhibitor markedly reduces tamoxifen efficacy"),
    ("clopidogrel",   "omeprazole",     "MODERATE", "CYP2C19 inhibition reduces clopidogrel bioactivation"),
    ("aspirin",       "ibuprofen",      "MODERATE", "Ibuprofen competitively antagonizes aspirin's antiplatelet effect"),
    ("insulin",       "furosemide",     "MODERATE", "Loop diuretics impair insulin secretion → hyperglycemia"),
    ("digoxin",       "furosemide",     "HIGH",     "Electrolyte depletion (K⁺/Mg²⁺) amplifies digoxin toxicity"),
    ("digoxin",       "amiodarone",     "HIGH",     "Amiodarone raises digoxin levels; narrow TI → toxicity"),
    ("levothyroxine", "calcium",        "LOW",      "Calcium chelates levothyroxine, reducing GI absorption"),
    ("levothyroxine", "iron",           "LOW",      "Iron reduces levothyroxine absorption; separate by ≥4 hours"),
    ("prednisone",    "insulin",        "MODERATE", "Corticosteroids raise blood glucose, counteracting insulin"),
    ("prednisone",    "ibuprofen",      "HIGH",     "Additive GI mucosal damage — ulceration risk"),
    ("lithium",       "ibuprofen",      "HIGH",     "NSAIDs reduce renal lithium clearance → toxicity"),
    ("lithium",       "naproxen",       "HIGH",     "NSAIDs reduce lithium excretion → toxicity"),
    ("sildenafil",    "nitroglycerin",  "HIGH",     "PDE5 inhibitor + nitrate → severe hypotension (absolute CI)"),
    ("sildenafil",    "isosorbide",     "HIGH",     "Nitrate + PDE5 inhibitor → profound, refractory hypotension"),
    ("methotrexate",  "ibuprofen",      "HIGH",     "NSAIDs reduce methotrexate renal clearance → toxicity"),
    ("phenytoin",     "carbamazepine",  "MODERATE", "Unpredictable bidirectional CYP effects on both drug levels"),
]

_RISK_ORDER = {"HIGH": 0, "MODERATE": 1, "LOW": 2}


class DDIScorer:
    def __init__(self, threshold: float = 0.45):
        self.threshold = threshold
        # Unique drug names across the KB
        self._drug_names: List[str] = list({d for row in _KB for d in row[:2]})
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), lowercase=True
        )
        self._drug_matrix = self._vectorizer.fit_transform(self._drug_names)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _best_match(self, token: str) -> Optional[tuple[str, float]]:
        vec = self._vectorizer.transform([token])
        sims = cosine_similarity(vec, self._drug_matrix)[0]
        idx = int(np.argmax(sims))
        score = float(sims[idx])
        if score >= self.threshold:
            return self._drug_names[idx], score
        return None

    def _extract_drugs(self, med_strings: List[str]) -> dict[str, float]:
        matched: dict[str, float] = {}
        for med in med_strings:
            for token in re.split(r"[\s,;/()\-]+", med.lower()):
                if len(token) < 4:
                    continue
                result = self._best_match(token)
                if result:
                    name, conf = result
                    if name not in matched or matched[name] < conf:
                        matched[name] = conf
        return matched

    # ── Public API ──────────────────────────────────────────────────────────

    def score(self, medications: List[str]) -> List[DDIResult]:
        """
        medications: raw medication strings from the patient (e.g. ["Warfarin 5mg daily"]).
        Returns a list of DDIResult sorted by risk severity.
        """
        found = self._extract_drugs(medications)
        results: List[DDIResult] = []
        seen: set = set()

        for drug_a, drug_b, risk, mechanism in _KB:
            if drug_a in found and drug_b in found:
                key = frozenset([drug_a, drug_b])
                if key not in seen:
                    seen.add(key)
                    results.append(
                        DDIResult(
                            drug_a=drug_a,
                            drug_b=drug_b,
                            risk=risk,
                            mechanism=mechanism,
                            confidence=round(min(found[drug_a], found[drug_b]), 2),
                        )
                    )

        results.sort(key=lambda r: _RISK_ORDER.get(r.risk, 3))
        return results
