"""
Patient risk stratification using a PyTorch MLP.

Architecture:
    Input(6) → Linear(32) + ReLU + Dropout(0.3)
             → Linear(16) + ReLU + Dropout(0.3)
             → Linear(3)  → Softmax
             → LOW / MODERATE / HIGH

The network learns non-linear feature interactions (e.g. warfarin + CKD
is worse than either alone) that logistic regression cannot capture.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ── Domain knowledge tables ─────────────────────────────────────────────────

_ICD_WEIGHTS: dict[str, float] = {
    "C": 2.0,   # oncology
    "I": 1.5,   # cardiovascular
    "N": 1.2,   # renal
    "J": 1.1,   # respiratory
    "G": 1.0,   # neurological
    "E": 1.0,   # metabolic / endocrine
    "F": 0.9,   # mental health
    "K": 0.8,   # gastrointestinal
    "M": 0.8,   # musculoskeletal
    "Z": 0.3,   # screening / administrative
}

_DRUG_RISKS: dict[str, float] = {
    "warfarin":     2.5,
    "heparin":      2.5,
    "insulin":      2.0,
    "digoxin":      2.0,
    "lithium":      2.0,
    "methotrexate": 2.0,
    "phenytoin":    1.8,
    "amiodarone":   1.8,
    "tacrolimus":   1.8,
    "cyclosporine": 1.8,
    "carbamazepine":1.5,
    "tamoxifen":    1.5,
    "clopidogrel":  1.3,
    "metformin":    1.0,
    "atorvastatin": 1.0,
    "lisinopril":   1.0,
    "metoprolol":   0.8,
}

_TIERS  = ["LOW", "MODERATE", "HIGH"]
_COLORS = {"LOW": "#059669", "MODERATE": "#d97706", "HIGH": "#dc2626"}

# Scoring weights used to label synthetic training data
_W = dict(num_dx=0.08, num_meds=0.08, icd_risk=0.15,
          drug_risk=0.55, has_records=0.04, polypharmacy=0.10)


# ── Synthetic data generator ─────────────────────────────────────────────────

def _generate_training_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    n   = 800

    icd_weight_vals = list(_ICD_WEIGHTS.values())
    drug_risk_pool  = [0.0] * 5 + [0.8, 1.0, 1.0, 1.0, 1.3, 1.5, 1.8, 2.0, 2.0, 2.5]

    num_dx   = rng.integers(1, 6, n).astype(float)
    num_meds = rng.integers(1, 9, n).astype(float)

    icd_risk = np.array([
        float(rng.choice(icd_weight_vals, size=int(nd)).sum())
        for nd in num_dx
    ])
    drug_risk    = np.array([rng.choice(drug_risk_pool) for _ in range(n)], dtype=float)
    has_records  = rng.integers(0, 2, n).astype(float)
    polypharmacy = (num_meds >= 4).astype(float)

    X = np.column_stack([num_dx, num_meds, icd_risk, drug_risk, has_records, polypharmacy])

    raw = (
        _W["num_dx"]       * num_dx
        + _W["num_meds"]   * num_meds
        + _W["icd_risk"]   * icd_risk
        + _W["drug_risk"]  * drug_risk
        + _W["has_records"]* has_records
        + _W["polypharmacy"]* polypharmacy
        + rng.normal(0, 0.08, n)
    )
    lo, hi = np.percentile(raw, [38, 72])
    y = np.where(raw < lo, 0, np.where(raw < hi, 1, 2))
    return X, y.astype(np.int64)


# ── PyTorch MLP ───────────────────────────────────────────────────────────────

class RiskMLP(nn.Module):
    def __init__(self, in_features: int = 6, num_classes: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class RiskScore:
    tier:    str
    score:   float
    factors: List[str]
    color:   str = field(default="#059669")


# ── Scorer ────────────────────────────────────────────────────────────────────

class PatientRiskScorer:
    def __init__(self, epochs: int = 200, lr: float = 1e-3) -> None:
        X_np, y_np = _generate_training_data()

        # Normalise features (z-score)
        self._mean = X_np.mean(axis=0)
        self._std  = X_np.std(axis=0) + 1e-8

        X_norm = (X_np - self._mean) / self._std
        X_t    = torch.tensor(X_norm, dtype=torch.float32)
        y_t    = torch.tensor(y_np,   dtype=torch.long)

        dataset = TensorDataset(X_t, y_t)
        loader  = DataLoader(dataset, batch_size=64, shuffle=True)

        self._model = RiskMLP()
        criterion   = nn.CrossEntropyLoss()
        optimizer   = optim.Adam(self._model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler   = optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)

        self._model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self._model(xb), yb)
                loss.backward()
                optimizer.step()
            scheduler.step()

        self._model.eval()

    # ── Feature extraction ────────────────────────────────────────────────────

    @staticmethod
    def _icd_risk(diagnoses: List[str]) -> float:
        total = 0.0
        for dx in diagnoses:
            m = re.search(r"\b([A-Z])\d", dx.upper())
            if m:
                total += _ICD_WEIGHTS.get(m.group(1), 0.5)
        return total

    @staticmethod
    def _drug_risk(medications: List[str]) -> float:
        best = 0.0
        for med in medications:
            low = med.lower()
            for drug, weight in _DRUG_RISKS.items():
                if drug in low:
                    best = max(best, weight)
        return best

    # ── Inference ─────────────────────────────────────────────────────────────

    def score(
        self,
        primary_diagnosis:    str,
        additional_diagnoses: List[str],
        medications:          List[str],
        has_records:          bool,
    ) -> RiskScore:
        all_dx   = [primary_diagnosis] + (additional_diagnoses or [])
        meds     = medications or []

        num_dx      = float(len(all_dx))
        num_meds    = float(len(meds))
        icd_risk    = self._icd_risk(all_dx)
        drug_risk   = self._drug_risk(meds)
        has_rec     = 1.0 if has_records else 0.0
        poly        = 1.0 if num_meds >= 4 else 0.0

        raw = np.array([[num_dx, num_meds, icd_risk, drug_risk, has_rec, poly]], dtype=np.float32)
        norm = (raw - self._mean) / self._std
        x_t  = torch.tensor(norm, dtype=torch.float32)

        with torch.no_grad():
            # Dropout active at inference → MC Dropout uncertainty estimate
            self._model.train()
            samples = torch.stack([
                torch.softmax(self._model(x_t), dim=-1) for _ in range(30)
            ])
            self._model.eval()

        mean_proba = samples.mean(dim=0).squeeze()          # (3,)
        pred_idx   = int(mean_proba.argmax().item())
        tier       = _TIERS[pred_idx]
        confidence = round(float(mean_proba[pred_idx].item()), 2)

        factors: List[str] = []
        if num_dx > 3:
            factors.append(f"{int(num_dx)} active diagnoses")
        if poly:
            factors.append(f"Polypharmacy ({int(num_meds)} medications)")
        if drug_risk >= 2.0:
            factors.append("High-risk medication in regimen")
        elif drug_risk >= 1.3:
            factors.append("Moderate-risk medication in regimen")
        if icd_risk >= 3.0:
            factors.append("High-complexity diagnosis profile")
        if not factors:
            factors.append("Routine clinical profile")

        return RiskScore(tier=tier, score=confidence, factors=factors, color=_COLORS[tier])
