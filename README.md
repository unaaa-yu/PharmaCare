# PharmaCare AI

An AI-powered medication care plan generator for clinical pharmacists. Accepts patient diagnoses and medication data, generates structured care plans via Claude LLM, detects drug-drug interactions using TF-IDF similarity, stratifies patient risk with a PyTorch neural network, and audits every claim for hallucinations using an LLM-as-Judge pipeline.

---

### Tech Stack

| Tech | Version | Purpose |
|------|---------|---------|
| FastAPI | 0.115 | REST API framework |
| Claude (claude-sonnet-4-6) | — | Care plan generation + LLM-as-Judge |
| PyTorch | 2.x | Patient risk stratification MLP |
| scikit-learn | 1.4+ | TF-IDF drug name fuzzy matching |
| SQLAlchemy | 2.x | ORM for care plan persistence |
| PostgreSQL | 16 | Primary database (SQLite for local dev) |
| pypdf | 4.x | PDF patient record extraction |
| Docker Compose | — | Local Postgres + app orchestration |

---

### Features

- **Care Plan Generation** — Claude generates a structured 6-section medication care plan tagged with source labels (`[PATIENT]`, `[GUIDELINE]`, `[INFERRED]`) for full clinical transparency
- **LLM-as-Judge** — A second Claude call audits every claim in the generated plan against the original patient data; reports Precision / Recall / F1 and flags hallucinations
- **Drug-Drug Interaction (DDI) Scorer** — scikit-learn TF-IDF character n-gram model fuzzy-matches patient medications against a curated 36-pair knowledge base; returns HIGH / MODERATE / LOW risk pairs with mechanism descriptions
- **Patient Risk Stratification** — PyTorch MLP (6 → 32 → 16 → 3) classifies patients into LOW / MODERATE / HIGH risk using ICD-10 diagnosis complexity, polypharmacy burden, and drug risk weights; MC Dropout used for uncertainty estimation
- **PDF Upload** — Patient records can be submitted as a PDF; text is extracted with pypdf
- **Frontend UI** — Single-page HTML/CSS/JS interface with risk score badge, DDI warnings panel, and collapsible verification report

---

### Project Structure

```
PharmaCare/
├── main.py                # FastAPI app — /generate/ endpoint
├── models.py              # SQLAlchemy CarePlan model
├── database.py            # Engine + session factory
├── judge.py               # LLM-as-Judge pipeline (P/R/F1)
├── verify_careplan.py     # Rule-based code verification (ICD-10, dose regex)
├── ml/
│   ├── __init__.py
│   ├── ddi_scorer.py      # TF-IDF drug interaction scorer (36-pair KB)
│   └── risk_score.py      # PyTorch MLP risk stratification
├── public/
│   └── index.html         # Frontend UI
├── docker-compose.yml     # Postgres 16 on port 5433
├── Dockerfile
└── requirements.txt
```

---

### ML Architecture

#### Drug-Drug Interaction Scorer (`ml/ddi_scorer.py`)
- `TfidfVectorizer` with character n-grams (2–4) fits over all known drug names
- Each token in the patient's medication string is matched via cosine similarity (threshold 0.45)
- All matched drug pairs are checked against the 36-pair knowledge base
- Results sorted: HIGH → MODERATE → LOW

#### Patient Risk Stratification (`ml/risk_score.py`)

```
Input features (6):
  num_diagnoses · num_medications · icd_risk_sum
  drug_risk · has_records · polypharmacy

Architecture:
  Linear(6 → 32) + ReLU + Dropout(0.3)
  Linear(32 → 16) + ReLU + Dropout(0.3)
  Linear(16 → 3)  → Softmax → LOW / MODERATE / HIGH
```

Trained on 800 synthetic patients with domain-weighted labels (drug risk weighted 55%). MC Dropout (30 forward passes averaged) provides calibrated confidence scores.

#### LLM-as-Judge (`judge.py`)
Each claim in the generated plan receives one of three verdicts:
- `VERIFIED` — directly supported by the patient input
- `UNVERIFIED` — clinically plausible but not confirmable
- `HALLUCINATION` — contradicts or fabricates patient-specific facts

Metrics: **Precision** = VERIFIED / (VERIFIED + HALLUCINATION) · **Recall** = ground-truth facts covered · **F1** = harmonic mean

---

### Getting Started

**1. Clone and install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure environment**
```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
DATABASE_URL=postgresql://pharmcare:pharmcare@localhost:5433/pharmcare
```

**3. Start Postgres (Docker)**
```bash
docker-compose up -d
```

**4. Run the server**
```bash
uvicorn main:app --reload   # http://localhost:8000
```

**Local dev without Docker (SQLite)**
```bash
DATABASE_URL=sqlite:///./pharmcare.db uvicorn main:app --reload
```

---

### API

#### `POST /generate/`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `patient_first_name` | string | ✓ | |
| `patient_last_name` | string | ✓ | |
| `patient_mrn` | string | ✓ | 6-digit MRN |
| `referring_provider` | string | ✓ | |
| `referring_provider_npi` | string | ✓ | CPSO number |
| `primary_diagnosis` | string | ✓ | ICD-10 code + description |
| `medication_name` | string | ✓ | Current medication |
| `additional_diagnoses` | string | — | Newline-separated ICD-10 codes |
| `medication_history` | string | — | Newline-separated past medications |
| `patient_records_text` | string | — | Clinical notes |
| `patient_records_file` | file | — | PDF upload (alternative to text) |

**Response**
```json
{
  "id": 42,
  "plan": "1. Medication Review...",
  "risk": {
    "tier": "HIGH",
    "score": 0.99,
    "factors": ["Polypharmacy (4 medications)", "High-risk medication in regimen"],
    "color": "#dc2626"
  },
  "drug_interactions": [
    {
      "drug_a": "warfarin",
      "drug_b": "aspirin",
      "risk": "HIGH",
      "mechanism": "Additive anticoagulation — major bleeding risk",
      "confidence": 1.0
    }
  ],
  "verification": {
    "status": "verified",
    "metrics": { "precision": 1.0, "recall": 0.875, "f1": 0.933 },
    "claims": [ { "text": "...", "verdict": "VERIFIED", "reason": "..." } ]
  }
}
```
