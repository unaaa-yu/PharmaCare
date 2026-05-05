import io
import os
import anthropic
import pypdf
from fastapi import FastAPI, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from mangum import Mangum
from sqlalchemy.orm import Session
from typing import Optional

from database import Base, engine, get_db
from models import CarePlan

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield

app = FastAPI(lifespan=lifespan)

is_vercel = os.environ.get("VERCEL")
if not is_vercel:
    app.mount("/static", StaticFiles(directory="public"), name="static")

    @app.get("/")
    def index():
        return FileResponse("public/index.html")


@app.post("/generate/")
async def generate(
    patient_first_name: str = Form(...),
    patient_last_name: str = Form(...),
    patient_mrn: str = Form(...),
    referring_provider: str = Form(...),
    referring_provider_npi: str = Form(...),
    primary_diagnosis: str = Form(...),
    medication_name: str = Form(...),
    additional_diagnoses: str = Form(""),
    medication_history: str = Form(""),
    patient_records_text: str = Form(""),
    patient_records_file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
):
    if patient_records_file and patient_records_file.filename:
        content = await patient_records_file.read()
        reader = pypdf.PdfReader(io.BytesIO(content))
        patient_records = "\n".join(
            page.extract_text() for page in reader.pages if page.extract_text()
        )
    else:
        patient_records = patient_records_text

    additional_diagnoses_list = [
        d.strip() for d in additional_diagnoses.splitlines() if d.strip()
    ]
    medication_history_list = [
        m.strip() for m in medication_history.splitlines() if m.strip()
    ]

    prompt = f"""You are a clinical pharmacist generating a medication care plan. Your most important task is transparency: every clinical statement must carry a source tag so the reviewing pharmacist knows exactly where each piece of information came from.

## Source tags — use exactly these three, nothing else:
- [PATIENT] — directly stated in the patient data or records supplied below. Only use this if the fact is explicitly present in the input.
- [GUIDELINE] — standard clinical practice supported by established guidelines (e.g. CPS, CPSO, Diabetes Canada, ACC/AHA). Only use this when you are confident the recommendation is a well-established, widely accepted standard — not just common practice.
- [INFERRED] — your clinical reasoning, extrapolation, or any statement not directly traceable to the patient data or a specific guideline. When in doubt, use [INFERRED] rather than [GUIDELINE].

## Tagging rules:
1. Tag every individual statement or bullet point — do not leave any line untagged.
2. If a single sentence combines patient data and your reasoning, tag it [INFERRED].
3. Prefer [INFERRED] over [GUIDELINE] whenever you are not certain of the specific guideline source.
4. Do not invent patient details that are not in the input. If information is missing, say "Not provided [PATIENT]" rather than assuming.

## Patient data (treat this as the only ground truth for [PATIENT] tags):
Patient: {patient_first_name} {patient_last_name}
MRN: {patient_mrn}
Referring Provider: {referring_provider} (CPSO: {referring_provider_npi})
Primary Diagnosis (ICD-10): {primary_diagnosis}
Additional Diagnoses (ICD-10): {", ".join(additional_diagnoses_list) if additional_diagnoses_list else "None"}
Current Medication: {medication_name}
Medication History: {chr(10).join(f"- {m}" for m in medication_history_list) if medication_history_list else "None"}
Patient Records / Notes:
{patient_records if patient_records else "None provided"}

## Output format:
Generate a structured care plan with the following six sections. Each bullet point must end with its source tag in square brackets.

1. Medication Review and Recommendations
2. Drug Interaction Concerns
3. Dosing Schedule
4. Monitoring Parameters
5. Patient Education Points
6. Follow-up Recommendations"""

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    plan_text = message.content[0].text

    record = CarePlan(
        patient_first_name=patient_first_name,
        patient_last_name=patient_last_name,
        patient_mrn=patient_mrn,
        referring_provider=referring_provider,
        referring_provider_npi=referring_provider_npi,
        primary_diagnosis=primary_diagnosis,
        medication_name=medication_name,
        additional_diagnoses=additional_diagnoses,
        medication_history=medication_history,
        patient_records=patient_records,
        plan=plan_text,
    )
    db.add(record)
    db.commit()

    return {"id": record.id, "plan": plan_text}


handler = Mangum(app)
