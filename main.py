import io
import os
import anthropic
import pypdf
from fastapi import FastAPI, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from typing import Optional

from database import Base, engine, get_db
from models import CarePlan

Base.metadata.create_all(bind=engine)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


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

    prompt = f"""You are a clinical pharmacist. Generate a detailed medication care plan for the following patient.

Patient: {patient_first_name} {patient_last_name}
MRN: {patient_mrn}
Referring Provider: {referring_provider} (CPSO: {referring_provider_npi})

Primary Diagnosis (ICD-10): {primary_diagnosis}
Additional Diagnoses (ICD-10): {", ".join(additional_diagnoses_list) if additional_diagnoses_list else "None"}

Current Medication: {medication_name}
Medication History: {chr(10).join(f"- {m}" for m in medication_history_list) if medication_history_list else "None"}

Patient Records / Notes:
{patient_records if patient_records else "None provided"}

Provide a structured care plan including:
1. Medication review and recommendations
2. Drug interaction concerns
3. Dosing schedule
4. Monitoring parameters
5. Patient education points
6. Follow-up recommendations"""

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
