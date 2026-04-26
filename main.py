import os
import anthropic
from fastapi import FastAPI, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import Base, engine, get_db
from models import CarePlan

Base.metadata.create_all(bind=engine)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


class PatientInput(BaseModel):
    patient_name: str
    age: int
    conditions: str
    medications: str
    goals: str


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.post("/generate/")
def generate(data: PatientInput, db: Session = Depends(get_db)):
    prompt = f"""You are a clinical pharmacist. Generate a detailed medication care plan for this patient.

Patient: {data.patient_name}, Age: {data.age}
Diagnoses: {data.conditions}
Current Medications: {data.medications}
Patient Goals: {data.goals}

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
        patient_name=data.patient_name,
        age=data.age,
        conditions=data.conditions,
        medications=data.medications,
        goals=data.goals,
        plan=plan_text,
    )
    db.add(record)
    db.commit()

    return {"id": record.id, "plan": plan_text}
