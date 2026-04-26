from datetime import datetime
from sqlalchemy import Integer, String, Text, DateTime
from sqlalchemy.orm import mapped_column, Mapped
from database import Base


class CarePlan(Base):
    __tablename__ = "careplans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_first_name: Mapped[str] = mapped_column(String(100))
    patient_last_name: Mapped[str] = mapped_column(String(100))
    patient_mrn: Mapped[str] = mapped_column(String(6))
    referring_provider: Mapped[str] = mapped_column(String(200))
    referring_provider_npi: Mapped[str] = mapped_column(String(10))
    primary_diagnosis: Mapped[str] = mapped_column(String(20))
    medication_name: Mapped[str] = mapped_column(String(200))
    additional_diagnoses: Mapped[str] = mapped_column(Text)
    medication_history: Mapped[str] = mapped_column(Text)
    patient_records: Mapped[str] = mapped_column(Text)
    plan: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
