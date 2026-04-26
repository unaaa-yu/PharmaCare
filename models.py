from datetime import datetime
from sqlalchemy import Integer, String, Text, DateTime
from sqlalchemy.orm import mapped_column, Mapped
from database import Base


class CarePlan(Base):
    __tablename__ = "careplans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_name: Mapped[str] = mapped_column(String(200))
    age: Mapped[int] = mapped_column(Integer)
    conditions: Mapped[str] = mapped_column(Text)
    medications: Mapped[str] = mapped_column(Text)
    goals: Mapped[str] = mapped_column(Text)
    plan: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
