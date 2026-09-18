"""
HemoSmart - ORM Models
--------------------------
Replaces agents/donors.json and agents/inventory.json with real tables,
and adds durable storage for patients/predictions (previously not
persisted at all -- every Day 1 prediction vanished after the request).

`users` and `audit_log` are defined now so the schema doesn't need to
change again on Day 3 (auth) / Day 5 (audit logging) -- they're just
not written to until those routes exist.
"""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from db.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=True)  # set once Day 3 auth lands
    role = Column(String, nullable=False, default="Hospital Staff")
    created_at = Column(DateTime, default=datetime.utcnow)


class Donor(Base):
    __tablename__ = "donors"

    id = Column(Integer, primary_key=True)
    donor_id = Column(String, unique=True, nullable=False, index=True)  # e.g. "D0001"
    name = Column(String, nullable=False)
    phone = Column(String, nullable=False)
    blood_type = Column(String, nullable=False, index=True)
    last_donation_date = Column(Date, nullable=False)
    alerts_sent = Column(Integer, nullable=False, default=0)
    alerts_responded = Column(Integer, nullable=False, default=0)

    alerts = relationship("DonorAlert", back_populates="donor")


class DonorAlert(Base):
    __tablename__ = "donor_alerts"

    id = Column(Integer, primary_key=True)
    donor_id = Column(String, ForeignKey("donors.donor_id"), nullable=False)
    blood_type = Column(String, nullable=False)
    sent_at = Column(DateTime, default=datetime.utcnow)
    method = Column(String, default="log")  # print/log stub today; a real
    # dispatch API (Twilio/WhatsApp) would set this to "sms"/"whatsapp"

    donor = relationship("Donor", back_populates="alerts")


class InventoryItem(Base):
    __tablename__ = "inventory"

    id = Column(Integer, primary_key=True)
    blood_type = Column(String, unique=True, nullable=False, index=True)
    units = Column(Integer, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True)
    hemoglobin = Column(Float, nullable=False)
    platelets = Column(Integer, nullable=False)
    inr = Column(Float, nullable=False)
    age = Column(Integer, nullable=False)
    surgery_type = Column(String, nullable=False)
    source_format = Column(String, default="unknown")
    source_hospital = Column(String, default="unknown")
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    predictions = relationship("Prediction", back_populates="patient", order_by="Prediction.created_at")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    transfusion_needed = Column(Boolean, nullable=False)
    confidence = Column(Float, nullable=False)
    explanation = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="predictions")


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String, nullable=False)
    resource_type = Column(String, nullable=True)
    resource_id = Column(String, nullable=True)
    ip_address = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    details = Column(JSON, nullable=True)


class ForecastRun(Base):
    __tablename__ = "forecasts"

    id = Column(Integer, primary_key=True)
    forecast_date = Column(DateTime, default=datetime.utcnow)
    model_used = Column(String, nullable=False)
    predicted_units_per_day = Column(JSON, nullable=False)
    average_daily_demand = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
