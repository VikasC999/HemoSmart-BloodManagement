"""
HemoSmart - ORM Models
--------------------------
Replaces agents/donors.json and agents/inventory.json with real tables,
and adds durable storage for patients/predictions (previously not
persisted at all -- every Day 1 prediction vanished after the request).

Blood stock is tracked per physical unit (`blood_units`): a unit has a
group, a component, an expiry, and a lifecycle -- available, reserved
for a transfusion, issued, expired (wastage), or transferred to a
partner hospital (wastage avoided). Counting units by status over time
is how HemoSmart measures the wastage it exists to reduce.
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
    # "whole_blood" | "apheresis"; None (older rows) is treated as whole blood
    last_donation_type = Column(String, nullable=True)
    alerts_sent = Column(Integer, nullable=False, default=0)
    alerts_responded = Column(Integer, nullable=False, default=0)
    # SIMULATION ONLY: each synthetic donor's hidden true probability of
    # answering an alert, used by the simulator to decide whether they
    # respond. The bandit never reads it -- it has to discover it from
    # alerts_sent/alerts_responded, exactly as it would with real donors.
    simulated_response_rate = Column(Float, nullable=True)

    alerts = relationship("DonorAlert", back_populates="donor")


class DonorAlert(Base):
    __tablename__ = "donor_alerts"

    id = Column(Integer, primary_key=True)
    donor_id = Column(String, ForeignKey("donors.donor_id"), nullable=False)
    blood_type = Column(String, nullable=False)
    component = Column(String, nullable=True)  # which shortage the alert answers
    sent_at = Column(DateTime, default=datetime.utcnow)
    method = Column(String, default="log")  # print/log stub today; a real
    # dispatch API (Twilio/WhatsApp) would set this to "sms"/"whatsapp"
    responded = Column(Boolean, nullable=True)  # None = still pending
    responded_at = Column(DateTime, nullable=True)
    cycle_id = Column(Integer, ForeignKey("agent_cycles.id"), nullable=True)

    donor = relationship("Donor", back_populates="alerts")


class BloodUnit(Base):
    __tablename__ = "blood_units"

    id = Column(Integer, primary_key=True)
    unit_id = Column(String, unique=True, nullable=False, index=True)  # e.g. "PLT-000123"
    blood_group = Column(String, nullable=False, index=True)
    component = Column(String, nullable=False, index=True)
    collected_at = Column(DateTime, nullable=False)
    expires_at = Column(DateTime, nullable=False, index=True)
    # available | reserved | issued | expired | transferred | discarded
    status = Column(String, nullable=False, default="available", index=True)
    status_changed_at = Column(DateTime, nullable=True)
    source = Column(String, default="stock")  # stock | donation | seed
    donor_id = Column(String, ForeignKey("donors.donor_id"), nullable=True)
    transfusion_id = Column(Integer, ForeignKey("transfusions.id"), nullable=True)
    transferred_to = Column(String, nullable=True)


class Transfusion(Base):
    """A request for blood for one patient: units are reserved FEFO on
    request, then issued on confirmation (or released on cancellation)."""
    __tablename__ = "transfusions"

    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=True)
    recipient_group = Column(String, nullable=False)
    status = Column(String, nullable=False, default="reserved")  # reserved | issued | cancelled
    requested = Column(JSON, nullable=False)    # {component: units asked for}
    shortfall = Column(JSON, nullable=True)     # {component: units stock couldn't cover}
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True)

    units = relationship("BloodUnit")


class AgentCycle(Base):
    """One run of the autonomous loop (agents/autonomy): what each agent
    observed, decided and did, so every automated action is traceable."""
    __tablename__ = "agent_cycles"

    id = Column(Integer, primary_key=True)
    mode = Column(String, nullable=False)  # live | simulation
    trigger = Column(String, nullable=False)  # scheduler | manual | simulation
    clock = Column(DateTime, nullable=False)  # the "now" the cycle reasoned about
    started_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
    status = Column(String, nullable=False, default="running")  # ok | degraded | failed
    summary = Column(JSON, nullable=True)
    events = Column(JSON, nullable=True)
    brief = Column(Text, nullable=True)


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True)
    hemoglobin = Column(Float, nullable=False)
    platelets = Column(Integer, nullable=False)
    inr = Column(Float, nullable=False)
    age = Column(Integer, nullable=False)
    surgery_type = Column(String, nullable=False)
    blood_group = Column(String, nullable=True)
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
    components = Column(JSON, nullable=True)   # [{component, probability, units}]
    urgency = Column(String, nullable=True)
    shap = Column(JSON, nullable=True)         # {component: {feature: contribution}}
    explanation = Column(Text, default="")
    citations = Column(JSON, nullable=True)
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
