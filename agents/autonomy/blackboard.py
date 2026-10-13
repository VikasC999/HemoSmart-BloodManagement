"""
The shared workspace the autonomous loop's agents coordinate through.

Agents never call each other. Each one reads what earlier agents posted
here, does its own job, and posts its results and an event log entry
for every observation, decision and action -- so one cycle's full
reasoning can be replayed from its event list afterwards (it's stored
on the AgentCycle row and shown in the dashboard).
"""

from dataclasses import dataclass, field
from datetime import date, datetime

EVENT_KINDS = ("observe", "decide", "act", "explain", "fallback", "error")


def _jsonable(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


@dataclass
class Blackboard:
    now: datetime
    cycle_id: int
    mode: str = "live"
    horizon_days: int = 7
    events: list = field(default_factory=list)

    # Posted by the agents, in pipeline order
    stock: dict = None             # InventoryAgent: stock summary after the expiry sweep
    expired_units: list = field(default_factory=list)
    donor_feedback: list = field(default_factory=list)  # DonorAgent: resolved alerts
    forecast_totals: dict = None   # ForecastAgent: {component: [{date, units, ...}]}
    forecast_by_key: dict = None   # ForecastAgent: {(group, component): [units/day]}
    projections: dict = None       # PlannerAgent: {(group, component): projection}
    shortages: list = field(default_factory=list)       # full-horizon projected shortages (reporting)
    needs: list = field(default_factory=list)           # actionable now: window gap + safety stock
    wastage_risks: list = field(default_factory=list)
    alerts: list = field(default_factory=list)          # DonorAgent: alerts sent
    escalations: list = field(default_factory=list)
    transfers: list = field(default_factory=list)       # LogisticsAgent
    brief: str = None                                   # ExplanationAgent

    def log(self, agent: str, kind: str, message: str, **data) -> dict:
        assert kind in EVENT_KINDS, kind
        event = {"agent": agent, "kind": kind, "message": message}
        if data:
            event["data"] = _jsonable(data)
        self.events.append(event)
        return event

    def events_of(self, kind: str) -> list:
        return [e for e in self.events if e["kind"] == kind]
