"""
HemoSmart MCP Server
------------------------
Exposes HemoSmart's own capabilities -- prediction, explanation,
inventory checking, donor alerting -- as MCP tools, so any
MCP-compatible AI client (Claude Desktop, Claude Code, etc.) can
connect to HemoSmart directly and use it as a tool provider.

This is the correct use of MCP here: the protocol connects LLM
clients to tools. It is deliberately NOT used to represent an
external hospital system -- see integrations/mcp/ for that (a plain
adapter interface, the right pattern for system-to-system
integration, which MCP was never designed for).

Every tool below wraps an existing function in agents/tools.py rather
than reimplementing logic -- the MCP server is a second interface onto
the same prediction/RAG/donor pipeline the REST API and chat agent
already use, not a separate implementation of it.

Run standalone over stdio (for a local client config, e.g. Claude
Desktop's claude_desktop_config.json launching this as a subprocess):
    python -m mcp_server.stdio_main

Or reachable over HTTP: mounted into the FastAPI app at /mcp
(see backend/main.py) -- this is what a deployed HemoSmart exposes to
a remote MCP client.
"""

import os
import sys
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP

from agents.tools import (
    check_inventory as _check_inventory,
    generate_explanation as _generate_explanation,
    get_supply_outlook as _get_supply_outlook,
    predict_transfusion as _predict_transfusion,
    send_donor_alert as _send_donor_alert,
)
from schema.patient_schema import PatientRecord

mcp = FastMCP(
    name="HemoSmart",
    instructions=(
        "Blood transfusion prediction and supply coordination. Use "
        "predict_transfusion to assess whether a patient needs a "
        "transfusion from their lab values, explain_prediction for a "
        "WHO-guideline-grounded clinical explanation of that result, "
        "check_inventory for current blood stock levels, "
        "supply_outlook for forecast shortages and units at risk of "
        "expiring unused, and alert_donors to notify eligible donors "
        "for a blood type running low."
    ),
    # FastMCP's own default path is "/mcp" -- mounted at "/mcp" in
    # backend/main.py, that would put the real route at /mcp/mcp.
    # Setting it to "/" here means the mount point alone (/mcp) is the
    # full external path.
    streamable_http_path="/",
)


@mcp.tool()
def predict_transfusion(
    hemoglobin: float, platelets: int, INR: float, age: int, surgery_type: str,
    blood_group: Optional[str] = None,
) -> dict:
    """Predicts whether a patient needs a blood transfusion from their
    clinical values, using the trained per-component XGBoost models.

    surgery_type must be one of: Cardiac, Orthopedic, General, Emergency.
    Returns transfusion_needed, confidence (P(any transfusion)), the
    components needed (PRBC / Platelets / FFP) with doses, urgency, and
    per-feature SHAP contributions.
    """
    record = PatientRecord(
        hemoglobin=hemoglobin, platelets=platelets, INR=INR,
        age=age, surgery_type=surgery_type, blood_group=blood_group, source_format="mcp",
    )
    return _predict_transfusion(record)


@mcp.tool()
def explain_prediction(
    hemoglobin: float, platelets: int, INR: float, age: int, surgery_type: str,
    transfusion_needed: bool,
) -> str:
    """Explains a transfusion prediction against WHO clinical
    guidelines, via retrieval-augmented generation over the real WHO
    transfusion guideline document. Pass the same patient values used
    in predict_transfusion, plus the transfusion_needed it returned.
    """
    record = PatientRecord(
        hemoglobin=hemoglobin, platelets=platelets, INR=INR,
        age=age, surgery_type=surgery_type, source_format="mcp",
    )
    return _generate_explanation(record, transfusion_needed)


@mcp.tool()
def check_inventory(blood_type: Optional[str] = None, component: Optional[str] = None) -> dict:
    """Checks current unit-level blood stock. Pass a blood type (e.g. "O-")
    and optionally a component (PRBC, Platelets, FFP) to check a slice,
    or omit both to see every group x component with its par level,
    status (Sufficient / Low / Critical) and near-expiry count.
    """
    return _check_inventory(blood_type, component)


@mcp.tool()
def supply_outlook(days: int = 7) -> dict:
    """Forecasts demand per blood group and component for the next `days`
    days and projects, day by day against current stock, which will run
    short (and from when) and how many units will expire unused.
    """
    result = _get_supply_outlook(min(max(days, 1), 30))
    return {k: result[k] for k in ("as_of", "horizon_days", "model_used", "shortages", "wastage_risks")}


@mcp.tool()
def alert_donors(blood_type: str, component: str = "PRBC") -> str:
    """Alerts the most eligible, most-likely-to-respond donors for a
    blood type running low, selected via a Multi-Armed Bandit
    (Thompson Sampling) over each donor's response history. component
    (PRBC, Platelets, FFP) decides the donation asked for.
    """
    return _send_donor_alert(blood_type, component)
