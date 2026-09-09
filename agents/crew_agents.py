"""
HemoSmart - Multi-Agent System (CrewAI)
--------------------------------------------
Matches the target architecture:

                    Orchestrator agent
                (Routes tasks, manages state)
        /        |         |         |         \\
  Prediction  Extraction   RAG    Inventory   Action
  (ML model)  (parses PDFs) (explains) (blood stock) (alerts)

DESIGN: GENUINELY AGENTIC, WITH THREE REAL SAFETY MECHANISMS
--------------------------------------------------------------
Earlier iterations of this file tried two extremes: letting agents
freely call live tools and trust their own free-text output (which
tested unreliable -- context corruption, hallucinated numbers,
duplicate/invented tool calls), and then over-correcting into
no-op/cached tools with post-hoc overwriting (which is safe but not
genuinely agentic -- the agents don't actually do anything).

This version keeps agents genuinely responsible for real work, and
addresses reliability with three targeted mechanisms instead:

  1. STRUCTURED OUTPUT (Pydantic via `output_pydantic`): each task's
     result must conform to a typed schema (PredictionResult,
     InventoryResult, ActionResult), not free text. A later agent
     receives structured data, not an ambiguous paragraph it has to
     correctly parse.

  2. IDEMPOTENT TOOLS: the donor-alert tool is the one place a
     duplicate call causes real-world harm (contacting a donor
     twice). It is made idempotent PER PIPELINE RUN -- calling it
     again for a blood type already alerted in this run is a no-op
     that returns a "duplicate suppressed" message instead of
     dispatching a second real alert. This makes the harmful failure
     mode structurally impossible, regardless of how many times an
     agent's own reasoning decides to call the tool.

  3. TRANSPARENT VALIDATION (not blind overwriting): after the crew
     runs, each agent's structured output is compared against an
     independently recomputed ground truth. If they match, the
     agent's own (genuine) output is used and logged as verified. If
     they don't match, the mismatch is logged explicitly and the
     verified value is used as a safety fallback. The agent's real
     work is the primary path; the ground-truth check is a safety net
     that only intervenes, visibly, when something is actually wrong.

Requires:
    pip install crewai pydantic

Run:
    python agents/crew_agents.py          (fixed pipeline demo)
    python agents/chat_agent.py           (interactive chat, routes dynamically)
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict
from pydantic import BaseModel, Field

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

from schema.patient_schema import PatientRecord
from agents.tools import (
    predict_transfusion,
    generate_explanation,
    check_inventory,
    send_donor_alert,
    extract_from_pdf,
)

agent_llm = LLM(model="ollama/llama3.1:8b", base_url="http://localhost:11434")


# ----------------------------------------------------------------------
# Structured output schemas -- agents' Final Answers are validated
# against these, not accepted as free text.
# ----------------------------------------------------------------------
class PredictionResult(BaseModel):
    transfusion_needed: bool
    confidence: float = Field(..., ge=0.0, le=1.0)


class ExplanationResult(BaseModel):
    explanation: str


class InventoryResult(BaseModel):
    inventory: Dict[str, int]
    low_stock_types: List[str]
    healthy_types: List[str]


class ActionResult(BaseModel):
    alerted_blood_types: List[str]
    no_action_blood_types: List[str]


class ExtractionResult(BaseModel):
    hemoglobin: float
    platelets: int
    inr: float
    age: int
    surgery_type: str


class OrchestratorResult(BaseModel):
    case_summary: str
    transfusion_needed: bool
    shortage_detected: bool
    workflow_status: str


# ----------------------------------------------------------------------
# Live tools. All are genuinely executed by their agent -- none are
# cached/no-op. The donor-alert tool is IDEMPOTENT PER PIPELINE RUN
# (see make_idempotent_action_tool) so a duplicate call cannot cause a
# duplicate real-world donor contact, regardless of agent behavior.
# ----------------------------------------------------------------------

@tool("Extract Patient Data From PDF")
def extract_pdf_tool(pdf_path: str) -> str:
    """Extracts patient hemoglobin, platelets, INR, age, and surgery
    type from an uploaded CBC lab report PDF."""
    record = extract_from_pdf(pdf_path)
    if record is None:
        return '{"error": "EXTRACTION_FAILED: route this patient to manual entry."}'
    return (
        f'{{"hemoglobin": {record.hemoglobin}, "platelets": {record.platelets}, '
        f'"inr": {record.INR}, "age": {record.age}, '
        f'"surgery_type": "{record.surgery_type}"}}'
    )


@tool("Predict Transfusion Need")
def predict_transfusion_tool(hemoglobin: float, platelets: int, inr: float,
                              age: int, surgery_type: str) -> str:
    """Predicts whether a patient needs a blood transfusion. Returns
    JSON-like text with transfusion_needed and confidence."""
    record = PatientRecord(
        hemoglobin=hemoglobin, platelets=platelets, INR=inr,
        age=age, surgery_type=surgery_type,
    )
    result = predict_transfusion(record)
    return f'{{"transfusion_needed": {result["transfusion_needed"]}, "confidence": {result["confidence"]}}}'


@tool("Explain Transfusion Prediction (RAG)")
def explain_prediction_tool(hemoglobin: float, platelets: int, inr: float,
                             age: int, surgery_type: str, prediction: bool) -> str:
    """Generates a clinical explanation for a transfusion prediction,
    grounded in WHO guideline text retrieved via RAG. The underlying
    pipeline (rag/explain.py) already computes threshold comparisons
    deterministically in Python -- only the final phrasing involves an
    LLM call, so the FACTS in this output are guaranteed correct even
    though exact wording may vary slightly between calls."""
    record = PatientRecord(
        hemoglobin=hemoglobin, platelets=platelets, INR=inr,
        age=age, surgery_type=surgery_type,
    )
    return generate_explanation(record, prediction)


@tool("Check Blood Inventory")
def check_inventory_tool() -> str:
    """Checks current blood inventory levels for all blood types and
    identifies which are low stock vs healthy."""
    result = check_inventory()
    low = list(result["low_stock_types"].keys())
    healthy = [bt for bt in result["inventory"] if bt not in low]
    return (
        f'{{"inventory": {result["inventory"]}, '
        f'"low_stock_types": {low}, "healthy_types": {healthy}}}'
    )


def make_idempotent_action_tool(verified_low_stock_types):
    """
    Factory for a donor-alert tool with TWO independent safety guarantees:

    1. VERIFIED-TYPE GUARD: the tool only ever dispatches a real alert
       for a blood type that is in `verified_low_stock_types` -- a list
       computed independently in Python from a fresh, real inventory
       check, BEFORE this tool exists. It does not matter what the
       Inventory or Action agents' own reasoning claims is low stock;
       if a blood type isn't genuinely verified low stock, this tool
       refuses to contact donors for it. This exists because testing
       showed the Inventory Agent can hallucinate fabricated blood
       types (e.g. schema field names used as fake data), which then
       caused the Action Agent to send real alerts to healthy blood
       types while missing a genuinely low one. Validating this AFTER
       the fact was not good enough -- real donors had already been
       contacted incorrectly by the time the mismatch was reported.
       This guard makes that specific harm structurally impossible.

    2. IDEMPOTENCY GUARD: each verified blood type can still only
       trigger one real alert per run, regardless of how many times
       the agent calls the tool for it.
    """
    already_alerted = set()

    @tool("Trigger Action (Donor Alert)")
    def action_tool(blood_type: str, units_needed: int = 0) -> str:
        """Sends a donor alert for a low-stock blood type. Only
        dispatches for blood types independently verified as low
        stock; requests for any other blood type are rejected. Safe
        to call more than once for the same blood type -- only the
        first valid call per blood type per run dispatches a real
        alert."""
        if blood_type not in verified_low_stock_types:
            return (
                f"REJECTED: {blood_type} is not verified as low stock. "
                f"No alert dispatched. Verified low-stock types this run "
                f"are: {sorted(verified_low_stock_types)}."
            )
        if blood_type in already_alerted:
            return (
                f"DUPLICATE SUPPRESSED: an alert for {blood_type} was "
                f"already sent earlier in this run. No new alert dispatched."
            )
        already_alerted.add(blood_type)
        return send_donor_alert(blood_type, units_needed if units_needed else None)

    return action_tool, already_alerted


# ----------------------------------------------------------------------
# The 5 specialist agents, matching the architecture diagram. These are
# genuinely responsible for calling their tools and producing
# structured output -- no caching, no post-hoc overwriting of content.
# ----------------------------------------------------------------------
prediction_agent = Agent(
    role="Prediction Agent",
    goal="Call the ML model exactly once to predict transfusion need, and report the exact result",
    backstory="Wraps the trained XGBoost model to produce transfusion predictions.",
    tools=[predict_transfusion_tool],
    llm=agent_llm,
    verbose=True,
    max_iter=3,
)

extraction_agent = Agent(
    role="Extraction Agent",
    goal="Parse uploaded CBC lab report PDFs into structured patient data",
    backstory="Converts unstructured lab report PDFs into the fields the Prediction Agent needs.",
    tools=[extract_pdf_tool],
    llm=agent_llm,
    verbose=True,
    max_iter=3,
)

rag_agent = Agent(
    role="RAG Agent",
    goal="Call the RAG explanation tool exactly once and report its result",
    backstory="Retrieves relevant WHO guideline passages and explains predictions in clinical terms.",
    tools=[explain_prediction_tool],
    llm=agent_llm,
    verbose=True,
    max_iter=3,
)

inventory_agent = Agent(
    role="Inventory Agent",
    goal="Call the inventory tool exactly once and report the exact low-stock and healthy blood types",
    backstory="Tracks blood bank inventory across all blood types.",
    tools=[check_inventory_tool],
    llm=agent_llm,
    verbose=True,
    max_iter=3,
)

action_agent = Agent(
    role="Action Agent",
    goal="Trigger a donor alert for each genuinely low-stock blood type, exactly once each",
    backstory="Executes follow-up actions -- donor alerts -- when the Inventory Agent flags a shortage.",
    tools=[],  # tools attached per-run (needs the idempotency tracker) -- see run_full_pipeline
    llm=agent_llm,
    verbose=True,
    max_iter=6,  # may need one call per low-stock type, plus a final answer
)

orchestrator_agent = Agent(
    role="Orchestrator Agent",
    goal="Consolidate the results of all specialist agents into one coherent case summary",
    backstory=(
        "Coordinates the full HemoSmart workflow -- reviews the outputs of "
        "the Extraction, Prediction, RAG, Inventory, and Action agents and "
        "produces a single consolidated summary of the case and system "
        "status. Does not repeat the specialists' work, only synthesizes it."
    ),
    tools=[],  # the Orchestrator has no tools of its own -- it reasons
               # over the other agents' already-produced structured results
    llm=agent_llm,
    verbose=True,
    max_iter=3,
)


def _independent_check(label, agent_value, ground_truth_value, verified_fallback):
    """
    Transparent validation: compares the agent's genuine structured
    output against an independently recomputed ground truth. Logs the
    result either way. Only substitutes the fallback if they actually
    disagree -- the agent's own correct output is used and reported as
    such whenever it matches, rather than being discarded on principle.
    """
    if agent_value == ground_truth_value:
        print(f"[VALIDATION] {label}: agent output matches ground truth. Using agent's result.")
        return agent_value, True
    else:
        print(f"[VALIDATION] {label}: MISMATCH detected.")
        print(f"    Agent produced:    {agent_value!r}")
        print(f"    Ground truth is:   {ground_truth_value!r}")
        print(f"    Falling back to verified value for safety.")
        return verified_fallback, False


def _run_isolated_task(label, agent, task, ground_truth_value, extract_fn, tools_for_agent=None):
    """
    Runs ONE task as its own single-task Crew, wrapped in try/except.
    This is the same pattern already used for Extraction (task0),
    extended to every task -- it exists because testing showed an
    agent's malformed structured output can raise an UNCAUGHT pydantic
    validation error deep inside CrewAI's task execution, which crashes
    the entire script and prevents every later task from running at
    all (this happened to Inventory, which corrupted the whole run).

    The agent still genuinely attempts the real work every time. If it
    succeeds and its output is extractable, extract_fn(task_output) is
    called to pull out the value to use. If ANYTHING goes wrong --
    a tool error, a malformed response, a hard pydantic crash -- the
    exception is caught here, logged, and ground_truth_value is used
    instead, so the pipeline always continues.
    """
    if tools_for_agent is not None:
        agent.tools = tools_for_agent
    try:
        mini_crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
        result = mini_crew.kickoff()
        value = extract_fn(result.tasks_output[0])
        if value is None:
            raise ValueError("Agent produced no usable structured output.")
        print(f"[VALIDATION] {label}: agent completed successfully, using its result.")
        return value, True, result
    except Exception as exc:
        print(f"[VALIDATION] {label}: FAILED ({exc.__class__.__name__}: {exc}). "
              f"Falling back to verified value for safety.")
        return ground_truth_value, False, None


def run_full_pipeline(pdf_path):
    """
    Full 6-agent pipeline: Extraction -> Prediction -> RAG -> Inventory
    -> Action -> Orchestrator. Every task runs as its own isolated
    mini-crew (see _run_isolated_task) so one agent's failure can never
    crash the whole pipeline or block later tasks. Every agent still
    genuinely attempts real tool execution every time; ground-truth
    values are used only when an agent's attempt actually fails or
    disagrees, and this is always logged, never silent.
    """
    # ---- Task 0: Extraction ----
    ground_truth_extraction = extract_from_pdf(pdf_path)
    task0 = Task(
        description=(
            f"Call the PDF extraction tool EXACTLY ONCE with pdf_path='{pdf_path}'. "
            f"Report the exact hemoglobin, platelets, inr, age, and surgery_type values it returns."
        ),
        expected_output="The extracted patient data as structured data.",
        agent=extraction_agent,
        output_pydantic=ExtractionResult,
    )

    def extract_extraction(task_output):
        p = task_output.pydantic
        if not p:
            return None
        if (p.hemoglobin == ground_truth_extraction.hemoglobin
                and p.platelets == ground_truth_extraction.platelets
                and p.inr == ground_truth_extraction.INR
                and p.age == ground_truth_extraction.age
                and p.surgery_type == ground_truth_extraction.surgery_type):
            return (p.hemoglobin, p.platelets, p.inr, p.age, p.surgery_type)
        return None  # treat any mismatch as a failure -> triggers fallback

    ground_truth_extraction_tuple = (
        ground_truth_extraction.hemoglobin, ground_truth_extraction.platelets,
        ground_truth_extraction.INR, ground_truth_extraction.age,
        ground_truth_extraction.surgery_type,
    )
    extraction_value, extraction_ok, _ = _run_isolated_task(
        "Extraction", extraction_agent, task0, ground_truth_extraction_tuple, extract_extraction,
    )
    hemoglobin, platelets, inr, age, surgery_type = extraction_value

    record = PatientRecord(
        hemoglobin=hemoglobin, platelets=platelets, INR=inr,
        age=age, surgery_type=surgery_type,
    )
    ground_truth = predict_transfusion(record)
    prediction_bool = ground_truth["transfusion_needed"]

    # ---- Task 1: Prediction ----
    task1 = Task(
        description=(
            f"Call the transfusion prediction tool EXACTLY ONCE with "
            f"hemoglobin={hemoglobin}, platelets={platelets}, INR={inr}, "
            f"age={age}, surgery_type={surgery_type}. Report the exact "
            f"transfusion_needed and confidence values it returns."
        ),
        expected_output="The prediction result as structured data.",
        agent=prediction_agent,
        output_pydantic=PredictionResult,
    )

    def extract_prediction(task_output):
        p = task_output.pydantic
        if p and p.transfusion_needed == ground_truth["transfusion_needed"]:
            return p.transfusion_needed
        return None

    verified_prediction, pred_ok, _ = _run_isolated_task(
        "Prediction", prediction_agent, task1, ground_truth["transfusion_needed"], extract_prediction,
    )

    # ---- Task 2: RAG ----
    ground_truth_explanation = generate_explanation(record, prediction_bool)
    task2 = Task(
        description=(
            f"Call the RAG explanation tool EXACTLY ONCE with hemoglobin={hemoglobin}, "
            f"platelets={platelets}, inr={inr}, age={age}, surgery_type='{surgery_type}', "
            f"prediction={prediction_bool}. All these values are already known -- do not "
            f"look them up or derive them, just pass them directly to the tool. "
            f"Report the tool's explanation text exactly as returned."
        ),
        expected_output="The RAG explanation tool's output text.",
        agent=rag_agent,
    )

    def extract_rag(task_output):
        text = task_output.raw
        if text and str(hemoglobin) in text and str(inr) in text:
            return text
        return None

    delivered_explanation, rag_ok, _ = _run_isolated_task(
        "RAG", rag_agent, task2, ground_truth_explanation, extract_rag,
    )

    # ---- Task 3: Inventory ----
    ground_truth_inventory = check_inventory()
    ground_truth_low = sorted(ground_truth_inventory["low_stock_types"].keys())
    ground_truth_healthy = sorted(
        bt for bt in ground_truth_inventory["inventory"] if bt not in ground_truth_low
    )
    task3 = Task(
        description=(
            "Call the inventory tool EXACTLY ONCE, with no arguments. "
            "Report the exact inventory, low_stock_types, and healthy_types it returns."
        ),
        expected_output="The inventory status as structured data.",
        agent=inventory_agent,
        output_pydantic=InventoryResult,
    )

    def extract_inventory(task_output):
        p = task_output.pydantic
        if p and sorted(p.low_stock_types) == ground_truth_low:
            return sorted(p.low_stock_types)
        return None

    verified_low_stock, inv_ok, _ = _run_isolated_task(
        "Inventory", inventory_agent, task3, ground_truth_low, extract_inventory,
    )

    # ---- Task 4: Action ----
    # The verified-type guard tool means real dispatch is safe regardless
    # of what the Action Agent's own reasoning claims -- see
    # make_idempotent_action_tool's docstring.
    action_tool_instance, alerted_set = make_idempotent_action_tool(set(verified_low_stock))
    task4 = Task(
        description=(
            f"The verified low-stock blood types are EXACTLY: {', '.join(verified_low_stock) or 'none'}. "
            f"Call the donor alert tool exactly once for EACH blood type in that list, "
            f"and no others. Report which blood types were alerted and which needed no action."
        ),
        expected_output="Which blood types were alerted and which needed no action, as structured data.",
        agent=action_agent,
        output_pydantic=ActionResult,
    )

    def extract_action(task_output):
        p = task_output.pydantic
        if p:
            return sorted(p.alerted_blood_types)
        return None

    reported_alerted, action_report_ok, _ = _run_isolated_task(
        "Action", action_agent, task4, list(alerted_set), extract_action,
        tools_for_agent=[action_tool_instance],
    )
    # Regardless of what the agent REPORTED, what actually happened is
    # tracked in alerted_set by the guarded tool itself -- this is the
    # authoritative record of real donor contact.
    #
    # GUARANTEED DISPATCH: testing showed the Action Agent can report a
    # structurally correct-looking answer (even one that matches the
    # verified list exactly) WITHOUT ever actually calling the tool --
    # no real alert gets sent, but the report looks fine. Validating
    # and logging that gap is not sufficient on its own: the actual
    # point of this agent is a real-world side effect, and a donor who
    # genuinely needed contacting must not silently go uncontacted just
    # because the agent's own reasoning skipped the tool call. Any
    # verified low-stock type NOT already covered by a real dispatch is
    # therefore alerted directly here. The agent still gets the first,
    # genuine attempt every run; this only fills in what it missed.
    missing_dispatch = set(ground_truth_low) - alerted_set
    if missing_dispatch:
        print(f"[VALIDATION] Action: agent did not actually dispatch for {sorted(missing_dispatch)}. "
              f"Dispatching directly to guarantee real donor contact.")
        for bt in sorted(missing_dispatch):
            units = ground_truth_inventory["low_stock_types"].get(bt, 0)
            send_donor_alert(bt, units if units else None)
            alerted_set.add(bt)

    action_ok = (sorted(alerted_set) == ground_truth_low)
    print(f"[VALIDATION] Action (real dispatch): {'matches' if action_ok else 'DIFFERS FROM'} verified low-stock list.")
    print(f"[VALIDATION] Blood types that actually received a real alert this run: {sorted(alerted_set)}")

    # ---- Task 5: Orchestrator ----
    task5 = Task(
        description=(
            f"The verified case facts are: transfusion_needed={prediction_bool}, "
            f"shortage_detected={bool(verified_low_stock)}. Write a short case_summary "
            f"sentence combining the prediction and inventory situation, restate these "
            f"two verified facts exactly, and set workflow_status to 'complete'."
        ),
        expected_output="A consolidated orchestration summary as structured data.",
        agent=orchestrator_agent,
        output_pydantic=OrchestratorResult,
    )

    def extract_orchestrator(task_output):
        p = task_output.pydantic
        if (p and p.transfusion_needed == prediction_bool
                and p.shortage_detected == bool(verified_low_stock)):
            return p
        return None

    ground_truth_orchestrator = {
        "case_summary": "Case processed by HemoSmart multi-agent pipeline.",
        "transfusion_needed": prediction_bool,
        "shortage_detected": bool(verified_low_stock),
        "workflow_status": "complete",
    }
    orchestrator_value, orch_ok, _ = _run_isolated_task(
        "Orchestrator", orchestrator_agent, task5, ground_truth_orchestrator, extract_orchestrator,
    )
    orchestrator_dict = (
        orchestrator_value.dict() if hasattr(orchestrator_value, "dict") else orchestrator_value
    )

    summary = {
        "extraction": {"verified_correct": extraction_ok,
                       "value": {"hemoglobin": hemoglobin, "platelets": platelets, "inr": inr,
                                 "age": age, "surgery_type": surgery_type}},
        "prediction": {"verified_correct": pred_ok, "value": verified_prediction},
        "explanation": {"verified_correct": rag_ok, "delivered_explanation": delivered_explanation},
        "inventory": {"verified_correct": inv_ok, "low_stock_types": verified_low_stock,
                      "healthy_types": ground_truth_healthy},
        "action": {"verified_correct": action_ok, "actually_alerted": sorted(alerted_set)},
        "orchestrator": orchestrator_dict,
    }
    return summary


if __name__ == "__main__":
    sample_pdf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adapters", "sample_cbc_report.pdf")
    summary = run_full_pipeline(pdf_path=sample_pdf)

    print("\n\n" + "=" * 60)
    print("PIPELINE SUMMARY (6 agents: Extraction, Prediction, RAG, Inventory, Action, Orchestrator)")
    print("=" * 60)
    import json
    print(json.dumps(summary, indent=2, default=str))