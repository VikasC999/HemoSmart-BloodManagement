# HemoSmart — Report ↔ Code Alignment

A claim-by-claim check of the Phase I report (*"HemoSmart: Intelligent Blood Management Platform"*, SJBIT CSE-DS, 2025–26) against the code as of P3, for writing the Phase II report. Every row gives the report's claim, what the code actually does (with the file to point an examiner to), and what to do.

**Verdicts**
- ✅ **Matches.** Keep it, ideally with the evidence and a number.
- ✏️ **Reword.** The idea is implemented differently or partially; use the suggested wording.
- 🔧 **Gap.** Not implemented. Either build it in P3 (see [§ 7](#7-p3-build-candidates--status)) or descope it in the text.

Design reasoning for each implementation choice is in [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) (D-numbers below). Measured results are in [evaluation/results/](../evaluation/results/).

---

## 1. The story to tell (abstract, 1.4, 4.1)

The Phase I report describes one system in which "specialized AI agents … coordinated by a central orchestrator" do everything. The code has **two agent layers**, and the Phase II report should describe both explicitly. That is a stronger design than one LLM crew doing everything, and examiners will ask about it.

| Layer | What it does | How | Where |
|---|---|---|---|
| **Autonomous supply loop** (hospital-wide, hourly) | Inventory → Forecast → Planner → Donor → Logistics → Explanation agents on a shared blackboard, coordinated by an Orchestrator | Deterministic Python agents. The LLM only writes the cycle brief; RAG cites the handbook for each action. | [agents/autonomy/](../agents/autonomy/) — D8, D30 |
| **Patient case pipeline** (per PDF report, on demand) | Extraction → Prediction → RAG → Inventory → Action → Orchestrator agents | CrewAI LLM agents with code guardrails and fallbacks | [agents/crew_agents.py](../agents/crew_agents.py), `POST /api/cases` — D9, D40 |

**Why this design (wording for the report):** "Decisions with real-world side effects — contacting donors, moving blood between hospitals — are made by deterministic agents, so they are reliable, testable and can be simulated over months of operation. LLM agents are used where language understanding is needed: reading unstructured lab reports, and explaining decisions against clinical guidelines."

**Replace "eliminates preventable shortages" (abstract) with measured numbers:** "In a 90-day simulation against a manual blood-bank policy, HemoSmart reduced unmet demand by 55% (96 → 43 units) and on-site expiry by 76% (46.7 → 11.0 units), without increasing the total number of units not used on site." Source: [simulation.md](../evaluation/results/simulation.md), D48. Always add "on synthetic data".

---

## 2. Chapter 1 — Introduction

| Report claim | Code reality | Verdict | Action / suggested text |
|---|---|---|---|
| 1.4 Demand prediction "with clinical data, **surgery schedules**, and transfusion history" | Prophet on daily consumption history, with weekday and yearly seasonality. No surgery-schedule input. | ✏️ | "…from historical consumption, capturing weekday surgery patterns and seasonal outbreaks (dengue)." |
| 1.4 Demand–supply gap with "**LSTM and Prophet**" | Prophet is served. The LSTM was benchmarked and not served because it wasn't better. | ✏️ | "Prophet (selected over an LSTM and two baselines in a rolling backtest)" — D51, [forecast_backtest.md](../evaluation/results/forecast_backtest.md) |
| 1.4 Gap analysis triggers corrective action before shortages | The Planner replays the allocation policy day by day per group × component and turns gaps into outreach needs | ✅ | [bloodbank/projection.py](../bloodbank/projection.py), D21, D45 |
| 1.4 RL donor coordination | Thompson-sampling multi-armed bandit with real feedback, plus eligibility intervals | ✅ | Call it a "multi-armed bandit (a form of RL)". Report the benchmark honestly: +19% donations vs random; greedy is ~2% better in simulation (D53). |
| 1.4 "**WHO guideline-based** explanations" | RAG over the WHO *Clinical Use of Blood* handbook, with page citations. But several thresholds are HemoSmart's own protocol, not WHO numbers. | ✏️ | "Explanations cite the WHO handbook page by page; decision thresholds are HemoSmart's protocol, each checked against the handbook." — D52 |
| 1.4 FEFO, near-expiry prioritised | Wastage-aware allocation: near-expiry compatible units first, then own group, FEFO | ✅ | [bloodbank/allocation.py](../bloodbank/allocation.py), D16 |
| 1.4 "Integrates with **EHRs**, blood bank databases, donor systems" | Hospital adapter interface with a mock hospital (fixture EHR sync, surplus offers); manual, CSV and PDF input | ✏️ | "Integration through a hospital adapter interface, demonstrated against a simulated partner hospital." |
| 1.4 "Encryption" | bcrypt password hashing; no field-level encryption | ✏️ | See § 5 Security. |
| 1.5 XGBoost **and Random Forest** | XGBoost served; Random Forest benchmarked as equivalent (D58) | ✏️ | "XGBoost (compared against Random Forest and logistic regression)" |
| 1.5 SHAP | Exact TreeSHAP per prediction, shown as bars in the UI | ✅ | D12 |
| 1.5 LangChain + FAISS RAG | Yes (sentence-transformers MiniLM, page-level index) | ✅ | D33 |
| 1.5 CrewAI coordinates prediction/forecasting/inventory/donor/explanation agents | CrewAI runs the patient case pipeline; the supply loop is Python agents | ✏️ | Use § 1. |
| 1.5 MCP integrates with EHRs, blood banks, donor services | MCP **server** exposing HemoSmart tools (predict, explain, inventory, outlook, alert donors) to any MCP client, JWT-protected | ✏️ | "HemoSmart exposes its capabilities as a Model Context Protocol server, so any MCP-compatible AI client can use them; hospital-system integration uses a separate adapter interface." (README "MCP server") |
| 1.5 PostgreSQL, FAISS, Redis | Yes. Redis is optional caching. | ✅ | |
| 1.5 "JWT, **OAuth 2.0, AES-256, TLS 1.3**" | JWT + RBAC only | ✏️ | § 5. |
| 1.5 React dashboard | Yes, with per-role dashboards | ✅ | |
| 1.5 "GitHub, Docker, **GitHub Actions, MLflow**" | Git and Docker yes; no CI workflow, no MLflow | ✏️ | Drop both (model metrics are written by the training scripts; tests run with `pytest`). |
| 1.6 Step 1: patient data "retrieved from the hospital EHR through MCP" | Manual form, CSV batch, PDF upload, and adapter sync from the mock hospital | ✏️ | "Patient data enters through the form, CSV, a CBC PDF, or a hospital-system adapter." |
| 1.6 Step 2: predicts need, component, **units** | Yes. Units and urgency come from protocol rules, not the model. | ✅ | D11 |
| 1.6 Steps 3–9 | All implemented; Step 9 = reserve → confirm/cancel transfusion workflow | ✅ | D36 |
| 1.8 Future work | Fine as is. Add: real donor-response data, partner-hospital pilots, donor-fatigue modelling (D53). | ✅ | |

---

## 3. Chapter 2 — Literature survey

| Issue | Action |
|---|---|
| **The reference list doesn't match the survey.** The survey reviews 10 papers ([1]–[10] in § 2.2), but the References page numbers them differently and leaves out several (MCP-FHIR, the XAI study, RAG in emergency care, hospital supply chain, RL + time series, Goel et al.). Several references (Soltani, Lewis, Tayefi, Chen, Lundberg, Taylor) aren't surveyed. | Use one numbering. Every surveyed paper must appear in the References, and every reference must be cited somewhere. |
| Paper [10] (Schilling): "deep learning outperforms statistical models" | Your own backtest found the LSTM no better than Prophet on synthetic data. Say so in the discussion; it's a good, honest contrast. |
| Paper [6] RL "94.2% accuracy" | Fine as a citation. Avoid implying HemoSmart's bandit is comparable. |

---

## 4. Chapter 3 — Requirements (FR / NFR)

### Functional requirements

| Req | Claim | Code reality | Verdict | Action / evidence |
|---|---|---|---|---|
| FR-01 | Inputs: Hb, platelets (×10³/µL), age, surgery type, **hematocrit, INR, BP, heart rate** | Hb, platelets, INR, age, surgery type, blood group. A platelet value of 1–999 is read as ×10³. **No hematocrit, BP or heart rate.** | ✏️ | Drop HCT/BP/HR ("vital signs are out of scope for the transfusion models"), or add them as stored-but-unused fields. Recommend dropping. |
| FR-01 | Validate inputs; error for missing or out-of-range values | Pydantic ranges on every field; 422 responses with messages | ✅ | [schema/patient_schema.py](../schema/patient_schema.py) |
| FR-02 | XGBoost **or RF** model; yes/no | XGBoost per component; P(any) = 1 − ∏(1 − p) | ✅ | D10 |
| FR-02 | Component: PRBC, Platelets, FFP, **Cryoprecipitate** | No cryo (needs fibrinogen, which isn't collected) | ✏️ | "PRBC, platelets and FFP; cryoprecipitate requires fibrinogen, which is outside the collected parameters." D15 |
| FR-02 | SHAP shown | Yes | ✅ | |
| FR-02 | Urgency Critical / Moderate / Low by thresholds | Yes | ✅ | [schema/clinical_thresholds.py](../schema/clinical_thresholds.py) `urgency()` |
| FR-03 | Upload CBC PDF; PyMuPDF; LLM → JSON | Yes (Groq LLM, tolerant JSON parsing) | ✅ | [adapters/pdf_adapter.py](../adapters/pdf_adapter.py) |
| FR-03 | Auto-populate input fields; **show extracted values for confirmation before prediction** | Extract → editable form → confirm → predict (P3) | ✅ | D55 |
| FR-04 | FAISS store of WHO guidelines; cosine retrieval; LLM explanation; cite the guideline | Yes, with page citations, persisted with the prediction | ✅ | D33; retrieval hit@2 and faithfulness in [rag_faithfulness.md](../evaluation/results/rag_faithfulness.md) |
| FR-04 | Explanation shown **alongside every** prediction | On request (Explain button), at 1 LLM call each; the deterministic threshold facts are always shown | ✏️ | "An explanation is generated on demand for any prediction" (explains the LLM budget, see DESIGN_DECISIONS "LLM call budget"). |
| FR-05 | **LSTM and Prophet** | Prophet served; LSTM benchmarked | ✏️ | D51 |
| FR-05 | Forecast per blood type, 7 days | Per blood group × component, 1–30 days (`/api/forecast?days=`) | ✅ | |
| FR-05 | Gap analysis shortage/surplus | Shortage and wastage-risk tables per group × component | ✅ | ForecastChart |
| FR-05 | "Update forecasts **continuously** as new data is recorded" | Prophet is retrained offline by script; the gap is recomputed live against current stock | ✏️ | "The demand–supply gap is recomputed against live stock on every request; the forecasting model is retrained periodically." |
| FR-05 | Charts of trends, seasonality, **forecast confidence** | Trend lines with Prophet's 80% band; coverage measured (77–79%, 68% for platelets in dengue) | ✅ | D59 |
| FR-06 | Monitor stock across groups and components | Unit-level stock grid | ✅ | D13 |
| FR-06 | Flag units in their **last 7 days** | Per component: PRBC 7 days, platelets 48 h, FFP 30 days | ✏️ | Use the per-component table (D14). This also reconciles the report's own "7 days" (FR-06) vs "48 hours" (§ 4.3.5). |
| FR-06 | "FEFO — **oldest units** are assigned first" | FEFO = **first-expiring** first. Not the same thing for mixed stock. | ✏️ | Fix the definition. |
| FR-06 | Sufficient / Low / Critical colour status; alert below par | Yes | ✅ | D23 |
| FR-07 *(numbered "3.5.6" twice — fix)* | — | — | ✏️ | The RBAC section is labelled 3.5.6 but should be 3.5.7, and FR-07 is skipped (FR-06 → FR-08). Renumber. |
| FR-08 | Roles: Health Professional, Lab Technician, Blood Bank Officer, Hospital Administrator, **Donor** | Admin, Blood Bank Manager, Hospital Staff, Donor Coordinator, Auditor. No donor login. | ✏️ | Use the implemented roles; the mapping is below. |
| FR-08 | Restrict by role; deny with an error | Server-side RBAC on every route (403) from one shared permission matrix, plus role dashboards (D65) | ✅ | Tests: [tests/test_auth_rbac.py](../tests/test_auth_rbac.py) |
| FR-08 | (assistant) | The assistant is available to every role and answers only what that role's pages allow. Each question is checked in code before any data is read or the LLM is called; a refusal says "Not authorised" and lists what the role can ask. Refusals are audit-logged (D66–D68). | ✅ | "The AI assistant is available to all roles and is restricted by the same permission matrix as the dashboard; access is decided in code, never by the language model." Tests: [tests/test_chat_rbac.py](../tests/test_chat_rbac.py) |
| FR-08 | **Terminate sessions after 15 min of inactivity** | JWT valid for 12 h; no idle timeout (kept deliberately for testing, D60) | ✏️ | Drop it or list it as future work. |
| — | Secure logout | Sign Out clears the token (stateless JWT, no server-side revocation) | ✅ | |

**Role mapping for the report:** Health Professional / Lab Technician → **Hospital Staff**; Blood Bank Officer → **Blood Bank Manager**; Hospital Administrator → **Admin**; Donor → *none* (donors are contacted, they don't log in), replaced by **Donor Coordinator** (records donor replies). **Auditor** is new (read-only audit log, for accountability).

### Non-functional requirements

| NFR | Claim | Reality | Verdict | Action |
|---|---|---|---|---|
| NFR-01 | Prediction < 2 s; RAG < 5 s; forecast < 10 s; dashboard < 3 s | Measured: prediction 0.01 s, forecast ≤ 0.35 s, RAG median 2.05 s (p95 16.9 s on the Groq free tier) | ✅/✏️ | Report the medians and name the RAG tail (D57). |
| NFR-02 | Labelled inputs with units and ranges; plain language; visual indicators | Mostly yes | ✅ | Check during the browser walkthrough. |
| NFR-03 | **AES-256 at rest, TLS 1.3 in transit** | Not implemented in the app | ✏️ | § 5. |
| NFR-03 | Passwords hashed | bcrypt | ✅ | [backend/services/auth.py](../backend/services/auth.py) |
| NFR-03 | Unauthorized users can't access pages | Yes | ✅ | |
| NFR-03 | Every access in an **immutable** audit log | Every `/api/*` request is logged; there's no update/delete endpoint, but the DB doesn't enforce append-only | ✏️ | "append-only from the application" (or add a DB trigger, § 7). |
| NFR-03 | **Mask identifiers to the last 4 digits** | Donor phones masked in the API, alerts and logs; patients store no identifiers | ✅ | "Patients are stored without direct identifiers; donor phone numbers are masked to the last 4 digits." (D56) |
| NFR-04 | Graceful invalid input; no crash on bad PDF | Validation + error responses | ✅ | |
| NFR-04 | ACID transactions | Postgres transactions; each agent step commits separately | ✅ | D30 |
| NFR-04 | **Agent failure → log + alternative path** | Yes: forecast → historical means, Planner fail → skip outreach, LLM → template; cycle marked `degraded`. CrewAI agents fall back to deterministic results. | ✅ | D30, D42; strong viva point |
| NFR-05 | Modular; replaceable model; updatable FAISS; Git | Yes | ✅ | |
| NFR-06 | Windows / Linux / macOS; Docker; browser | Developed on macOS; Docker image; the Windows-only LSTM bridge was removed | ✏️ | "Runs anywhere Docker runs; developed and tested on macOS." |
| Compliance | "Complies with **DPDP Act 2023** and **NACO**" | Not audited | ✏️ | "Designed with DPDP Act 2023 principles (access control, audit trail, data minimisation) and NACO donor-interval rules in mind; formal compliance is out of scope." |

### 3.4, 3.7–3.8 (stack and interfaces)

| Claim | Reality | Action |
|---|---|---|
| "React **or Streamlit**" | React (Vite) | Say React. |
| "**LLaMA or Mistral** LLM" | Groq-hosted `openai/gpt-oss-20b`; Ollama `llama3.1:8b` supported locally | "An open-weight LLM (gpt-oss-20b via Groq; Llama 3.1 8B via Ollama locally)." |
| "TensorFlow or PyTorch for LSTM" | PyTorch LSTM in the benchmark only | OK as written, plus "(evaluation only)". |
| "Psycopg2" | psycopg 3 | Minor. |
| SMS / WhatsApp API for donor alerts | Dispatch is a log stub | "Alert dispatch is stubbed (logged); an SMS/WhatsApp gateway plugs into the same call." |
| 3.8.5 MCP tools for "inventory queries, **EHR data retrieval**, donor alert dispatch, **audit log writing**" | MCP tools: predict, explain, check_inventory, supply_outlook, alert_donors. No EHR retrieval or audit-writing tools (MCP calls are audit-logged by the API). | Reword to the actual tool list. |
| 3.8.3 ML input includes **HCT** | No HCT | Remove. |
| 3.10 "Single hospital deployment" | True, plus redistribution to a simulated partner | Fine. |

---

## 5. Security wording (fixes several rows above)

> "Authentication uses JWT with bcrypt-hashed passwords; authorization is role-based and enforced on every API route and on the MCP endpoint. Every API request is written to an audit log readable only by Auditors and Admins. Transport encryption (TLS) and storage encryption are provided by the hosting platform and managed PostgreSQL service and are outside the application's scope; AES-256 field-level encryption of patient identifiers is future work."

Drop OAuth 2.0 (there's no third-party login).

---

## 6. Chapter 4 — System design

| Section | Problem | Action |
|---|---|---|
| 4.1 | The system is called "**HemoAssist**" | Typo: HemoSmart. |
| 4.1 step 5 | "The **XGBoost model predicts blood demand**" | Wrong. XGBoost predicts **patient transfusion need**; **Prophet** forecasts demand. |
| 4.1 steps 4, 7 | "LangChain agent decides DB/RAG/forecast"; "**Mistral 7B** synthesizes" | The chat agent routes by intent (keywords first, LLM only when unclear); the LLM is gpt-oss-20b. Every routed intent passes a role permission check first (D66). Redraw Fig 4.1 with the two agent layers (§ 1). |
| 4.3.3 | "XGBoost **and Random Forest**"; estimates **unit quantity** | No RF; units from protocol rules (D11). |
| 4.3.4 | "**7, 14 and 30-day**" forecasts | API supports 1–30 days; the Planner uses 7. ✅ |
| 4.3.4 | "Adjusts for seasonal disease patterns **and surgery schedules**" | Seasonality yes (dengue), surgery schedules no. |
| 4.3.4 | Automated shortage alerts when the gap crosses a threshold | ✅ The Planner turns gaps inside the shelf-life window into donor outreach (D45, D46). |
| 4.3.5 | "Final **48 hours**" | Per component (D14). |
| 4.3.5 | Inventory updated after every transfusion | ✅ reserve → confirm issues units. |
| 4.3.6 | Alerts "via **MCP notification service**" | Stub; MCP isn't the notification path. |
| 4.3.6 | Eligibility "interval **and donation type**" | ✅ D47 (whole blood 90 d; apheresis 28 d). |
| 4.3.7 | Explainability | ✅ plus the protocol-vs-WHO wording (D52). |
| 4.3.8 | Orchestrator delegates to "**Prediction**, Forecast, Inventory, Donor, Explanation" agents | The supply loop has Inventory, Forecast, Planner, Donor, Logistics, Explanation; Prediction lives in the case pipeline. Describe both (§ 1). |
| 4.3.8 | Agent monitoring (new) | ✅ Add: a live **Agent monitor** shows each agent's status (working / waiting / degraded / failed / idle), time per step, a scheduler heartbeat, stuck-step and orphaned-cycle detection, and a "Needs attention" queue. Every step reports an allow-listed, patient-data-free event (D71–D75). |
| 4.3.8 | Failure recovery (new evidence) | ✅ Shown live: a rate-limited CrewAI agent shows "Waiting · retry in N s", then continues; a failed supply-loop step shows "Failed", the cycle "degraded", and the dependent steps "skipped". |
| 4.3.8 | "**Without human intervention**" | True for the hourly loop. The patient pipeline only *recommends* dispatch unless the user is a manager (`dispatch_allowed`). Mention the human-in-the-loop for clinical decisions. That's a plus. |
| 4.3.9 | MCP "eliminates custom connectors", "plug-and-play" | Overclaim. Use the adapter wording (§ 2, 1.5 MCP row). |
| 4.3.10 | AES-256 / TLS 1.3 / "full DPDP compliance" | § 5. |
| 4.4 | DFDs | Redraw from the implemented flow (both layers + redistribution). |
| 4.5 #2 | "**Parallel** execution" of agents | Agents run sequentially on a blackboard (each reads the previous one's output). Say "independent, sequentially coordinated". The Agent monitor's run timeline makes the sequential order visible. |
| 4.5 #6 | "**Continuous learning** improves prediction and forecasting" | Only the donor bandit learns online. "The donor-selection model learns continuously from every response; predictive models are retrained offline." |

---

## 7. P3 build candidates — status

| # | Item | Report refs | Status |
|---|---|---|---|
| 1 | 15-minute inactivity logout | FR-08, 4.3.2 | **Not built (owner's choice, D60).** Keeps the 12 h login for testing. Drop it from the report or list it as future work. |
| 2 | Confirm extracted PDF values before predicting | FR-03 | **Built** (D55). FR-03 now ✅ |
| 3 | Mask identifiers to the last 4 characters | NFR-03 | **Built** for donor phones; patients hold no identifiers (D56). NFR-03 masking ✅ with that wording. |
| 4 | Measure the NFR-01 response times | NFR-01 | **Built** ([latency.md](../evaluation/results/latency.md), D57). Prediction and forecast ✅; RAG ✅ at median, ✗ at p95 on the Groq free tier. Dashboard load: browser walkthrough. |
| 5 | Random Forest in the benchmark | 1.5, FR-02, 4.3.3 | **Built** (D58). Equivalent to XGBoost. Keep the RF mention as "compared". |
| 6 | GitHub Actions CI | 1.5 | Not built (owner's choice). Drop it from the report. |
| 7 | DB-level append-only audit log | NFR-03 | Not built (owner's choice). Use the "append-only from the application" wording. |
| 8 | Forecast confidence band | FR-05 | **Built**, with measured coverage (D59). FR-05 "forecast confidence" ✅ |
| — | Hematocrit / BP / HR, cryoprecipitate, OAuth, AES-256, MLflow, real SMS, DPDP audit | various | **Descope** in the text (wording above). |

## 8. New Phase II sections the report needs

| Section | Content | Source |
|---|---|---|
| Implementation | The two agent layers; the bloodbank domain model (unit-level stock, ABO/RhD compatibility per component, allocation, gap projection); donor bandit; RAG with citations; the case pipeline with guardrails | D8–D44 |
| Evaluation methodology | Synthetic world, ablation design, the honest wastage metric, seeds, reproducibility | D48, D49 |
| Results | Tables: simulation, sensitivity, prediction benchmark, forecast backtest, bandit, RAG | evaluation/results/*.md |
| Discussion | What worked (allocation, agents, redistribution); the v1 policy failure and how the simulation caught it (D45–D47); greedy vs Thompson; protocol vs WHO; LSTM vs Prophet | DESIGN_DECISIONS |
| Limitations | Synthetic data; simulated donor replies, partner and supply; a small RAG evaluation | DESIGN_DECISIONS "Known limitations" |
| Figures (done) | `fig_architecture` (replaces Fig 4.1), `fig_simulation`, `fig_sensitivity`, `fig_prediction`, `fig_forecast`, `fig_bandit` in [docs/figures/](figures/) | `python -m evaluation.figures` regenerates them from the results JSON |
