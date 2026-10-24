# HemoSmart — Design Decisions & Trade-offs

A log of every deliberate design choice made while turning HemoSmart into a
**wastage-reducing autonomous blood management system using RAG and
multi-agents**: what was decided, why, what was given up, and how to revert
it. Newest phase last. When you change one of these, update its entry rather
than deleting it, so the reasoning trail survives.

Each entry: **Decision** · **Why** · **Alternatives considered** ·
**Trade-off** · **How to revert / tune**.

---

## Contents

- [P0 — Integrity & security](#p0--integrity--security)
- [P1 — Wastage-reducing autonomous system](#p1--wastage-reducing-autonomous-system)
  - [Agent architecture](#agent-architecture)
  - [Prediction](#prediction)
  - [Blood-bank domain](#blood-bank-domain)
  - [Forecasting & gap projection](#forecasting--gap-projection)
  - [Donors](#donors)
  - [Redistribution](#redistribution)
  - [Orchestration & scheduling](#orchestration--scheduling)
  - [RAG](#rag)
  - [Data, storage & API](#data-storage--api)
- [P2 — CrewAI case pipeline in the UI](#p2--crewai-case-pipeline-in-the-ui)
- [P2 — Evaluation](#p2--evaluation)
  - [Outreach policy (fixes found by the simulation)](#outreach-policy-fixes-found-by-the-simulation)
  - [Simulation & results](#simulation--results)
  - [Prediction](#prediction-1)
  - [Forecasting](#forecasting)
  - [RAG faithfulness](#rag-faithfulness)
  - [Donor bandit](#donor-bandit)
  - [Tooling](#tooling)
- [P3 — Report alignment](#p3--report-alignment) (D55–D63) · frontend design system (D64)
- [Role-aware assistant (FR-08)](#role-aware-assistant-fr-08) (D65–D70)
- [Agent monitor](#agent-monitor) (D71–D75)
- [LLM call budget](#llm-call-budget)
- [Known limitations](#known-limitations)

---

## P0 — Integrity & security

### D1. User registration is Admin-only
- **Decision:** `POST /api/auth/register` requires an Admin token ([backend/routes/auth.py](../backend/routes/auth.py)).
- **Why:** the endpoint lets the caller choose any role, including Admin. Open self-registration was a privilege escalation.
- **Alternatives:** self-registration limited to non-privileged roles with Admin approval.
- **Trade-off:** someone has to provision staff accounts; there's no sign-up page.
- **Revert:** remove `dependencies=[Depends(require_role("Admin"))]` from the route (not recommended).

### D2. The MCP endpoint uses the same JWT, restricted to Admin + Blood Bank Manager
- **Decision:** [backend/middleware/mcp_auth.py](../backend/middleware/mcp_auth.py) wraps `/mcp`. The role is read from the database on every request, not from the token's claim. The stdio MCP server has no auth.
- **Why:** MCP authorizes per *connection*, not per tool. Only roles that hold every permission the tools expose (predict, inventory, outlook, donor alerts) may connect, so MCP can't bypass the REST RBAC. Reading the role from the DB means a demoted user's old token stops working.
- **Alternatives:** a separate static MCP API key; per-tool role checks inside each tool.
- **Trade-off:** Hospital Staff can't use MCP at all, even for the tools they could use over REST. stdio is trusted because it runs as a local subprocess.
- **Revert / tune:** edit `MCP_ALLOWED_ROLES`, or unwrap the mount in [backend/main.py](../backend/main.py).

### D3. No insecure defaults
- **Decision:**
  - JWT secret unset (or the `.env.example` placeholder) → a random per-process secret.
  - Admin password unset → a random one-time password printed to the log.
  - CORS unset → `localhost:5173` only, never `*`.
- **Why:** hardcoded fallbacks were public in the repo; anyone could forge an Admin token.
- **Trade-off:** without `JWT_SECRET_KEY` set, logins don't survive a server restart (annoying with `--reload`).
- **Revert / tune:** set `JWT_SECRET_KEY`, `ADMIN_PASSWORD` and `CORS_ORIGINS` in `backend/.env`.

### D4. One source of truth for clinical thresholds; platelet threshold 50,000/µL
- **Decision:** [schema/clinical_thresholds.py](../schema/clinical_thresholds.py) holds Hb 8 g/dL (10 for Emergency), platelets 50,000/µL and INR 1.5. The dataset labels, the explanation facts and the tests all import from it.
- **Why:** labels used 80k while explanations used 50k, so the model and its explanation contradicted each other for 50–80k patients. 50k matches the guideline text.
- **Updated in P2:** thresholds became context-dependent (cardiac Hb 9 g/dL, platelets 100k; [D50](#d50-labels-are-simulated-clinician-decisions-not-the-threshold-rule)). Several numbers turned out not to be in the WHO handbook; they are HemoSmart's protocol ([D52](#d52-protocol-thresholds-are-attributed-to-hemosmart-not-to-who-retrieval-uses-the-handbooks-vocabulary)).
- **Trade-off:** the patient data had to be regenerated and XGBoost retrained.
- **Revert / tune:** change the constant, then run `python data/create_dataset.py && python models/train_xgboost.py`.

### D5. Thrombocytopenic subgroup in synthetic patients
- **Decision:** 8% of synthetic patients get platelets drawn from 10k–120k ([data/create_dataset.py](../data/create_dataset.py)), using a *separate* RNG.
- **Why:** the healthy-centred draw was clipped at 50k, so no patient ever crossed the platelet threshold and the model could never learn it. The separate RNG keeps `blood_demand.csv` byte-identical.
- **Revert / tune:** change `0.08` or the range.

### D6. Platelet unit normalization
- **Decision:** a platelet value from 1 to 999 is interpreted as ×10³/µL and multiplied by 1,000 ([schema/patient_schema.py](../schema/patient_schema.py)). The PDF extraction prompt says the same.
- **Why:** many labs report "128 x10^3/uL" (see `adapters/sample_cbc_report_2.pdf`). A count under 1,000/µL is not a real measurement, so the conversion is unambiguous.
- **Trade-off:** a genuinely mistyped "500" becomes 500,000 instead of being rejected.
- **Revert:** delete the `normalize_platelet_units` validator.

### D7. Tests run on throwaway SQLite
- **Decision:** [tests/conftest.py](../tests/conftest.py) points `DATABASE_URL` at a temp SQLite file before anything imports `db.database`.
- **Why:** tests must never touch the Postgres demo data.
- **Trade-off:** Postgres-only behaviour (JSONB, ALTER TABLE) isn't covered by the suite; it was verified manually on a pg_dump copy.

---

## P1 — Wastage-reducing autonomous system

### Agent architecture

#### D8. The autonomous loop uses deterministic Python agents; LLM/RAG only for language *(chosen by the project owner)*
- **Decision:** in [agents/autonomy/](../agents/autonomy/), the Donor, Inventory, Forecast, Planner, Logistics and Explanation agents are Python classes. Each reads and writes a shared blackboard and logs observe/decide/act events. The LLM is used only to write the cycle brief; RAG supplies a guideline citation for each kind of action.
- **Why:** these decisions have real-world side effects (contacting donors, moving blood). Code is reliable, testable and fast enough (~1 s per cycle) to simulate months of operation.
- **Alternatives considered:**
  1. *CrewAI LLM agents choose the actions.* Closest to the report's wording, but several Groq calls per cycle, slow and non-deterministic. A 90-day simulation becomes impractical. The repo's own history showed CrewAI agents inventing blood types and skipping tool calls.
  2. *Hybrid:* Python agents propose and one RAG-grounded LLM "supervisor" approves or vetoes. More agentic, but one LLM call per cycle, and the simulation would need a supervisor-off mode.
- **Trade-off:** the autonomous agents aren't "LLM agents"; the report must describe them as rule-based specialist agents with LLM explanation.
- **Revert:** option 2 is the natural next step. Add a `SupervisorAgent` between `Donor.act`/`Logistics` and `Explanation` in [orchestrator.py](../agents/autonomy/orchestrator.py), and make proposals pending until approved.

#### D9. Two multi-agent layers, not one
- **Decision:** the deterministic autonomous loop (hospital-wide supply) and the CrewAI pipeline (per-patient PDF case, LLM agents with guardrails) coexist. Neither changes the other. See [P2](#p2--crewai-case-pipeline-in-the-ui).
- **Why:** examiners expect visible LLM-agent collaboration; supply decisions need determinism. Each layer does what it's suited to.
- **Overlap:** both can alert donors through the same donor module, which never alerts a donor who already has a pending alert. The loop also counts pending alerts before sending more.

### Prediction

#### D10. One binary XGBoost model per component (multi-label)
- **Decision:** PRBC, Platelets and FFP each get their own classifier ([models/train_xgboost.py](../models/train_xgboost.py)). Transfusion probability = `1 − ∏(1 − p_component)`. If that is ≥ 0.5 but no single component is, the most likely component is recommended.
- **Why:** a patient can need several components at once; multi-class could only pick one.
- **Trade-off:** three models to maintain. The combined probability treats components as independent.
- **Revert:** a single binary model was the pre-P1 design (`git show 77ffcbc:models/train_xgboost.py`).

#### D11. Dose and urgency are protocol rules, not ML
- **Decision:** [schema/clinical_thresholds.py](../schema/clinical_thresholds.py) defines these. The model decides *which* components are needed; the rules decide how much and how fast.

  | | Rule |
  |---|---|
  | PRBC | ⌈threshold − Hb⌉ units, 1–4 |
  | Platelets | 1 adult dose (2 if count < 20k) |
  | FFP | 2 units (4 if INR > 2) |
  | Critical | Hb < 7, platelets < 20k, INR > 2.5, or Emergency |

- **Why:** FR-02 says urgency is "based on parameter thresholds", and the explanation can state the exact rule. There's no dose label to learn from synthetic data anyway.
- **Trade-off:** doses assume a ~70 kg adult; weight isn't collected.
- **Tune:** constants at the top of the dose/urgency section.

#### D12. SHAP computed by XGBoost itself
- **Decision:** `booster.predict(pred_contribs=True)` (exact TreeSHAP) in [agents/tools.py](../agents/tools.py).
- **Why:** no `shap` runtime dependency; it's exact and fast. `shap` is only used offline for the summary plots (`backend/requirements-dev.txt`).
- **Trade-off:** contributions are in log-odds, not probability.

### Blood-bank domain

#### D13. Unit-level inventory
- **Decision:** the `blood_units` table tracks each unit's group, component, collection and expiry dates, and status: available → reserved → issued, or expired / transferred / discarded. There is no "set stock to N" endpoint; stock changes only through traceable events.
- **Why:** wastage can't be measured or reduced without expiry per unit.
- **Revert:** the old `inventory` table (counts per group) still exists in old databases but is unused.

#### D14. Shelf lives and near-expiry windows are per component
- **Decision:** [bloodbank/catalog.py](../bloodbank/catalog.py):

  | Component | Shelf life | Near-expiry window |
  |---|---|---|
  | PRBC | 42 days | 7 days |
  | Platelets | 5 days | 48 hours |
  | FFP | 365 days | 30 days |

- **Why:** the WHO handbook (p.27) gives 72 h for platelets *unless in validated packs*. Indian blood banks use 5-day packs, matching the report's "5–7 days". A flat 7-day window would mark every platelet unit near-expiry from the moment it's collected. This also reconciles the report's two numbers (7 days in FR-06, 48 h in §4.3.5).
- **Tune:** edit the dicts.

#### D15. Cryoprecipitate isn't tracked
- **Why:** its indication (low fibrinogen) needs a lab value HemoSmart doesn't collect. It could never be predicted, so it would only sit in stock and expire.
- **Revert:** add fibrinogen to `PatientRecord`, a cryo label/model, and `"Cryo"` to `COMPONENTS`.

#### D16. Wastage-aware allocation policy
- **Decision:** [bloodbank/allocation.py](../bloodbank/allocation.py) only considers compatible units, then ranks them:
  1. near-expiry units from *any* compatible group;
  2. otherwise the patient's own group;
  3. FEFO within each tier.
- **Why:** a unit expiring in 36 hours is better issued than wasted. Otherwise, own-group-first conserves universal units (O− red cells, AB plasma).
- **Alternatives:** pure FEFO (spends O− freely); pure identical-first (wastes near-expiry cross-group units).
- **Trade-off:** O− red cells near expiry will be given to non-O− patients. Good for wastage, but it draws down O− in a shortage.

#### D17. Platelet compatibility
- **Decision:** platelets follow the plasma (reverse ABO) rule, and RhD-negative recipients get only RhD-negative platelets.
- **Why:** platelet packs carry some plasma and a few red cells (anti-D sensitization risk).
- **Trade-off:** simplified. Real practice often allows ABO-nonidentical platelets more freely, and gives anti-D cover when RhD-negative platelets are short.

### Forecasting & gap projection

#### D18. Top-down hierarchical forecast: Prophet per component, split by group share
- **Decision:** [models/train_prophet.py](../models/train_prophet.py) and [bloodbank/forecast.py](../bloodbank/forecast.py). Group shares come from the last 365 days.
- **Why:** rare groups (AB−, A−) see a fraction of a unit per day, mostly zeros; per-series models can't learn seasonality from that. Component totals carry the signal (weekday surgery, the dengue platelet spike).
- **Alternatives:** 24 Prophet models (one per group × component); bottom-up.
- **Trade-off:** group mix is assumed constant over the horizon.

#### D19. Forecasts extrapolate from history ending 2025-12-31
- **Decision:** Prophet predicts arbitrary future dates (today onwards). The synthetic history isn't regenerated relative to "now".
- **Why:** reproducible data; Prophet extrapolates trend and seasonality.
- **Trade-off:** forecasts ~10 months beyond the training data rely on a near-flat synthetic trend.
- **Tune:** extend the date range in `data/create_dataset.py` and retrain.

#### D20. The LSTM is no longer served
- **Decision:** `_forecast_with_lstm` (subprocess into a separate TensorFlow venv, Windows path) was removed from serving. `agents/lstm_forecast_standalone.py` and `models/train_lstm_colab.ipynb` are kept.
- **Why:** it forecast only aggregate demand (not per component/group), never ran on macOS (hardcoded `Scripts/python.exe`), and needs a TensorFlow env that conflicts with CrewAI.
- **Revert:** P2 brings an LSTM back as a benchmarked comparison; serving it would need per-component training plus the subprocess bridge (`git show 77ffcbc:agents/tools.py`).

#### D21. The gap projection replays the real allocation policy, jointly per component
- **Decision:** [bloodbank/projection.py](../bloodbank/projection.py) replays the horizon day by day:
  - each recipient group's forecast demand is served using `rank_units` (the same policy as at issue time);
  - the most restricted recipients are served first;
  - fractional demand carries over between days;
  - whatever expires unused counts as projected waste.
- **Why:** treating groups in isolation overstated both waste and shortages. On the seeded stock: 14 → 7 units at risk and 2 → 1 shortages, because compatible patients use rare-group units. This also guarantees the forecast gap and actual issuing behaviour can't disagree.
- **Alternatives:** stock − forecast per group (wrong for perishables); a per-group FEFO walk (the first P1 version).
- **Trade-off:** more compute (still milliseconds); "most restricted first" is a heuristic ordering.

#### D22. Routine supply inflow in the projection
- **Decision:** `ROUTINE_SUPPLY_SHARE` = PRBC 0.85, Platelets 0.8, FFP 0.85 of average daily demand, arriving as fresh units ([bloodbank/catalog.py](../bloodbank/catalog.py)). Virtual inflow units are never reported as at-risk.
- **Why:** without it, a normal 3-day buffer looks like a crisis over 7 days. The first smoke test flagged 13 of 24 group/components as shortages and would have spammed donors.
- **Trade-off:** an assumption, not data. In live mode the inflow only physically appears if someone records received units.
- **Tune:** change the shares. A value of 0 reproduces the alert storm; 1.0 hides real deficits.

#### D23. Par levels and status
- **Decision:** par = historical mean daily demand × `DAYS_OF_COVER` (PRBC 3, Platelets 2, FFP 3), with a floor of 2 units. Sufficient ≥ par; Low ≥ par/2; otherwise Critical.
- **Why:** simple and explainable. The status grid is a dashboard signal; the projection drives actions.

### Donors

#### D24. Thompson-Sampling bandit with a real feedback loop
- **Decision:** [agents/donor_module.py](../agents/donor_module.py):
  - `alerts_sent` increments when the alert is sent (a trial);
  - `alerts_responded` increments on a reply (a success);
  - a reply = a donation: units enter stock and `last_donation_date` resets;
  - no reply within 48 h counts as a non-response.
- **Why:** before P1 the counters never changed, so the bandit never learned.
- **Trade-off:** a slow donor who replies after 48 h counts as a non-response.
- **Tune:** `RESPONSE_WINDOW_HOURS`.

#### D25. Donation types and eligibility intervals
- **Decision:** whole blood (yields 1 PRBC + 1 FFP) requires 90 days since the last donation; platelet apheresis (1 adult dose) requires 28 days. Which interval applies also depends on the *previous* donation's type ([D47](#d47-eligibility-depends-on-the-previous-donations-type)).
- **Why:** these are the donation types in practice. 90 days is the standard whole-blood gap.
- **Trade-off:** the 28-day apheresis interval is a configurable policy, not a NACO citation. Sex-specific whole-blood intervals (90 vs 120 days) are ignored because sex isn't stored.

#### D26. Outreach sizing and merging
- **Decision:**
  - alerts = ⌈gap / expected reply rate⌉ − pending alerts, capped at 16 per need (`DonorAgent.MAX_ALERTS_PER_SHORTAGE`; was 8 in P1, raised by the sensitivity sweep, [D45](#d45-outreach-acts-only-on-gaps-inside-the-components-shelf-life));
  - the gap is only the part inside the outreach window plus a safety-stock top-up ([D45](#d45-outreach-acts-only-on-gaps-inside-the-components-shelf-life), [D46](#d46-safety-stock-outreach-fires-before-stock-hits-zero));
  - the expected rate is the mean Beta-posterior of eligible donors;
  - PRBC and FFP gaps in the same group merge into one whole-blood need of max(gap), because one donation covers both;
  - whatever donors can't cover is escalated (logged).
- **Why:** sized outreach avoids both under-alerting and spamming donors.
- **Trade-off:** escalation to a "regional blood bank" is a log entry only; there is no real integration.

#### D27. Simulated donor replies
- **Decision:** with `HEMOSMART_SIMULATE_DONOR_RESPONSES=1` (default), the loop samples replies from each synthetic donor's hidden `simulated_response_rate`. The bandit never reads that column. With `0`, only a Donor Coordinator's recorded replies count.
- **Why:** synthetic project; the feedback loop must be demonstrable.
- **Revert:** set the env var to 0 for "live" behaviour.

#### D28. Donor pool of 400
- **Why:** 50 donors across 8 groups left large groups with no eligible donors and triggered constant escalations.
- **Note:** existing databases keep their old pool; only a fresh seed creates 400.

### Redistribution

#### D29. Near-expiry surplus is offered to a partner hospital
- **Decision:** the Logistics agent offers units that are both (a) projected to expire unused and (b) inside the near-expiry window to a `HospitalSystemAdapter.offer_surplus_units`. The mock partner accepts units with ≥ 12 h left, up to 6 per offer.
- **Why:** redistribution turns one site's waste into another site's supply, the second wastage lever after FEFO.
- **Trade-off:** acceptance is **simulated**. Transferred units count as "wastage avoided", which assumes the partner uses them.
- **Tune:** `MockHospitalAdapter.MIN_REMAINING_HOURS` / `MAX_UNITS_PER_OFFER`.

### Orchestration & scheduling

#### D30. Each agent step commits on its own, with fallbacks
- **Decision:** [agents/autonomy/orchestrator.py](../agents/autonomy/orchestrator.py):
  - forecast fails → plan on historical averages;
  - Planner fails → skip outreach and logistics;
  - LLM fails → template brief;
  - any failure → cycle status `degraded`.
- **Why:** an alert already sent must stay recorded even if a later agent fails. This implements the "fallback routing" requirement (NFR-04).
- **Alternatives:** one transaction per cycle (all-or-nothing); SAVEPOINTs, avoided because of SQLite/pysqlite savepoint quirks.

#### D31. One cycle at a time, in-process scheduler
- **Decision:** a process-level lock, and an asyncio task in the FastAPI lifespan. `HEMOSMART_AUTONOMY_INTERVAL_MINUTES` defaults to 60 (0 disables); the first run happens one interval after startup.
- **Trade-off:** not safe with multiple uvicorn workers (each would run its own scheduler). A multi-worker deployment needs an external scheduler or a DB-level lock.

#### D32. The cycle brief: template by default, LLM if available
- **Decision:** the Explanation agent builds numeric facts in code and asks the LLM to phrase them (with handbook citations). Any failure falls back to the template.
- **Why:** every number in the brief comes from the agents, not the LLM.

### RAG

#### D33. The FAISS index carries page metadata; explanations cite pages and are persisted
- **Decision:**
  - [rag/setup_rag.py](../rag/setup_rag.py) indexes one document per PDF page;
  - [rag/explain.py](../rag/explain.py) numbers the sources `[n]` in the prompt and returns `citations`;
  - `POST /api/explain` with `prediction_id` saves the explanation and citations;
  - `GuidelineRetriever` is shared with the Explanation agent.
- **Why:** FR-04 says to cite the guideline used.
- **Trade-off:** citation *quality* is weak today (see [Known limitations](#known-limitations)).

#### D34. Thresholds are computed in code; the LLM only phrases them
- **Decision:** unchanged from before P0, now fed by the shared thresholds module.
- **Why:** small LLMs make numeric comparison errors.

### Data, storage & API

#### D35. Additive schema sync instead of Alembic
- **Decision:** on startup, `init_db()` adds missing *nullable* columns ([db/database.py](../db/database.py)). It never alters or drops.
- **Why:** pre-P1 databases upgrade in place with zero setup. Verified on a copy of the local DB: 10 columns added, data intact.
- **Trade-off:** it can't rename, retype or drop columns. Adopt Alembic before any such change.

#### D36. Two-step transfusion: reserve (FEFO) → confirm / cancel
- **Decision:** [backend/routes/transfusions.py](../backend/routes/transfusions.py). Partial reservations are allowed, with a `shortfall` recorded.
- **Why:** matches report Step 9 ("approved transfusions are recorded"), keeps units held for a patient from being issued twice, and surfaces a shortage at the bedside.

#### D37. Only the forecast is cached
- **Decision:** `GET /api/forecast` caches the Prophet output per date in Redis; the gap is recomputed against live stock every time.
- **Why:** a cached gap would be stale right after a transfusion or an autonomy cycle.

#### D38. Removed artifacts
- **Decision:** `models/xgb_model.pkl`, `models/prophet_model.pkl`, `models/shap_summary.png` and `working.py` were deleted, replaced by per-component models.
- **Revert:** `git show 77ffcbc:<path>`.

#### D39. Stale alerts expire before simulated replies; flush after each response
- **Decision:** the Donor agent always expires alerts older than 48 h first, then simulates replies. `record_donor_response` flushes.
- **Why:** found while testing on a copy of the real DB. Pre-P1 alerts (NULL `responded`) would otherwise have become "donations today". Sessions use `autoflush=False`, so without the flush the simulator saw already-resolved alerts as pending and crashed the Donor step.

---

## P2 — CrewAI case pipeline in the UI

### D40. Wire the existing CrewAI PDF-case pipeline into the app (option A)
- **Decision:** "Process full case (agents)" on the Upload card runs [agents/crew_agents.py](../agents/crew_agents.py) as a background job with per-agent progress. The normal PDF upload stays as it is.
- **Why:** makes LLM-agent collaboration visible. The ✓ / fell-back marker per agent measures how often LLM agents get it right on their own, which is evidence for D8.
- **Alternatives:**
  - B — keep it CLI-only and describe it as a prototype;
  - C — delete it and describe only the autonomous loop as multi-agent.
- **Trade-offs:**
  - **Cost:** ~15–20 LLM calls per case (see [LLM call budget](#llm-call-budget)), roughly 20–45 s on Groq.
  - **Rate limits:** Groq's free tier has per-minute caps, so only **one case runs at a time**.
  - **Jobs are in memory:** a server restart loses in-flight job status (the patient and prediction are still saved).
  - **Real tool calls are kept:** the agent's tool call and the guardrail's reference value each do the expensive work. Caching the reference into the tool would save ~3 calls but make the agent's work partly ceremonial.
  - **RBAC:** the Action agent only *dispatches* donor alerts if the caller may send alerts (Blood Bank Manager/Admin). For Hospital Staff it records recommendations without contacting anyone.
- **Revert:** remove the route and the Upload option; the CLI (`python agents/crew_agents.py`) keeps working.
- **Measured** on the free Groq tier, sample reports, after D41–D44:

  | Run | Agents verified | LLM requests | Time |
  |---|---|---|---|
  | report 1, staff | 6 / 6 | 16 | 58.5 s |
  | report 2, manager | 5 / 6 | 25 | 67.4 s |
  | report 1, manager | 4 / 6 | 30 | 72.3 s |

  Request counts include rate-limited attempts; most of the time is spent waiting on the token-per-minute cap, not on the model. Runs vary because LLM agents vary; that variance is the point the ✓ / fell-back markers make.

### D41. CrewAI's prompt-cache marker is neutralised
- **Finding:** CrewAI 1.15 adds a `cache_breakpoint` key to messages. Its native adapters strip it, but the LiteLLM path that Groq uses does not, and Groq rejects the request. **Every** agent failed on its first call, and the guardrails were silently falling back on every run, including from the CLI before P2.
- **Decision:** [agents/crew_agents.py](../agents/crew_agents.py) replaces `crewai.llms.cache.mark_cache_breakpoint` with a no-op. CrewAI is pinned to `1.15.21` in [backend/requirements.txt](../backend/requirements.txt) because the patch touches its internals.
- **Trade-off:** we lose Anthropic prompt caching, which HemoSmart doesn't use.
- **Revert:** delete the two lines once CrewAI strips the key for LiteLLM providers, then unpin.

### D42. Rate-limit-aware retries and a lazy fallback
- **Finding:** Groq's free tier allows 8,000 tokens per minute for `openai/gpt-oss-20b`, and one case needs more than that, so later agents were hitting rate limits rather than getting answers wrong.
- **Decision:**
  - a rate-limited task is retried after Groq's suggested wait (minimum 10 s, maximum 65 s, up to 4 retries);
  - the RAG step's fallback explanation is computed only if the agent fails (it used to be generated up front, wasting a ~2.5k-token call on every successful run).
- **Trade-off:** a case takes about 1 minute instead of about 15 s. A paid tier or a smaller prompt would remove the waits.
- **Tune:** `RATE_LIMIT_RETRIES`, `MIN_RATE_LIMIT_WAIT_SECONDS`, `MAX_RATE_LIMIT_WAIT_SECONDS`.

### D43. No-argument tools get a dummy parameter
- **Finding:** Groq rejects a tool schema with no properties ("'required' present but 'properties' is missing").
- **Decision:** `check_inventory_tool(scope="all")`.

### D44. The Orchestrator agent answers in plain JSON, validated in code
- **Finding:** with `output_pydantic`, CrewAI forces a hidden structured-output tool call that `gpt-oss-20b` repeatedly skipped ("Tool choice is required, but model did not call a tool").
- **Decision:** the Orchestrator task has no `output_pydantic`. The agent returns JSON, which is parsed with the PDF adapter's tolerant JSON extractor and validated against `OrchestratorResult` plus the verified facts.
- **Trade-off:** the other agents still use `output_pydantic` and could hit the same issue with another model.

---

## P2 — Evaluation

Every claim the report makes about HemoSmart now has a script that measures it. All scripts live in [evaluation/](../evaluation/), write a `.md` and a `.json` to [evaluation/results/](../evaluation/results/), and run against a throwaway SQLite database ([D54](#d54-evaluations-can-never-touch-the-real-database)).

| Script | Question | Runtime |
|---|---|---|
| `python -m evaluation.simulate` | Does the system reduce wastage without causing shortages? | ~1.5 min |
| `python -m evaluation.sensitivity` | Which outreach window / alert cap? | ~8 min |
| `python -m evaluation.prediction_benchmark` | Is the model better than a threshold rule? | ~30 s |
| `python -m evaluation.forecast_backtest` | Is Prophet better than what a blood bank does by hand? | ~3 min |
| `python -m evaluation.bandit_regret` | Does Thompson sampling pick better donors? | ~15 s |
| `python -m evaluation.rag_faithfulness [--llm]` | Does retrieval find the right pages; does the LLM stay faithful? | 10 s / ~5 min (20 LLM calls) |

### Outreach policy (fixes found by the simulation)

The first 90-day simulation (saved as [simulation_v1_initial_policy.md](../evaluation/results/simulation_v1_initial_policy.md)) showed the P1 outreach policy was wrong in two opposite ways:

| v1 (P1 policy), mean of 3 seeds | baseline | agents | hemosmart |
|---|---|---|---|
| Platelets expired | 46.7 | **70.7** | 19.3 |
| PRBC units short | 12.3 | **37.3** | 42.7 |
| FFP units short | 2.3 | **31.3** | 33.0 |

The agents collected platelets for gaps a week away (which expired before they were needed) while red cells and plasma ran short. Three fixes followed.

#### D45. Outreach acts only on gaps inside the component's shelf life
- **Decision:** the Planner turns a projected gap into a donor-outreach need only for the days inside `OUTREACH_WINDOW_DAYS` = PRBC 7, **Platelets 5**, FFP 7 (shelf life, capped by the 7-day horizon) ([bloodbank/catalog.py](../bloodbank/catalog.py), `PlannerAgent._actionable_need`). The per-need alert cap went from 8 to 16. Shortages beyond the window are still *reported* (`bb.shortages`); only `bb.needs` drives alerts.
- **Why:** a unit donated now can only cover a gap within its shelf life. Calling donors for a platelet gap on day 7 collects units that expire on day 5.
- **How it was chosen:** [evaluation/sensitivity.py](../evaluation/sensitivity.py) swept platelet windows 2–7 × caps 8/16 and scored cost = 3 × units short + units not used here (expired + shipped out). Shortage is weighted 3× because an unmet transfusion harms a patient, while a wasted unit costs a donation. Results ([sensitivity.md](../evaluation/results/sensitivity.md), cap 16):

  | Platelet window | 2 | 3 | 4 | **5** | 6 | 7 |
  |---|---|---|---|---|---|---|
  | Units short | 118.0 | 95.0 | 70.3 | **43.0** | 27.3 | 30.3 |
  | Not used here | 28.7 | 29.3 | 33.0 | **47.7** | 93.7 | 137.7 |
  | Cost | 382.7 | 314.3 | 243.9 | **176.7** | 175.6 | 228.6 |

  Window 6 scores 1 point lower on cost, which is a tie. It does so by collecting twice the units that end up expired or shipped out, for days beyond the 5-day shelf life. We kept 5: it's the principled bound, and the redistribution partner is simulated ([D29](#d29-near-expiry-surplus-is-offered-to-a-partner-hospital)), so leaning on it is the riskier bet.
- **An intermediate attempt failed:** a 2-day platelet window (v2) cut expiry but left 118 units short, worse than the baseline's 96. The diagnosis showed the cap was being hit and escalations piling up, which is what led to the sweep.
- **Trade-off:** the 3:1 weight is a value judgement. At 1:1, a shorter window would win.
- **Tune:** `OUTREACH_WINDOW_DAYS`, `DonorAgent.MAX_ALERTS_PER_SHORTAGE`, `SHORTAGE_WEIGHT` in sensitivity.py.

#### D46. Safety stock: outreach fires before stock hits zero
- **Decision:** a need also includes the amount that keeps projected end-of-day stock at `SAFETY_STOCK_DAYS` × average daily demand (PRBC 1.0, Platelets 0.5, FFP 1.0) over the window.
- **Why:** in v1, outreach reacted only to projected zero stock. With donation lead times and Poisson demand, PRBC and FFP shortages tripled. Platelets get a smaller buffer because buffer stock is exactly what expires.
- **Revert / tune:** set a component's value to 0 to react only to a projected stock-out.

#### D47. Eligibility depends on the previous donation's type
- **Decision:** `required_interval_days(component, last_donation_type)` uses 28 days for apheresis, or for any donation after an apheresis, and 90 days for whole blood after whole blood ([agents/donor_module.py](../agents/donor_module.py)). `Donor.last_donation_type` records it.
- **Why:** v1 treated a platelet (apheresis) donation as blocking whole blood for 90 days. Apheresis returns the red cells, so that was medically wrong, and it starved red-cell outreach: the donors who answered platelet calls became unusable for PRBC.
- **Trade-off:** the 28-day figure is a configurable policy (as in [D25](#d25-donation-types-and-eligibility-intervals)), not a cited regulation.

### Simulation & results

#### D48. The simulation is an ablation over the real code, with an honest wastage metric
- **Decision:** [evaluation/simulate.py](../evaluation/simulate.py) replays 90 days (from 2026-08-01, covering the dengue platelet peak), 3 seeds. Every seed uses one world for all policies: demand from `data/demand_process.py`, Poisson routine supply, the same starting stock and donor pool. Four policies each add one capability:
  - **baseline:** manual own-group FIFO plus 3 random donor calls when a cell turns Critical;
  - **allocation:** + wastage-aware allocation;
  - **agents:** + forecast, gap projection and bandit outreach;
  - **hemosmart:** + redistribution.

  All of them run through the real `bloodbank/`, `agents/autonomy/` and donor code.
- **Honest metric:** shipping a unit to the partner is not "used". The headline wastage figure is therefore *units not used here* = expired + redistributed, reported next to plain expiry. This stops redistribution from hiding over-collection.
- **Final results** ([simulation.md](../evaluation/results/simulation.md), mean ± std of 3 seeds):

  | | baseline | allocation | agents | **hemosmart** |
  |---|---|---|---|---|
  | Units expired | 46.7 ± 9.9 | 28.0 ± 6.2 | 41.3 ± 6.7 | **11.0 ± 2.2** |
  | Units redistributed | 0 | 0 | 0 | 36.7 ± 2.6 |
  | Not used here | 46.7 | 28.0 | 41.3 | 47.7 |
  | Units short | 96.0 ± 10.6 | 96.3 ± 10.3 | 39.3 ± 5.0 | **43.0 ± 8.2** |
  | Service level | 97.0% | 97.0% | 98.8% | 98.7% |
  | Donor alerts | 1130 | 1099 | 1289 | 1324 |

- **How to read it (and how the report should state it):**
  1. Compared with the manual baseline, HemoSmart **cuts unmet demand by 55%** (96 → 43 units) and **on-site expiry by 76%** (46.7 → 11.0). It does this **without increasing the total units not used here** (46.7 → 47.7, within noise).
  2. Allocation alone removes 40% of expiry at no cost. The agents are what remove shortages (96 → 39), at the price of collecting more platelets that the hospital can't use in time (41.3 expired). Redistribution is what turns that surplus into supply elsewhere.
  3. The redistribution benefit depends on the partner using the units, and the partner is simulated.
  4. 12% more donor alerts are sent (1130 → 1324).
- **Simulated, not measured:** donor replies, partner acceptance, routine supply.

#### D49. The simulation is exactly reproducible
- **Finding:** two runs with the same seed gave slightly different results, for two reasons:
  - units with equal expiry times came back from the DB in arbitrary order;
  - `project()` iterated a Python `set` of components, whose order changes with `PYTHONHASHSEED`, so the order donors were alerted in changed.
- **Decision:** `available_units` orders by `(expires_at, id)`, and `project()` walks components in the fixed `COMPONENTS` order.
- **Verified:** three different `PYTHONHASHSEED` values give identical results (seed 1 hemosmart: expired 9, short 43, transferred 33, alerts 1224).

### Prediction

#### D50. Labels are simulated clinician decisions, not the threshold rule
- **Finding:** labels were `Hb < threshold OR …`, a pure function of the features, so every model scored 100%. That number is meaningless and an examiner would spot it.
- **Decision:** [data/label_process.py](../data/label_process.py) draws each label from P(clinician orders it):
  - **context-dependent thresholds:** Hb 9 g/dL and platelets 100k for cardiac surgery, Hb 10 for emergency, now in [schema/clinical_thresholds.py](../schema/clinical_thresholds.py) and also used for explanation facts and doses;
  - **a soft boundary:** a logistic curve with width Hb 0.3 g/dL, platelets 6,000/µL, INR 0.08;
  - **1% record noise.**

  The transfusion rate is now 24%; the context-aware rule agrees with 93–98% of labels.
- **Benchmark** ([prediction_benchmark.md](../evaluation/results/prediction_benchmark.md), 5-fold CV, 5000 patients):

  | | Static rule F1 | Logistic F1 | **XGBoost F1** | Ceiling F1 |
  |---|---|---|---|---|
  | PRBC | 0.654 | 0.599 | **0.705** | 0.714 |
  | Platelets | 0.685 | 0.737 | **0.783** | 0.795 |
  | FFP | 0.740 | 0.738 | 0.721 | 0.750 |
  | Any transfusion | 0.736 | 0.747 | **0.753** | 0.770 |

  - **Static rule:** the context-free rule (one Hb threshold, 50k platelets for everyone).
  - **Ceiling:** the label generator's own probabilities, i.e. the best possible on this data.
  - XGBoost's gain is in sensitivity: PRBC 0.52 → 0.61, Platelets 0.55 → 0.73. It learns the cardiac and emergency adjustments the static rule ignores (36 PRBC and 45 platelet patients in the dataset), and comes within about 0.01 F1 of the ceiling.
  - On FFP, which has no context adjustment, XGBoost is slightly *worse* than the rule.
- **How the report should state it:** the model recovers context-dependent decision patterns from data without being given them. It does **not** beat a hand-written context-aware rule, because on this synthetic data that rule *is* the ceiling. Accuracy (~89–98%) is no longer the headline: with 6% PRBC prevalence, accuracy mostly reflects the negatives.
- **Trade-off:** the context adjustments are the benchmark's design, so "the model beats the static rule" is partly built in. What the benchmark really shows is that the model *finds* them, and how close it gets.
- **Revert:** `git show HEAD:data/create_dataset.py`, then regenerate and retrain.

### Forecasting

#### D51. Prophet is kept; the LSTM is not brought back
- **Benchmark** ([forecast_backtest.md](../evaluation/results/forecast_backtest.md)): rolling origin, 15 origins every 14 days through the second half of 2025, a 7-day horizon, every method refit at each origin. MAE in units/day:

  | | Last week | 4-week weekday mean | **Prophet** | LSTM | Noise floor |
  |---|---|---|---|---|---|
  | PRBC | 4.71 | 3.45 | **3.44** | 3.42 | 3.29 |
  | Platelets | 3.43 | 2.67 | **2.41** | 2.69 | 2.32 |
  | FFP | 2.96 | 2.27 | **2.08** | 2.56 | 2.01 |
  | Platelets, dengue window | 4.52 | 3.35 | **2.87** | 3.53 | 2.69 |

  The *noise floor* is the error of the true expected demand: Poisson variation no forecast can remove.
- **Reading:**
  - Prophet is within 0.07–0.15 units/day of the noise floor on every component.
  - It beats the manual 4-week weekday mean by 10% on platelets, rising to 14% in the dengue window, because it anticipates the seasonal spike instead of lagging it.
  - On PRBC it ties the simple mean.
  - The LSTM (small PyTorch model, retrained per origin, no TensorFlow needed) is never better than Prophet and is worse on platelets and FFP.
- **Decision:** keep Prophet ([D18](#d18-top-down-hierarchical-forecast-prophet-per-component-split-by-group-share)). The LSTM stays a benchmarked comparison and is not served ([D20](#d20-the-lstm-is-no-longer-served)).
- **Caveats:**
  - Synthetic demand has exactly the seasonality Prophet models; real demand would be harder.
  - At the 2025 backtest origins Prophet has under two years of history, and it warns that yearly seasonality may be under-identified. The dengue peak is learned from a single previous season.

### RAG faithfulness

#### D52. Protocol thresholds are attributed to HemoSmart, not to WHO; retrieval uses the handbook's vocabulary
- **Finding 1 (attribution):** several protocol numbers are **not in the WHO handbook** the index cites. They came from placeholder text (`rag/guideline_reference.py`):
  - INR > 1.5 for FFP: the handbook says INR < 2.0 before most surgery (p.162) and gives FFP for coagulation factor deficiencies (p.30);
  - platelets 100k for cardiac surgery: the handbook gives 50 ×10⁹/L with bleeding and 10–20 prophylactically (p.77, p.102, p.147, p.162);
  - Hb 8 and 10 are consistent with the handbook's "approximately 7–8 g/dl" for minor surgery and "higher" with cardiorespiratory disease or major surgery (p.160), but the exact numbers are ours.

  The old prompt called the protocol "the guidelines", and the LLM wrote "WHO guidelines state…".
- **Finding 2 (retrieval):** using a hand-labelled set of relevant pages (`RELEVANT_PAGES` in [rag_faithfulness.py](../evaluation/rag_faithfulness.py)), the old plasma query ("INR fresh frozen plasma transfusion threshold") **never** retrieved a plasma page in its top 2 (hit@2 = 0.00). The handbook writes "haemoglobin" and "50 x 10⁹/L", while the queries used American spelling and generic wording.
- **Decision:**
  1. Threshold facts now say "the HemoSmart protocol threshold". The [clinical_thresholds.py](../schema/clinical_thresholds.py) docstring lists, page by page, what the handbook does and doesn't say.
  2. A new prompt separates "HemoSmart protocol checks" from "WHO handbook excerpts", and forbids crediting WHO with a number that isn't in a cited excerpt.
  3. `parameter_queries()` uses the handbook's vocabulary. Plasma hit@2 goes 0.00 → 1.00, surgery-chapter MRR 0.46 → 0.58; Hb and platelets were already at 1.00.
- **Faithfulness results** ([rag_faithfulness.md](../evaluation/results/rag_faithfulness.md), `openai/gpt-oss-20b`, the same 10 cases per prompt, every flag checked by hand):

  | Explanations with… | Legacy | Current |
  |---|---|---|
  | a number credited to WHO/guidelines that no excerpt contains | 2 | 1 (borderline) |
  | a number from nowhere in the grounding | 0 | 0 |
  | a threshold claim contradicting the protocol checks | 0 | 1 |
  | at least one citation | 9 | 10 |

  - **Legacy misattributions (both real):** "a level at which guidelines recommend" (for the protocol's 10 g/dL) and "According to WHO guidance… FFP is indicated only when INR exceeds 1.5".
  - **Current misattribution (borderline):** a sentence cites WHO for platelet therapy and, in another clause, the protocol's 50,000/µL.
  - **Current contradiction (real):** it says INR 1.1 is "above" its threshold, although the conclusion (no FFP) is right.
  - **Reading:** the current version attributes protocol numbers to HemoSmart by name in every case and stops crediting WHO with INR 1.5. But on 10 cases the difference is small and not statistically meaningful, and the LLM can still slip on direction words.
  - **Mitigation:** every number shown to the user also appears in the code-computed `threshold_facts` next to the explanation in the UI.
- **Checker false positives were fixed before scoring:** "32 000/µL", "×10³" and "×10⁹/L" were parsed wrongly, and "at or above the threshold" was read as crossing it. The checks were rerun on the saved explanations with `--rescore`, which costs no LLM calls.
- **Caveats:**
  - I wrote the relevance labels and the new queries after reading those pages, so retrieval was tuned on its own test set. It's 4 query families, not a held-out benchmark.
  - The faithfulness checks are regex-based. They catch numeric misattribution and contradictions, not every unsupported qualitative claim.
- **For the report:** HemoSmart is a *protocol* grounded in and cited against the WHO handbook, not a WHO implementation. Say which numbers are local.
- **Revert:** pass `prompt_template=` / `queries=` to `ExplanationGenerator`; the legacy versions are kept in rag_faithfulness.py.

### Donor bandit

#### D53. Thompson sampling vs simpler policies: honest result
- **Benchmark** ([bandit_regret.md](../evaluation/results/bandit_regret.md)): 400 donors, 5 alerts/day for 365 days, 20 seeds; a donor who donates is ineligible for 90 days. Donations:

  | | Random | Greedy | ε-greedy | **Thompson (HemoSmart)** | Oracle |
  |---|---|---|---|---|---|
  | With prior history | 823 | 1003 | 993 | **979** | 1083 |
  | Cold start | 823 | 947 | 941 | **928** | 1083 |
  | Drift (⅓ of rates change at day 180) | 821 | 961 | 954 | **948** | 1082 |

- **Reading:**
  - Learning donor behaviour matters: Thompson gets **19% more donations than random** (2.22 → 1.86 alerts per donation), which is the comparison with the baseline's random calls.
  - But **plain greedy beats Thompson by 1.5–2.5%** in every scenario, including drift. The 90-day ineligibility forces rotation through the pool, which explores for free, so Thompson's extra exploration costs donations here.
- **Why greedy wins here:** with a Beta(1, 1) prior, an untried donor scores 0.5, which is above any donor with a poor record, so greedy already tries newcomers. The 90-day ineligibility then rotates the best donors out of the pool. Together these give greedy the exploration Thompson pays for.
- **Decision (project owner, 2026-10-13):** keep Thompson sampling, and report the comparison honestly.
  - The gap is small: about 23 donations a year, 1.82 vs 1.86 alerts per donation.
  - Thompson is the standard choice for this problem, and the report names it.
  - It should be more robust to things this simulation doesn't model: donor fatigue (greedy keeps messaging the same top donors) and a steady inflow of new donors.
- **Revert:** to switch, rank by `posterior_mean` instead of `thompson_sample_score` in `select_top_donors` (one line) and call it "Bayesian greedy".
- **Proposed experiment, not yet run: donor fatigue.** This is the evidence for the robustness argument above; until it's run, that argument is a hypothesis.
  - **The scenario:** add a `fatigue` scenario to [evaluation/bandit_regret.py](../evaluation/bandit_regret.py). Each alert a donor received in the last 30 days lowers their reply probability, e.g. `rate × 0.85^recent_alerts`, and the effect recovers as alerts age out of the window.
  - **The test:** greedy concentrates alerts on the same top donors, so fatigue should hit it harder than Thompson. If Thompson wins under fatigue, the decision above is backed by data. If not, revisit (b).
  - **Also report:** how evenly alerts are spread, e.g. the Gini coefficient of alerts per donor, as a donor-experience metric.
  - **Cost:** about 20 lines plus one more scenario row, ~5 s of runtime.

### Tooling

#### D54. Evaluations can never touch the real database
- **Incident:** the first bandit evaluation imported `agents.donor_module`, which initialises `db.database` and ran the additive schema sync ([D35](#d35-additive-schema-sync-instead-of-alembic)) against the developer's local Postgres. It added 11 nullable columns and any missing P1 tables. No rows were read or changed, and the backend does the same thing on its next startup, but it was unintended.
- **Decision:** [evaluation/__init__.py](../evaluation/__init__.py) points `DATABASE_URL` at a throwaway SQLite file before any evaluation module loads, so every `python -m evaluation.*` is sandboxed. Scripts that need the Groq key load `backend/.env` without overriding variables already set.

---

## P3 — Report alignment

[REPORT_ALIGNMENT.md](REPORT_ALIGNMENT.md) checks every claim in the Phase I report against the code. These entries are the gaps closed in code rather than in the text.

#### D55. PDF values are confirmed before anything is predicted (FR-03)
- **Decision:** the dashboard's PDF flow has two steps:
  1. `POST /api/predict/pdf/extract` returns the extracted values. It doesn't predict or save anything.
  2. The Upload card shows them as an editable form, and "Confirm & predict" calls `POST /api/predict` with `source_format: "pdf"`, so the record keeps its provenance.

  `source_format` only accepts `manual` or `pdf`. The one-shot `POST /api/predict/pdf` stays for API and MCP clients.
- **Why:** FR-03 asks for it, and LLM extraction can misread a report (units, a wrong row). A clinician correcting a value *before* it becomes a prediction is the safe order.
- **Trade-off:** one extra click. The patient record only exists after confirmation.

#### D56. Data minimisation for patients; donor phones masked to the last 4 digits (NFR-03)
- **Decision:**
  - Patients store no direct identifiers (no name, no hospital number), only clinical values and an internal sequence number. That is stronger than masking.
  - The real personal identifier, the donor phone, is masked by `schema.privacy.mask_identifier` (`•••••••••1358`) in the donor list API, in alert results (REST, MCP, chat) and in the dispatch log line.
- **Trade-off:** coordinators can't read a donor's number in the UI. A real SMS gateway would take the full number from the database at the dispatch point, which is the only place that needs it.

#### D57. Response times are measured, not promised (NFR-01)
- **Result** ([latency.md](../evaluation/results/latency.md), in-process through the real app, network excluded):

  | Operation | Median | p95 | Target |
  |---|---|---|---|
  | Prediction | 0.01 s | 0.01 s | < 2 s ✓ |
  | Forecast + gap, uncached | 0.07 s | 0.35 s | < 10 s ✓ |
  | RAG explanation (1 Groq call) | 2.05 s | 16.9 s | < 5 s ✗ at p95 |

- **Reading:** everything that runs locally is far within target. The explanation's *typical* time meets the target, but one of 5 calls waited ~17 s on Groq's free tier, so the p95 doesn't. The report should state the median and name the tail's cause. A paid tier or a local model would remove it.
- **Method note:** the first run reused the Redis forecast cache; each call now uses a distinct horizon, so every forecast is computed.

#### D58. Random Forest benchmarked; XGBoost kept
- **Result** ([prediction_benchmark.md](../evaluation/results/prediction_benchmark.md)): RF and XGBoost are within one standard deviation on every component. F1: PRBC 0.692 vs 0.705; Platelets 0.791 vs 0.783; FFP 0.739 vs 0.721; any 0.765 vs 0.753.
- **Decision:** keep XGBoost. Accuracy is a tie, and XGBoost gives exact per-patient SHAP natively (D12) and is already integrated. The report can say "XGBoost and Random Forest were compared; they perform equivalently."

#### D59. The forecast chart shows Prophet's 80% interval, and its calibration is measured
- **Decision:** the chart shades `yhat_lower`–`yhat_upper` (Prophet's default 80% interval) per component. The band comes from the residual noise of the fit plus simulated future trend changes; it does not include seasonality uncertainty. The backtest measures its **coverage**: how often actual demand fell inside.
- **Result** ([forecast_backtest.md](../evaluation/results/forecast_backtest.md)):

  | | FFP | PRBC | Platelets |
  |---|---|---|---|
  | Coverage, all origins | 77% | 79% | 79% |
  | Coverage, dengue window | 81% | 74% | **68%** |

  The band is honest overall but **too narrow for platelets during the dengue spike**. The chart's caption says so.
- **Caveats:**
  - Agents plan on the point forecast plus safety stock (D46), not on the band.
  - Per-group values are the component band scaled by group share (approximate).

#### D60. Session lifetime stays at 12 hours (no 15-minute idle logout)
- **Decision (project owner, 2026-10-13):** keep `ACCESS_TOKEN_EXPIRE_MINUTES = 720` so testing and demos aren't interrupted by repeated logins.
- **For the report:** drop FR-08's "terminate after 15 minutes of inactivity", or list it as future work.
- **To build later:** a frontend idle timer that calls `auth.logout()`, plus a shorter token lifetime.

#### D61. Browser walkthrough of every role
- **Method:** Playwright drove real Chrome against a throwaway SQLite copy (seeded demo accounts; never the developer's Postgres). It covered:
  - every role dashboard, plus the Admin tabs;
  - a manual prediction → reserve → confirm transfusion;
  - Explain;
  - PDF extract → confirm → predict;
  - forecast with the band, an autonomy cycle, the donor list, the audit log;
  - desktop (1366 px) and phone (390 px) widths.
- **Found and fixed:**
  - inputs spilled out of cards in the 3-column layout (`.form-grid` now uses `minmax(0, 1fr)`);
  - the transfusion and forecast tables clipped their last column (wrapped in `.table-scroll`);
  - the chart legend overlapped the caption (moved to the top);
  - a missing favicon logged a 404 on every page.
- **Result after fixes:** no console errors, no failed requests, no horizontal scroll at either width; dashboards load in 0.8–0.9 s (NFR-01 target < 3 s). The PDF flow extracted the sample report correctly, including "128 x10³" → 128,000.

#### D62. The local database is refreshed additively, never reset
- **Decision:** [scripts/rescore.py](../scripts/rescore.py) appends a prediction from the current models to each patient whose latest prediction predates per-component models (`--all` for everyone, `--dry-run` to preview). Old predictions stay in the history.
- **Applied to the developer's Postgres on 2026-10-13:**
  - backed up first with `pg_dump`;
  - 19 predictions added, and every yes/no decision agreed with the old model;
  - the idempotent seed (no `--reset`) added 144 blood units and the `@hemosmart.app` demo accounts;
  - the donor pool was topped up from 50 to 400 with the same deterministic generator (D0051–D0400), and the 50 old donors' missing simulated reply rates were filled from it.
- **Why:** `seed --reset` wipes every table, including the existing `@hemosmart-test.com` accounts. A demo database should be reset; a working one shouldn't.

#### D63. P3 cleanup
- **Removed:**
  - four empty, never-imported files (`agents/inventory_agent.py`, `orchestrator_Agent.py`, `prediction_agent.py`, `rag_agent.py`; the CrewAI agents with those names live in `crew_agents.py`);
  - the unused `ExplanationGenerator.retrieve_context*`, its `top_k` parameter, and `tools.get_blood_demand_forecast`.
- **Relabelled:**
  - `agents/lstm_forecast_standalone.py` as LEGACY (D20, D51);
  - `rag/guideline_reference.py` as fallback-only, with the protocol-vs-WHO note (D52).
- **Pydantic:** class-based `Config` → `ConfigDict`, which removes a deprecation warning. The two remaining test warnings come from third-party code (starlette's test client, the langchain-community sunset notice).
- **Cycle brief wording fixed:** the brief said donors were "alerted across N shortage(s)" when outreach had fired for a safety-stock top-up with no projected shortage. The dashboard showed "0 shortages" next to it.
  - Alerts are now described as *outreach needs*, each labelled "unmet demand", "safety-stock top-up" or both.
  - Cycle summaries gain an `outreach_needs` count.
  - This is wording only: no decision logic changed, so the simulation results stand.

#### D64. Frontend design system: HemoSmart tokens, app shell, clinical states
- **Decision:**
  - The frontend follows the HemoSmart frontend design skill: goldfish yellow for brand, actions and AI; amber, coral and red only for risk.
  - **Palette: Ivory & Ink**, chosen by the owner from six contrast-checked options that were each applied to the running app.
    - Pure white ground, near-black ink text and a pale stone secondary surface.
    - Tighter corners (6/9/12px) and no card shadows: borders and spacing carry the structure.
    - Yellow appears only on actions, AI and the current page, so it always means something.
    - Dark mode uses near-black ink surfaces.
    - This replaces the skill's default cream and beige ground. The brand yellow and every risk colour are unchanged.
  - Tokens live in [frontend/src/styles/tokens.css](../frontend/src/styles/tokens.css) as CSS variables. There's no hard-coded colour anywhere else, and every pairing passes WCAG AA.
  - Each role gets a sidebar of sections instead of one long stacked page; the section lives in the URL hash, so there's no router.
  - Shared primitives in [components/ui](../frontend/src/components/ui/index.jsx): severity badge (icon + word), skeletons, empty / error / stale states, "Updated" stamp, AI panel with the decision-support line.
- **Why CSS variables, not Tailwind:** every component already used class names against one stylesheet, so variables re-skinned everything without rewriting JSX. Tailwind would have meant migrating every component for the same result.
- **Clinical rules enforced in the UI:**
  - a prediction is never shown in a success colour;
  - data, model prediction, rule-based protocol checks and AI explanation are visually separate;
  - nothing shows a placeholder number while loading;
  - a partial list (e.g. forecast failed) says it's incomplete;
  - extracted PDF values must be confirmed before predicting.
- **Skill conflict:** impeccable bans coloured left borders thicker than 1px; the HemoSmart skill uses a 3px yellow border to mark AI content and the active nav item. HemoSmart wins, being a clinical-safety signal, and the border is used *only* for those two jobs.
- **Added dependencies:** lucide-react (icons; severity must never be colour alone) and self-hosted Inter and Manrope variable fonts (identical rendering on Windows and macOS). No CDN calls.
- **Still open (needs data or a decision, not faked):**
  - High/Medium/Low confidence cut-offs;
  - the stale-data refresh interval;
  - per-field PDF extraction confidence;
  - CBC reference ranges and the upload size limit;
  - a prediction `created_at` in the API response;
  - donor eligibility status in `/api/donors`.
- **Logo ("refined tile"):** the mark is a slim ink teardrop on a goldfish-yellow tile with rounded corners, with a hairline pulse line through it. It means blood, read by data. The owner chose it from four lockups.
  - It lives in [Logo.jsx](../frontend/src/components/layout/Logo.jsx) and [public/favicon.svg](../frontend/public/favicon.svg) and uses fixed brand colours, so it looks the same in light and dark mode.
  - The wordmark is Manrope in one ink colour: "Hemo" bold, "Smart" regular.
- **Typography: Manrope + Inter**, chosen by the owner from five options shown on the running app.
  - Manrope (titles, big numbers, wordmark) and Inter (body, labels, tables, inputs) give each role its own face, so a heading never reads as body text.
  - Scale: page title 30/800, card title 17/750, body 15/400 with line-height 1.6, labels 13.5/500 in secondary ink, notes 12.5, big numbers 28/700.
  - Each step changes weight or colour as well as size.
  - Inter's `ss03` and `cv11` features make I, l and 1 easier to tell apart in lab values.
  - Both fonts are self-hosted through `@fontsource-variable`, so the app makes no font CDN calls.

---

## Role-aware assistant (FR-08)

The assistant is available to every role and answers only from what that role's pages allow. Before this, only Hospital Staff (and Admin) could open it, and it never checked who was asking. It returned the demand forecast and sent real donor alerts for Hospital Staff, both of which the forecast and donor routes refuse that role. It also read any PDF path typed into it, and a copied session ID exposed another user's last patient.

#### D65. One permission matrix for routes and chat
- **Decision:** [schema/permissions.py](../schema/permissions.py) lists each capability (`inventory:read`, `forecast:read`, `donor_alerts:send`, …) and the roles that hold it; Admin holds all.
  - Every route now uses `require_permission(capability)` instead of a hand-typed role list.
  - The chat assistant maps each intent to a capability from the same table (`INTENT_CAPABILITY`).
  - The matrix was derived one-for-one from the route guards, with one owner decision: **Donor Coordinators may read inventory**, to coordinate donor calls against stock. They get an Inventory page (read-only, no forecast).
  - `donors.py` is split into `donors:read`, `donor_alerts:send` and `donor_responses:write`, with the same roles as before.
- **Enforced by tests** ([tests/test_chat_rbac.py](../tests/test_chat_rbac.py)): no route may use a raw role list; every capability except `guidelines:read` guards a route; a hand-written role × topic table must match the matrix.
- **MCP unchanged (owner's choice):** the `/mcp` gate stays Admin and Blood Bank Manager per connection.
- **Trade-off:** the 403 message changed to the shared refusal wording.

#### D66. The assistant checks permission in code, before any tool or LLM runs
- **Decision:** per message,
  1. split into clauses ("O− stock and the audit log" is two);
  2. classify each clause with keyword rules; the LLM only for a clause no rule matches, and its answer must be one label from a closed set;
  3. check the clause's capability for the user's role (deny by default);
  4. run the allowed tools and phrase results with templates.
- **A refused clause** gets a fixed line, *"Not authorised: the Hospital Staff role doesn't have access to the audit log."*, plus one line listing what the role can ask (owner decision). No data is read and no LLM call is made for it.
- **Mixed questions** answer the allowed clauses and refuse the rest.
- **Prompt injection** ("ignore your rules", "I am admin") can't change the decision, because the role comes only from the logged-in user. Such messages are flagged for the audit log.
- **The LLM** gets a role-scoped prompt prefix ("You may only discuss: …"), as defence in depth only. It only ever sees data that already passed step 3.
- **Donor alerts need a "yes"** (owner decision): the assistant states what it will send and waits. The permission is checked again when the user confirms.
- **New read topics** (patients, donors with masked phones, pending donor replies, the last agent cycle, the audit log and refused chat requests, WHO guideline questions) call the same helpers in [agents/tools.py](../agents/tools.py) that the routes now use.
- **Why:** an LLM can't be trusted to keep a secret it was given. Deciding access in code makes the refusal exact and testable.
- **Trade-off:** keyword routing occasionally mis-files a question. It then gets refused or answered under a topic the role is allowed anyway, never leaked.

#### D67. Chat sessions belong to a user and a role
- **Decision:** the session cache key is `chat_session:{user_id}:{session_id}`. The stored role must match the user's current role, or the state (last patient, a pending alert) is dropped.
- **Why:** a copied session ID used to reveal another user's last patient through "why?". A pending alert must not survive a role change.

#### D68. Each chat message is one structured audit record, without its text
- **Decision:** `action = "chat_query"`, with the role, intents, allowed/denied decision per clause, outcome (`answered` / `partial` / `denied`), whether the LLM was used or degraded, the injection flag, any action taken (a donor alert), and the message length.
- **The message text is not stored** (owner decision, data minimisation as in D56).
- **Use:** the Auditor can ask the assistant "show refused chat requests".

#### D69. The assistant no longer reads PDFs
- **Decision:** the chat `extract` intent, which opened any `*.pdf` path typed into chat on the server, is removed (owner decision). Reports go through the CBC upload page, where the clinician confirms the values (D55).

#### D70. When the LLM is down, the assistant degrades instead of erroring
- **Decision:** `/api/chat` no longer returns 502. A failed classifier call answers with the role's "what you can ask" list; a failed explanation falls back to the rule-based protocol checks; a failed guideline answer shows the retrieved handbook passages. The reply is marked `degraded` and the UI says "Limited mode".
- **Known gap:** a pending donor alert lives in the Redis session. Without Redis, "yes" finds nothing to confirm and nothing is sent, which fails safe.

---

## Agent monitor

A live page showing whether the agents are running, how long each step takes, how often the LLM agents' answers check out, and what needs a person. It is a flat (2D) take on the Agents Office idea, planned in [2dvision.md](2dvision.md). The **Agent activity** page shows what the supply loop *decided*; the monitor shows how the agents are *operating*. It sits next to the agents, not inside them: each layer gets a few lines of hooks, and no agent logic changes.

Owner decisions (2026-10-24, all the plan's recommended defaults): a new `agents:monitor` permission; Hospital Staff don't see case cards; a separate page; polling, not push; 14-day retention; stuck after 5 min (loop step) or 6 min (case step); the filename leak fixed; CrewAI telemetry off; orphaned cycles marked failed at startup; an unanswered donor alert needs attention after 24 h; no Groq budget spent on an evidence table.

#### D71. Agent telemetry is its own append-only table, written through one `emit()` seam
- **Decision:** a new `agent_events` table ([db/models.py](../db/models.py) `AgentEvent`): `run_id` (`cycle:{id}` / `case:{job_id}`), `layer` (`supply_loop` / `case_pipeline` / `scheduler`), `agent`, `step`, `type` (`run_started`, `step_started`, `step_finished`, `llm_wait`, `run_finished`, `heartbeat`), `status`, `ts`, `duration_ms`, `llm_calls`, `data`.
  - Everything goes through [agents/telemetry.py](../agents/telemetry.py): `emit()` writes one row, and `step()` is a context manager that records start, end, duration and the LLM-call delta.
  - Both **never raise**, and they use their own session, so a telemetry failure can't roll back an agent's work.
  - **Off for simulation:** `emit()` does nothing in `mode="simulation"`, without a run ID, or with `HEMOSMART_AGENT_EVENTS=0`. The evaluation scripts run thousands of cycles and write zero events.
  - **Retention:** 14 days (`HEMOSMART_AGENT_EVENTS_DAYS`), pruned on every scheduler tick. That is about 500 rows a day.
- **Why:** `AgentCycle.events` is written only when a cycle ends, so nothing is visible mid-run. Case jobs live in memory, keep the last 20 and are lost on restart.
  - `AgentCycle.events` stays as it is: it is the domain reasoning log that Agent activity and the assistant read.
- **Alternatives:**
  - CrewAI's internal event bus: rejected, because CrewAI is pinned and already patched once (D41).
  - Langfuse or OpenTelemetry: rejected, because they need self-hosting, and traces would leave the system (against D56/D68). They are the production path behind the same `emit()` seam.
- **Trade-off:** `llm_calls` is a process-wide counter, so a parallel chat request can add to a step's count. Cycles and cases each run under a lock, which keeps this small. No token counts.
- **Revert:** set `HEMOSMART_AGENT_EVENTS=0`, or remove the `telemetry` calls; the table is never read by any agent.

#### D72. Telemetry payloads are allow-listed; patient data can't reach them
- **Decision:** `emit()` keeps only these `data` keys and silently drops the rest:
  - counts (`expired_units`, `donors_alerted`, `transferred_units`, `escalations`, `units`, `attempt`, `wait_s`, `recommended_alerts`, `interval_min`);
  - categories (`blood_group`, `component`, `outcome`, `trigger`, `dispatch_allowed`);
  - `exc_class`: the exception **class name only**.

  Values must be plain scalars, so a nested dict can't smuggle anything in.

  It never stores lab values, the PDF filename or path, chat text, the cycle brief, donor IDs or phones, or exception *messages* (Pydantic errors echo input values such as hemoglobin).
- **Leak fixed along the way:** `agent_case_started` audit rows stored the uploaded **filename**, and CBC files are often named after the patient (`Ravi_Kumar_CBC.pdf`). The audit row and the in-memory job now record only the file size.
- **Also:** CrewAI's anonymous telemetry is turned off (`CREWAI_DISABLE_TELEMETRY`, `OTEL_SDK_DISABLED`) before CrewAI is imported, which keeps the data-minimisation story consistent.
- **Why an allow-list:** a deny-list fails open when someone adds a new field, and an allow-list fails closed. A test passes lab values, a filename, a donor ID, a phone number and an exception message, and checks that none of them is stored.

#### D73. The monitor polls; it doesn't push
- **Decision:** `GET /api/agents/overview` every **2 s while a run is open**, every **20 s when idle**, paused while the tab is hidden. This is the pattern the upload page already uses for case status.
- **Why not SSE or WebSockets:**
  - events arrive seconds apart, so 2 s polling looks live;
  - events come from worker threads, so a push bus needs thread-to-event-loop plumbing (and Redis for more than one worker);
  - the browser's `EventSource` can't send the `Authorization` header, so it would need a token in the URL, which the audit log would record;
  - the audit middleware writes one row per connection.
- **Trade-off:** every poll is an `/api/*` request, so it writes an audit row (about 180 an hour per open tab when idle, 1,800 an hour while a run is open). This is accepted as normal page traffic, the same as the upload page's polling. Excluding read-only polling from the audit log would be a separate decision.
- **Upgrade path:** `sse_starlette` is already installed. One SSE endpoint with a short-lived stream ticket could be added later, keeping polling as the fallback.

#### D74. A new `agents:monitor` permission, for Blood Bank Manager and Auditor
- **Decision:** `agents:monitor` in [schema/permissions.py](../schema/permissions.py) (Admin implicit), guarding every route in [backend/routes/agents_monitor.py](../backend/routes/agents_monitor.py):
  - `GET /api/agents/overview`
  - `GET /api/agents/runs`
  - `GET /api/agents/runs/{run_id}`
  - `GET /api/agents/metrics`
- **Not a chat topic:** the assistant doesn't answer monitoring questions.
- **Hospital Staff** don't see case cards. They already watch their own run on the upload page, and a per-run ownership check isn't worth its tests.
- **Why not reuse `autonomy:read`:** a separate permission lets the monitor be granted on its own later, and keeps the assistant's topic list unchanged. It holds the same roles today.
- **Read-only:** the monitor links to the page where each action lives (Agent activity, Donor responses, Upload), so each action keeps its own permission check in one place.

#### D75. Agent status is derived from events, never stored
- **Decision:** each agent's status comes from its latest events ([backend/services/agent_monitor.py](../backend/services/agent_monitor.py), pure functions with an injectable clock):

  | Status | Rule |
  |---|---|
  | working | a `step_started` with no matching `step_finished` |
  | waiting | the latest event is an `llm_wait` from the last 70 s |
  | degraded | the last finished step was `fell_back` or `skipped` |
  | failed | the last finished step was `error` |
  | idle | otherwise |

- **Alerts:**
  - **stuck:** working for more than 5 min (loop step) or 6 min (case step: 4 rate-limit retries × 65 s, plus the calls);
  - **orphaned cycle:** an `AgentCycle` still `running` after 10 min. At startup, such cycles are marked `failed`;
  - **scheduler missed a tick:** now is past the last heartbeat + the interval + 5 min. An interval of 0 shows "Scheduler off", not an alert;
  - **repeated fallback:** the same agent fell back in 3 runs in a row.
- **Needs attention** also lists degraded cycles, escalations to the regional bank, case runs whose donor alerts were only recommended, and donor alerts unanswered after 24 h.
- **Not shown:** chat "yes" confirmations. They are per user and per session in Redis (D67, D70), so there is no global list to show.
- **Why derived:** stored status drifts when a process dies. Deriving it from events plus the clock means a crash shows up as "stuck", not as a frozen "working".
- **Restart:** at startup, every run with no `run_finished` event is closed as failed ("interrupted"), because a single API process means it died with the old process.
- **Severity:** stuck, orphaned and missed-tick alerts are *High*, fallbacks and escalations *Moderate*, the rest *Info*. *Critical* stays reserved for clinical act-now states (D64).
- **Kind:** the supply loop's Explanation agent is shown as an LLM agent (yellow AI border), because an LLM writes its brief. The other supply-loop agents are rule-based.
- **LLM calls today** counts agent steps against `HEMOSMART_LLM_DAILY_LIMIT` (default 1,000, Groq's free-tier requests per day for the model; check the console if the plan changes). The assistant's LLM use is shown on its own card.
- **Tune:** the thresholds are constants at the top of `agent_monitor.py`.

---

## LLM call budget

| Feature | LLM calls | Notes |
|---|---|---|
| Manual / CSV prediction | 0 | XGBoost only |
| PDF upload (normal) | 1 | PDF text → JSON |
| Explain button | 1 | + FAISS retrieval |
| Chat | 0–1 per clause | keyword routing first; LLM only for unclear clauses (+1 for an explanation or a guideline question); 0 for any refused clause (D66) |
| Autonomous cycle | 0–1 | optional brief; citations are retrieval-only |
| **CrewAI full case** | **measured 16–30 requests, ~1 min** | budgeted ~12–17 successful calls (Extraction 4–5 · Prediction 2–3 · RAG 3 · Inventory 2–3 · Action 2–3 · Orchestrator 1); the rest are rate-limited retries (D42) |
| MCP `explain_prediction` | 1 | other MCP tools: 0 |

The provider is Groq (`HEMOSMART_LLM_PROVIDER`, default `groq`; model `openai/gpt-oss-20b`). Ollama is supported locally via [rag/llm_client.py](../rag/llm_client.py).

---

## Known limitations

Status at the end of P2 (open items first):

- **Bandit:** greedy beats Thompson sampling by ~2% in simulation. Thompson is kept by the owner's decision; the donor-fatigue experiment that would justify it is proposed but not yet run ([D53](#d53-thompson-sampling-vs-simpler-policies-honest-result)).
- **Protocol vs WHO:** INR 1.5 and the 100k cardiac platelet threshold are local protocol, not WHO handbook numbers ([D52](#d52-protocol-thresholds-are-attributed-to-hemosmart-not-to-who-retrieval-uses-the-handbooks-vocabulary)). The report must word this carefully.
- **Synthetic everything:** labels, demand, donor behaviour, partner acceptance and routine supply are all simulated. Every result says "on synthetic data", and the scripts make the assumptions explicit.
- **RAG evaluation is small:** 4 query families with hand-labelled pages, and 10 LLM cases checked by regex. It was tuned on its own test set.
- **Redistribution benefit** assumes the partner uses what it accepts ([D48](#d48-the-simulation-is-an-ablation-over-the-real-code-with-an-honest-wastage-metric)).
- **Escalations repeat each cycle** while a gap persists (logged only, no dedupe).
- **Frontend:** build-checked and API-verified, not browser-tested.
- **Existing databases** keep predictions made by the earlier (100%-accuracy) models until re-seeded (`python scripts/seed.py --reset --yes`).

Resolved in P2: circular labels ([D50](#d50-labels-are-simulated-clinician-decisions-not-the-threshold-rule)); the LSTM benchmarked ([D51](#d51-prophet-is-kept-the-lstm-is-not-brought-back)); RAG overclaiming WHO ([D52](#d52-protocol-thresholds-are-attributed-to-hemosmart-not-to-who-retrieval-uses-the-handbooks-vocabulary)); simulated parts are now quantified by the simulation ([D48](#d48-the-simulation-is-an-ablation-over-the-real-code-with-an-honest-wastage-metric)).
