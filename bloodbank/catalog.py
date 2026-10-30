"""
HemoSmart - Blood Bank Catalog
----------------------------------
The fixed domain facts every other blood-bank module builds on: blood
groups, the components HemoSmart tracks, how long each one keeps, when
it counts as near-expiry, and who can receive what.

Shelf lives (storage conditions per the WHO handbook indexed for RAG,
rag/who_guidelines.pdf):
  - PRBC: 42 days at +2 to +6 C in SAGM additive solution (WHO pp.24-25
    gives storage "same as whole blood"; 42 days is the standard
    additive-solution limit).
  - Platelets: 5 days at +20 to +24 C with agitation. WHO p.27 gives
    72 hours "unless collected in specialized platelet packs validated
    for longer storage" -- the 5-day validated packs used in Indian
    blood banks, matching the report's "five to seven days".
  - FFP: 1 year at -25 C or colder (WHO p.30).

Cryoprecipitate is not tracked: its indication (low fibrinogen) needs a
lab value HemoSmart doesn't collect, so it could never be predicted --
it would only ever sit in stock and expire.
"""

BLOOD_GROUPS = ("A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-")
COMPONENTS = ("PRBC", "Platelets", "FFP")

SHELF_LIFE_DAYS = {"PRBC": 42, "Platelets": 5, "FFP": 365}

# How close to expiry a unit must be before it's issued ahead of better
# ABO matches and offered to partner hospitals. Component-specific: a
# flat 7-day window would mark every platelet unit "near expiry" from
# the day it was collected. Platelets use the 48-hour window from the
# report's Inventory Module; PRBC uses the report's 7 days (FR-06).
NEAR_EXPIRY_DAYS = {"PRBC": 7, "Platelets": 2, "FFP": 30}

# Target stock, in days of forecast demand, before a group/component
# counts as "Low". Below half of it is "Critical".
DAYS_OF_COVER = {"PRBC": 3, "Platelets": 2, "FFP": 3}
MIN_PAR_UNITS = 2  # rare groups still need a floor, however low their demand

# Routine supply: the units regular voluntary donation and scheduled
# collection camps bring in without any shortage-driven outreach, as a
# share of average daily demand. The gap projection counts this inflow
# -- otherwise a normal few-days-of-cover buffer looks like a crisis
# over a 7-day horizon, and every group would trigger donor alerts.
# Below 1.0 on purpose: routine collection alone doesn't quite keep
# pace, which is the gap targeted outreach exists to close.
ROUTINE_SUPPLY_SHARE = {"PRBC": 0.85, "Platelets": 0.8, "FFP": 0.85}

# Donor outreach acts on a gap only if it falls within this many days:
# the component's shelf life, capped by the 7-day forecast horizon. A
# unit donated now can cover any gap inside its shelf life; alerting for
# gaps beyond it collects units that expire first. Chosen by
# evaluation/sensitivity.py: for platelets (alert cap 16), windows of
# 2-4 days left 56-87 units short over 90 days, 5 days (= shelf life)
# cut that to 46. 6 days cost more (200.8 vs 189.8) and nearly doubled
# the units expiring or shipped out (94 vs 53); 7 days went to 132 --
# so the window stops at the shelf life. (docs/DESIGN_DECISIONS.md D45;
# re-run with the D83 donor pool)
OUTREACH_WINDOW_DAYS = {"PRBC": 7, "Platelets": 5, "FFP": 7}

# Demand level the gap projection plans for: None = the forecast mean;
# a quantile q (e.g. 0.8) = cumulative demand that q of days stay under,
# by a Poisson approximation per blood group (bloodbank/projection.py,
# quantile_demand). Chosen by evaluation/simulate.py --planning-sweep
# (docs/DESIGN_DECISIONS.md D81).
PLANNING_DEMAND_QUANTILE = None

# A donor alerted in the last CONTACT_COOLDOWN_DAYS days isn't alerted again,
# whatever the selection policy. Chosen by evaluation/simulate.py
# --cooldown-sweep as the longest cooldown that doesn't raise units short
# against no cooldown; 14 days, which helped in the fixed-budget bandit
# evaluation, starved outreach to small blood groups in the 90-day
# simulation (docs/DESIGN_DECISIONS.md D85). 0 switches it off.
CONTACT_COOLDOWN_DAYS = 7

# Safety stock, in days of forecast demand: outreach also fires when
# projected stock dips below this, not only when it would hit zero --
# reacting only at zero let red-cell and plasma shortages triple in
# simulation. Smaller for platelets, where buffer stock is what expires.
SAFETY_STOCK_DAYS = {"PRBC": 1.0, "Platelets": 0.5, "FFP": 1.0}

# Donations: a whole-blood donation is separated into one PRBC and one
# FFP unit; platelets come from an apheresis donation (one adult dose).
DONATION_YIELD = {
    "PRBC": ("PRBC", "FFP"),
    "FFP": ("PRBC", "FFP"),
    "Platelets": ("Platelets",),
}

# Red cells: the donor's cells must carry no antigen the recipient
# lacks. Ordered identical-first.
_RBC_DONORS = {
    "O-": ("O-",),
    "O+": ("O+", "O-"),
    "A-": ("A-", "O-"),
    "A+": ("A+", "A-", "O+", "O-"),
    "B-": ("B-", "O-"),
    "B+": ("B+", "B-", "O+", "O-"),
    "AB-": ("AB-", "A-", "B-", "O-"),
    "AB+": ("AB+", "AB-", "A+", "A-", "B+", "B-", "O+", "O-"),
}

# Plasma: the reverse -- the donor's plasma must carry no antibody
# against the recipient's red cells. RhD doesn't matter for plasma.
_PLASMA_DONOR_ABO = {
    "O": ("O", "A", "B", "AB"),
    "A": ("A", "AB"),
    "B": ("B", "AB"),
    "AB": ("AB",),
}


def abo(group: str) -> str:
    return group[:-1]


def is_rh_negative(group: str) -> bool:
    return group.endswith("-")


def compatible_donor_groups(recipient: str, component: str) -> tuple:
    """Donor groups whose `component` this recipient can receive,
    identical group first."""
    if recipient not in BLOOD_GROUPS:
        raise ValueError(f"Unknown blood group: {recipient}")
    if component not in COMPONENTS:
        raise ValueError(f"Unknown component: {component}")

    if component == "PRBC":
        return _RBC_DONORS[recipient]

    allowed_abo = _PLASMA_DONOR_ABO[abo(recipient)]
    groups = [g for g in BLOOD_GROUPS if abo(g) in allowed_abo]
    if component == "Platelets" and is_rh_negative(recipient):
        # Platelet packs carry a few red cells, so an RhD-negative
        # recipient gets RhD-negative platelets (anti-D sensitization).
        groups = [g for g in groups if is_rh_negative(g)]
    groups.sort(key=lambda g: g != recipient)
    return tuple(groups)
