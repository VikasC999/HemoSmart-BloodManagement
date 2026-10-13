"""
HemoSmart - Wastage-Aware Unit Allocation (FEFO)
----------------------------------------------------
Decides WHICH physical units go to a patient. Pure functions over any
objects with `blood_group`, `component` and `expires_at` attributes
(ORM rows in the app, plain stand-ins in tests).

Policy, in priority order:
  1. Only ABO/RhD-compatible units (bloodbank.catalog).
  2. Near-expiry units first, from ANY compatible group -- a compatible
     unit that will expire in 36 hours is better issued now than wasted.
  3. Otherwise the patient's own group first, which keeps universal
     units (O- red cells, AB plasma) for the patients who can only
     receive those.
  4. Within each tier: First-Expired-First-Out.
"""

from datetime import datetime, timedelta

from bloodbank.catalog import NEAR_EXPIRY_DAYS, compatible_donor_groups


def is_near_expiry(unit, now: datetime) -> bool:
    return unit.expires_at - now <= timedelta(days=NEAR_EXPIRY_DAYS[unit.component])


def rank_units(units, recipient_group: str, component: str, now: datetime) -> list:
    """Compatible, unexpired units of `component`, best-to-issue first."""
    compatible = set(compatible_donor_groups(recipient_group, component))
    candidates = [
        u for u in units
        if u.component == component and u.blood_group in compatible and u.expires_at > now
    ]
    return sorted(
        candidates,
        key=lambda u: (
            not is_near_expiry(u, now),
            u.blood_group != recipient_group,
            u.expires_at,
        ),
    )


def choose_units(units, recipient_group: str, component: str, count: int, now: datetime) -> list:
    """The `count` units to issue, or fewer if stock can't cover it."""
    return rank_units(units, recipient_group, component, now)[:count]
