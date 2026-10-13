"""
HemoSmart - Identifier Masking (NFR-03)
-------------------------------------------
Screens and logs that don't need a person's full identifier show only
its last 4 characters. Patients carry no direct identifiers at all
(no name, no hospital number -- data minimisation), so in practice this
covers donor phone numbers: the dispatch gateway is the only component
that would need the full number.
"""

MASK_CHAR = "•"


def mask_identifier(value, keep: int = 4) -> str:
    """'+91XXXXX01358' -> '•••••••••1358'. Values of `keep` characters or
    fewer are masked entirely, so nothing short leaks whole."""
    if value is None:
        return None
    text = str(value)
    if len(text) <= keep:
        return MASK_CHAR * len(text)
    return MASK_CHAR * (len(text) - keep) + text[-keep:]
