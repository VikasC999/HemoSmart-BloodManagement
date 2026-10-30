"""
Donor Notifier -- how a donor alert actually reaches the donor.

Today every alert is logged (LogNotifier): the alert is recorded, the
bandit learns from the reply the coordinator enters, and the message
text is printed with the phone masked. Sending real SMS or WhatsApp is
a plugin: implement DonorNotifier.send() for a provider (Twilio, MSG91,
a WhatsApp Business sandbox), register it in NOTIFIERS, and set
HEMOSMART_NOTIFIER=<name>. No calling code changes -- alert_donors()
only ever talks to this interface.

A provider failure never loses the alert: it is recorded with the
log method and the failure is printed, so the coordinator still sees
who was meant to be contacted.
"""

import os
from abc import ABC, abstractmethod

from schema.privacy import mask_identifier


class DonorNotifier(ABC):
    method: str = "unknown"  # stored on DonorAlert.method

    @abstractmethod
    def send(self, phone: str, name: str, message: str) -> None:
        """Delivers one message to one donor. `phone` is the full number
        (a provider needs it); never log it unmasked. Raise on failure."""
        raise NotImplementedError


class LogNotifier(DonorNotifier):
    method = "log"

    def send(self, phone: str, name: str, message: str) -> None:
        print(f"[DONOR ALERT] To {name} ({mask_identifier(phone)}): {message}")


# Plugins register here, e.g. NOTIFIERS["sms"] = TwilioNotifier.
NOTIFIERS = {"log": LogNotifier}


def get_notifier(name: str = None) -> DonorNotifier:
    name = (name or os.environ.get("HEMOSMART_NOTIFIER", "log")).lower()
    if name not in NOTIFIERS:
        raise ValueError(f"Unknown notifier '{name}'. Available: {', '.join(sorted(NOTIFIERS))}.")
    return NOTIFIERS[name]()
