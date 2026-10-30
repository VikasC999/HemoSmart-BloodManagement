"""Donor notifier plugin: alerts go through the interface, and a failed
provider never loses the alert."""

from datetime import date, datetime

import pytest

from agents.donor_module import alert_donors
from db.database import SessionLocal
from db.models import Donor, DonorAlert
from integrations.notify import DonorNotifier, get_notifier

NOW = datetime(2026, 10, 30, 9, 0)


class Recorder(DonorNotifier):
    method = "sms"

    def __init__(self, fail=False):
        self.sent, self.fail = [], fail

    def send(self, phone, name, message):
        if self.fail:
            raise ConnectionError("provider down")
        self.sent.append((phone, message))


@pytest.fixture
def donor():
    with SessionLocal() as db:
        db.query(DonorAlert).filter(DonorAlert.donor_id == "D9002").delete()
        db.query(Donor).filter(Donor.donor_id == "D9002").delete()
        db.add(Donor(donor_id="D9002", name="Notify Test", phone="+919876500002", blood_type="AB-",
                     last_donation_date=date(2026, 1, 1), alerts_sent=0, alerts_responded=0))
        db.commit()
    yield
    with SessionLocal() as db:
        db.query(DonorAlert).filter(DonorAlert.donor_id == "D9002").delete()
        db.query(Donor).filter(Donor.donor_id == "D9002").delete()
        db.commit()


def _alert(notifier):
    only_ours = lambda eligible, k: [d for d in eligible if d["donor_id"] == "D9002"]
    with SessionLocal() as db:
        alert_donors(db, "AB-", "PRBC", 1, NOW, selector=only_ours, notifier=notifier)
        db.commit()
        return db.query(DonorAlert).filter(DonorAlert.donor_id == "D9002").one().method


def test_alert_goes_through_the_notifier(donor):
    notifier = Recorder()
    assert _alert(notifier) == "sms"
    assert notifier.sent and notifier.sent[0][0] == "+919876500002" and "AB-" in notifier.sent[0][1]


def test_failed_delivery_still_records_the_alert(donor):
    assert _alert(Recorder(fail=True)) == "log"


def test_default_is_log_and_unknown_is_refused(monkeypatch):
    monkeypatch.delenv("HEMOSMART_NOTIFIER", raising=False)
    assert get_notifier().method == "log"
    with pytest.raises(ValueError):
        get_notifier("carrier-pigeon")
