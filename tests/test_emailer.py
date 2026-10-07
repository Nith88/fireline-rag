import smtplib

import pytest

from app import emailer
from app.analytics import Report, ReportSection
from app.config import get_settings

REPORT = Report(title="Checkout outage\r\nBcc: evil@x.com", summary="40 minutes.", sections=[ReportSection(heading="Cause", body="Pool.")], incident_refs=[])
ALLOW = "ops@acme.com, @example.org"


@pytest.fixture
def smtp(monkeypatch):
    s = get_settings()
    for k, v in dict(smtp_host="smtp.test", smtp_port=587, smtp_user="u", smtp_password="p",
                     smtp_from="fireline@test.com", email_allowlist=ALLOW).items():
        monkeypatch.setattr(s, k, v)
    events = []

    class FakeSMTP:
        def __init__(self, host, port, timeout=None, context=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def starttls(self, context=None):
            events.append("starttls")

        def login(self, user, password):
            events.append(("login", user))

        def send_message(self, msg):
            events.append(msg)

    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    return events


@pytest.mark.parametrize("addr,ok", [
    ("ops@acme.com", True), ("OPS@ACME.COM", True), ("anyone@example.org", True),
    ("other@acme.com", False), ("ops@acme.com.evil.com", False), ("x@sub.example.org", False),
    ("ops@acme.com,evil@x.com", False), ("ops@acme.com\nBcc: evil@x.com", False), ("not-an-email", False), ("", False),
])
def test_allowlist(addr, ok):
    assert emailer.is_allowed(addr, emailer.parse_allowlist(ALLOW)) is ok


def test_sends_a_templated_message_over_starttls(smtp):
    emailer.send_report("ops@acme.com", REPORT, ["postmortem.pdf"])
    msg = smtp[-1]
    assert smtp[:2] == ["starttls", ("login", "u")] and msg["To"] == "ops@acme.com"
    assert "\n" not in msg["Subject"] and msg["Bcc"] is None  # header injection through the title is neutralised
    assert "Sources: postmortem.pdf" in msg.get_content()


def test_refuses_recipients_outside_the_allowlist(smtp):
    with pytest.raises(emailer.EmailError, match="allowed recipients"):
        emailer.send_report("someone@gmail.com", REPORT, [])
    assert smtp == []


def test_not_configured(monkeypatch):
    monkeypatch.setattr(get_settings(), "smtp_host", None)
    assert not emailer.email_configured()
    with pytest.raises(emailer.EmailError, match="not configured"):
        emailer.send_report("ops@acme.com", REPORT, [])


def test_smtp_failure_does_not_leak_details(smtp, monkeypatch):
    def boom(self, msg):
        raise smtplib.SMTPAuthenticationError(535, b"bad credentials for u/p")

    monkeypatch.setattr(smtplib.SMTP, "send_message", boom)
    with pytest.raises(emailer.EmailError) as e:
        emailer.send_report("ops@acme.com", REPORT, [])
    assert "credentials" not in str(e.value)
