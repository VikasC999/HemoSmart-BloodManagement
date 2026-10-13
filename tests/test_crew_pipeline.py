"""
Offline checks for the CrewAI case pipeline's guardrails. The agents
themselves need Groq and are exercised manually (see
docs/DESIGN_DECISIONS.md D40-D44 for measured runs).
"""

import pytest

crew = pytest.importorskip("agents.crew_agents")


@pytest.mark.parametrize("message, expected", [
    ("RateLimitError: Rate limit reached ... Please try again in 7.41s.", 10.0),   # floor applies
    ("RateLimitError: Rate limit reached ... Please try again in 21.3s.", 22.3),
    ("RateLimitError: Rate limit reached ... Please try again in 1m30s.", 65.0),   # capped
    ("RateLimitError: Rate limit reached (no hint)", 21.0),
])
def test_rate_limit_wait(message, expected):
    assert crew._rate_limit_wait(Exception(message)) == pytest.approx(expected)


def test_other_errors_are_not_retried():
    assert crew._rate_limit_wait(ValueError("Tool choice is required")) is None


def test_recommend_only_mode_never_contacts_donors(monkeypatch):
    sent = []
    monkeypatch.setattr(crew, "_dispatch", lambda key: sent.append(key) or "sent")
    tool, recorded = crew.make_idempotent_action_tool({"O- PRBC"}, allow_dispatch=False)
    out = tool.run(blood_type="O- PRBC")
    assert "RECOMMENDED" in out and sent == [] and recorded == {"O- PRBC"}


def test_dispatch_guard_rejects_unverified_and_duplicates(monkeypatch):
    sent = []
    monkeypatch.setattr(crew, "_dispatch", lambda key: sent.append(key) or "sent")
    tool, _ = crew.make_idempotent_action_tool({"O- PRBC"}, allow_dispatch=True)
    assert "REJECTED" in tool.run(blood_type="B+ PRBC")
    tool.run(blood_type="O- PRBC")
    assert "DUPLICATE" in tool.run(blood_type="O- PRBC")
    assert sent == ["O- PRBC"]


def test_crewai_cache_marker_is_neutralised():
    import crewai.llms.cache as cache
    assert "cache_breakpoint" not in cache.mark_cache_breakpoint({"role": "system", "content": "x"})
