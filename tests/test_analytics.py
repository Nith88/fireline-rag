import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.analytics import AnalyticsAgent, Report, ReportSection, build_overview, fetch_records
from app.ids import tenant_uuid
from app.models import Scope

pytestmark = pytest.mark.usefixtures("db")


def scope(tenant="acme", **kw) -> Scope:
    return Scope(tenant_id=tenant_uuid(tenant), **kw)


class ScriptedToolLLM:
    """Calls the given tools on the first turn, then stops (what a real tool-calling model does)."""

    def __init__(self, calls):
        self.calls, self.turn = calls, 0

    def bind_tools(self, tools):
        def step(_messages):
            self.turn += 1
            if self.turn == 1:
                return AIMessage(content="", tool_calls=[{"name": n, "args": a, "id": f"c{i}"} for i, (n, a) in enumerate(self.calls)])
            return AIMessage(content="DONE")

        return RunnableLambda(step)


def report(text="Five VPN incidents.", refs=("INC1001",)):
    return Report(title="Overview", summary=text, sections=[ReportSection(heading="VPN", body=text)], incident_refs=list(refs))


def agent(calls=(("incident_counts", {"group_by": "severity"}),), out=None, seen=None):
    def write(messages):
        if seen is not None:
            seen.append(messages[-1].content)
        return out or report()

    return AnalyticsAgent(tool_llm=ScriptedToolLLM(list(calls)), report_llm=RunnableLambda(write), search_factory=lambda s: lambda q, k: [])


def test_records_are_tenant_scoped():
    assert {r["ref"] for r in fetch_records(scope("globex"))} == {"INC2001"}
    assert "INC2001" not in {r["ref"] for r in fetch_records(scope("acme"))}


def test_scope_filters_narrow_records():
    assert {r["ref"] for r in fetch_records(scope(service="vpn", environment="production", region="india"))} == {"INC1001", "INC1002", "INC1005"}


def test_overview_is_computed_from_data():
    o = build_overview(fetch_records(scope("acme")))
    assert o.total == 7 and o.open == 1 and o.by_severity == {"P1": 1, "P2": 3, "P3": 2, "P4": 1}
    assert [i["ref"] for i in o.open_incidents] == ["INC1005"]


def test_report_mode_and_tool_use_is_recorded():
    seen: list[str] = []
    res = agent(seen=seen).run("overview", scope("acme"))
    assert res.mode == "report" and res.steps == ['incident_counts({"group_by": "severity"})']
    assert '"P2": 3' in seen[0]  # tool output reached the writer


def test_tools_cannot_cross_tenants():
    seen: list[str] = []
    agent(calls=[("list_incidents", {"service": "vpn"})], seen=seen).run("vpn", scope("globex"))
    assert "INC2001" in seen[0] and "INC1001" not in seen[0]


def test_invented_incident_id_is_rejected():
    res = agent(out=report("See INC9999.", refs=())).run("overview", scope("acme"))
    assert res.mode == "fallback" and res.fallback_reason == "unknown_incident"
    assert res.overview.total == 7  # the deterministic numbers survive


def test_other_tenants_incident_id_is_rejected():
    res = agent(out=report("Same as INC2001.", refs=("INC2001",))).run("overview", scope("acme"))
    assert res.fallback_reason == "unknown_incident"


def test_bad_tool_arguments_do_not_crash():
    res = agent(calls=[("incident_counts", {"group_by": "nonsense"}), ("nope", {})]).run("overview", scope("acme"))
    assert res.mode == "report"


def test_llm_failure_falls_back_with_overview():
    def boom(_):
        raise TimeoutError("slow")

    a = AnalyticsAgent(tool_llm=ScriptedToolLLM([]), report_llm=RunnableLambda(boom), search_factory=lambda s: lambda q, k: [])
    res = a.run("overview", scope("acme"))
    assert res.mode == "fallback" and res.fallback_reason == "llm_error:TimeoutError" and res.overview.total == 7


def test_empty_scope_skips_the_llm():
    res = agent().run("overview", scope("nobody"))
    assert res.mode == "no_data" and res.report is None


def test_gathering_switches_to_the_fallback_model():
    class Busy:
        def bind_tools(self, tools):
            def boom(_):
                raise RuntimeError("503 UNAVAILABLE")

            return RunnableLambda(boom)

    a = AnalyticsAgent(tool_llm=[Busy(), ScriptedToolLLM([("incident_counts", {"group_by": "status"})])],
                       report_llm=RunnableLambda(lambda _: report()), search_factory=lambda s: lambda q, k: [])
    res = a.run("overview", scope("acme"))
    assert res.mode == "report" and res.steps == ['incident_counts({"group_by": "status"})']
