"""Tests for planner-triggered follow-up research and its budget."""

from __future__ import annotations

from typing import Any, cast

import pytest

from issue_planner import graph as graph_module
from issue_planner import nodes as nodes_module
from issue_planner.graph import run_agent
from issue_planner.models import (
    Critique,
    ImplementationPlan,
    ResearchSummary,
    Source,
    SubTask,
    WorkflowState,
)
from issue_planner.nodes import planner_node, research_node, route_after_planner
from issue_planner.output import build_document


def make_plan(research_requests: list[str] | None = None) -> ImplementationPlan:
    return ImplementationPlan(
        summary="Harden the public API",
        subtasks=[
            SubTask(
                id="T1",
                title="Add a token bucket",
                description="Introduce a token bucket limiter.",
                implementation_prompt="Add a token bucket limiter with tests.",
                acceptance_criteria=["Requests above the limit are rejected"],
                verification="pytest tests/test_ratelimit.py",
            )
        ],
        definition_of_done=["Rate limiting is enabled"],
        research_requests=list(research_requests or []),
    )


def make_state(**overrides: Any) -> WorkflowState:
    state: dict[str, Any] = {
        "issue": "Add rate limiting to the public API",
        "domain": "software",
        "research": None,
        "sources": [],
        "research_error": None,
        "research_calls": 0,
        "research_requests": [],
        "plan": None,
        "critique": None,
        "evaluations": 0,
        "approved": False,
    }
    state.update(overrides)
    return cast(WorkflowState, state)


def test_route_after_planner_goes_to_research_when_asked_and_budget_remains() -> None:
    state = make_state(
        research_calls=1,
        research_requests=["What is the current rate-limit guidance?"],
    )

    assert route_after_planner(state, max_research_calls=2) == "research"


def test_route_after_planner_goes_to_critic_when_the_budget_is_exhausted() -> None:
    state = make_state(
        research_calls=2,
        research_requests=["What is the current rate-limit guidance?"],
    )

    assert route_after_planner(state, max_research_calls=2) == "critic"


def test_route_after_planner_goes_to_critic_without_requests() -> None:
    assert route_after_planner(make_state(), max_research_calls=2) == "critic"


def test_route_after_planner_skips_research_when_it_is_unavailable() -> None:
    state = make_state(
        research_calls=1,
        research_requests=["What is the current rate-limit guidance?"],
    )

    assert (
        route_after_planner(
            state, max_research_calls=2, research_available=False
        )
        == "critic"
    )


def test_planner_node_surfaces_deduplicated_research_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_invoke_structured(
        _model: object,
        _schema: object,
        _provider: str,
        _messages: object,
        _label: str,
        _max_retries: int,
    ) -> ImplementationPlan:
        return make_plan(
            research_requests=[
                " current guidance ",
                "current guidance",
                "known limits",
                "third query",
                "ignored beyond the limit",
            ]
        )

    monkeypatch.setattr(nodes_module, "invoke_structured", fake_invoke_structured)

    result = planner_node(
        make_state(),
        model=object(),  # type: ignore[arg-type]
        provider="openai",
        max_retries=1,
    )

    assert result["research_requests"] == [
        "current guidance",
        "known limits",
        "third query",
    ]


def install_fake_nodes(
    monkeypatch: pytest.MonkeyPatch, request_always: bool = False
) -> tuple[list[list[str]], list[int]]:
    """Replace the LLM nodes and record how often each one ran."""
    research_calls: list[list[str]] = []
    planner_calls: list[int] = []

    def fake_research(state: WorkflowState, **_kwargs: Any) -> dict[str, Any]:
        research_calls.append(list(state.get("research_requests") or []))
        return {
            "research": ResearchSummary(recommended_approach="Token buckets"),
            "sources": [],
            "research_error": None,
            "research_calls": state.get("research_calls", 0) + 1,
            "research_requests": [],
        }

    def fake_planner(state: WorkflowState, **_kwargs: Any) -> dict[str, Any]:
        planner_calls.append(state.get("research_calls", 0))
        wants_research = request_always or len(planner_calls) == 1
        requests = (
            ["What is the current rate-limit guidance?"] if wants_research else []
        )
        return {
            "plan": make_plan(research_requests=requests),
            "research_requests": requests,
            "approved": False,
        }

    def fake_critic(_state: WorkflowState, **_kwargs: Any) -> dict[str, Any]:
        return {
            "critique": Critique(approved=True, summary="Ready"),
            "evaluations": 1,
            "approved": True,
        }

    monkeypatch.setattr(graph_module, "research_node", fake_research)
    monkeypatch.setattr(graph_module, "planner_node", fake_planner)
    monkeypatch.setattr(graph_module, "critic_node", fake_critic)
    return research_calls, planner_calls


def test_workflow_runs_one_follow_up_research_and_then_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research_calls, planner_calls = install_fake_nodes(monkeypatch)

    final_state = run_agent(
        "Add rate limiting",
        model=object(),  # type: ignore[arg-type]
        search_client=object(),
        max_research_calls=2,
    )

    assert research_calls == [[], ["What is the current rate-limit guidance?"]]
    assert planner_calls == [1, 2]
    assert final_state["research_calls"] == 2
    assert final_state["approved"] is True


def test_workflow_does_not_run_follow_up_research_when_the_budget_is_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research_calls, planner_calls = install_fake_nodes(
        monkeypatch, request_always=True
    )

    run_agent(
        "Add rate limiting",
        model=object(),  # type: ignore[arg-type]
        search_client=object(),
        max_research_calls=1,
    )

    assert research_calls == [[]]
    assert planner_calls == [1]


def test_workflow_does_not_route_to_research_when_it_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    research_calls, planner_calls = install_fake_nodes(
        monkeypatch, request_always=True
    )

    run_agent(
        "Add rate limiting",
        model=object(),  # type: ignore[arg-type]
        search_client=object(),
        skip_research=True,
        max_research_calls=2,
    )

    assert research_calls == [[]]
    assert planner_calls == [1]


class FakeSearch:
    """Minimal search runnable: it records queries and returns one result each."""

    def __init__(self, fail: bool = False) -> None:
        self.queries: list[str] = []
        self.fail = fail

    def invoke(self, query: str) -> dict[str, Any]:
        self.queries.append(query)
        if self.fail:
            raise RuntimeError("search unavailable")
        return {
            "results": [
                {
                    "title": query,
                    "url": f"https://example.com/{len(self.queries)}",
                    "content": f"Result for {query}",
                }
            ]
        }


def capture_research_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_invoke_structured(
        _model: object,
        _schema: object,
        _provider: str,
        messages: object,
        _label: str,
        _max_retries: int,
    ) -> ResearchSummary:
        captured["messages"] = messages
        return ResearchSummary(recommended_approach="Merged research")

    monkeypatch.setattr(nodes_module, "invoke_structured", fake_invoke_structured)
    return captured


def test_research_node_searches_the_issue_on_the_first_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = capture_research_messages(monkeypatch)
    search = FakeSearch()

    result = research_node(
        make_state(),
        model=object(),  # type: ignore[arg-type]
        provider="openai",
        search_client=search,
        max_search_results=5,
        skip_research=False,
        max_retries=1,
    )

    assert search.queries == ["Add rate limiting to the public API"]
    assert result["research_calls"] == 1
    assert [source.url for source in result["sources"]] == [
        "https://example.com/1"
    ]
    rendered = "\n".join(
        str(message.content) for message in captured["messages"]
    )
    assert "(none; initial research)" in rendered


def test_research_node_merges_follow_up_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = capture_research_messages(monkeypatch)
    previous = ResearchSummary(recommended_approach="Use token buckets")
    existing = Source(
        title="Existing",
        url="https://example.com/existing",
        snippet="Earlier evidence",
    )
    search = FakeSearch()

    result = research_node(
        make_state(
            research=previous,
            sources=[existing],
            research_calls=1,
            research_requests=["current guidance", "known limits"],
        ),
        model=object(),  # type: ignore[arg-type]
        provider="openai",
        search_client=search,
        max_search_results=5,
        skip_research=False,
        max_retries=1,
    )

    assert search.queries == ["current guidance", "known limits"]
    assert result["research_calls"] == 2
    assert result["research_requests"] == []
    assert [source.url for source in result["sources"]] == [
        "https://example.com/existing",
        "https://example.com/1",
        "https://example.com/2",
    ]
    rendered = "\n".join(
        str(message.content) for message in captured["messages"]
    )
    assert "current guidance" in rendered
    assert "known limits" in rendered
    assert "Use token buckets" in rendered


def test_research_node_keeps_prior_evidence_when_follow_up_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = capture_research_messages(monkeypatch)
    previous = ResearchSummary(recommended_approach="Use token buckets")
    existing = Source(
        title="Existing",
        url="https://example.com/existing",
        snippet="Earlier evidence",
    )

    result = research_node(
        make_state(
            research=previous,
            sources=[existing],
            research_calls=1,
            research_requests=["current guidance"],
        ),
        model=object(),  # type: ignore[arg-type]
        provider="openai",
        search_client=FakeSearch(fail=True),
        max_search_results=5,
        skip_research=False,
        max_retries=1,
    )

    assert result["sources"] == [existing]
    assert result["research_calls"] == 2
    assert "Web search failed" in str(result["research_error"])
    rendered = "\n".join(
        str(message.content) for message in captured["messages"]
    )
    assert "Use token buckets" in rendered


def test_artifact_records_the_research_budget_without_control_fields() -> None:
    plan = make_plan(research_requests=["What is the current guidance?"])
    state = make_state(
        plan=plan,
        research_calls=2,
        research_requests=["What is the current guidance?"],
    )

    document = build_document(
        state,
        provider="openai",
        model_name="gpt-5-mini",
        max_iterations=2,
        max_research_calls=2,
    )

    assert document["metadata"]["research_calls"] == 2
    assert document["metadata"]["max_research_calls"] == 2
    assert "research_requests" not in document
    assert "research_requests" not in document["subtasks"][0]
