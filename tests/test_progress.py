"""Tests for live progress reporting and partial artifacts."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, cast

import pytest

from issue_planner import graph as graph_module
from issue_planner.graph import run_agent
from issue_planner.models import (
    Critique,
    ImplementationPlan,
    ResearchSummary,
    SubTask,
    WorkflowState,
)
from issue_planner.output import build_progress_document, render_markdown
from issue_planner.progress import ProgressReporter

GENERATED_AT = "2026-10-18T10:00:00+00:00"


def make_plan() -> ImplementationPlan:
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
            ),
            SubTask(
                id="T2",
                title="Document the limits",
                description="Document the new rate limits.",
                implementation_prompt="Document the rate limits in the README.",
                depends_on=["T1"],
                acceptance_criteria=["The README documents every limit"],
                verification="Read the README",
            ),
        ],
        definition_of_done=["Rate limiting is enabled and documented"],
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


def test_progress_document_reports_subtask_count_and_titles() -> None:
    state = make_state(plan=make_plan(), evaluations=1)

    document = build_progress_document(
        state,
        provider="openai",
        model_name="gpt-5-mini",
        max_iterations=2,
        stage="planner_done",
        completed_stages=["research", "planner"],
        generated_at=GENERATED_AT,
    )

    progress = document["metadata"]["progress"]
    assert document["status"] == "running"
    assert document["generated_at"] == GENERATED_AT
    assert progress["subtask_count"] == 2
    assert progress["subtask_titles"] == [
        "Add a token bucket",
        "Document the limits",
    ]
    assert progress["stage"] == "planner_done"
    assert progress["completed_stages"] == ["research", "planner"]
    assert [task["id"] for task in document["subtasks"]] == ["T1", "T2"]

    rendered = render_markdown(document)
    assert "**Status:** running" in rendered
    assert "- Subtasks so far: 2" in rendered
    assert "Add a token bucket" in rendered


def test_progress_document_works_before_the_first_plan() -> None:
    document = build_progress_document(
        make_state(),
        provider="openai",
        model_name="gpt-5-mini",
        max_iterations=2,
        stage="research",
        running_stage="research",
        generated_at=GENERATED_AT,
    )

    assert document["status"] == "running"
    assert document["subtasks"] == []
    assert document["summary"] == ""
    assert document["metadata"]["progress"]["subtask_count"] == 0
    assert document["metadata"]["progress"]["subtask_titles"] == []
    assert document["metadata"]["progress"]["running_stage"] == "research"

    rendered = render_markdown(document)
    assert "**Status:** running" in rendered
    assert "- Subtasks so far: 0" in rendered


def test_reporter_prints_progress_and_updates_the_partial_artifact(
    tmp_path: Path,
) -> None:
    stream = io.StringIO()
    output_path = tmp_path / "output-test.json"
    reporter = ProgressReporter(
        output_path=output_path,
        max_iterations=2,
        stream=stream,
        generated_at=GENERATED_AT,
    )

    reporter.workflow_started("Add rate limiting to the public API")
    initial = json.loads(output_path.read_text(encoding="utf-8"))
    assert initial["metadata"]["progress"]["stage"] == "starting"

    state = make_state(plan=make_plan())
    reporter.node_started("planner", state)
    reporter.node_finished("planner", state)

    document = json.loads(output_path.read_text(encoding="utf-8"))
    assert document["metadata"]["progress"]["subtask_count"] == 2
    assert document["metadata"]["progress"]["completed_stages"] == ["planner"]
    assert document["metadata"]["progress"]["stage"] == "planner_done"
    assert document["generated_at"] == GENERATED_AT

    log = stream.getvalue()
    assert "[agent] planning started" in log
    assert "[2/3] planner sub-agent: running..." in log
    assert "[2/3] planner sub-agent: done" in log
    assert "subtasks=2" in log
    assert "    - T1: Add a token bucket" in log
    assert "    - T2: Document the limits" in log
    assert "partial result file" in log
    assert reporter.partial_written is True


def test_reporter_announces_a_revision_after_a_rejected_review(
    tmp_path: Path,
) -> None:
    stream = io.StringIO()
    reporter = ProgressReporter(
        output_path=tmp_path / "output-test.json",
        max_iterations=2,
        stream=stream,
    )
    reporter.workflow_started("Add rate limiting")
    state = make_state(
        plan=make_plan(),
        critique=Critique(
            approved=False,
            summary="Needs work",
            blocking_issues=["Missing rollback plan"],
        ),
        evaluations=1,
        approved=False,
    )

    reporter.node_finished("critic", state)

    log = stream.getvalue()
    assert "approved=no | blocking issues=1" in log
    assert "next sub-agent: planner revision (review 2/2)" in log


def test_reporter_announces_follow_up_research_after_the_planner(
    tmp_path: Path,
) -> None:
    stream = io.StringIO()
    reporter = ProgressReporter(
        output_path=tmp_path / "output-test.json",
        max_iterations=2,
        max_research_calls=2,
        stream=stream,
    )
    reporter.workflow_started("Add rate limiting")
    state = make_state(
        plan=make_plan(),
        research_calls=1,
        research_requests=["Current rate-limit guidance"],
    )

    reporter.node_started("research", state)
    reporter.node_finished("planner", state)

    log = stream.getvalue()
    assert "follow-up research (2/2) | questions=1" in log
    assert (
        "next sub-agent: research follow-up (research 2/2) | questions=1" in log
    )


def test_reporter_stops_requesting_research_when_the_budget_is_exhausted(
    tmp_path: Path,
) -> None:
    stream = io.StringIO()
    reporter = ProgressReporter(
        output_path=tmp_path / "output-test.json",
        max_research_calls=2,
        stream=stream,
    )
    reporter.workflow_started("Add rate limiting")
    state = make_state(
        plan=make_plan(),
        research_calls=2,
        research_requests=["Current rate-limit guidance"],
    )

    reporter.node_finished("planner", state)

    log = stream.getvalue()
    assert "research budget exhausted (2/2); continuing to critic" in log


def test_reporter_does_not_announce_unavailable_follow_up_research(
    tmp_path: Path,
) -> None:
    stream = io.StringIO()
    reporter = ProgressReporter(
        output_path=tmp_path / "output-test.json",
        max_research_calls=2,
        research_available=False,
        stream=stream,
    )
    reporter.workflow_started("Add rate limiting")
    state = make_state(
        plan=make_plan(),
        research_calls=1,
        research_requests=["Current rate-limit guidance"],
    )

    reporter.node_finished("planner", state)

    log = stream.getvalue()
    assert "next sub-agent: critic (follow-up research unavailable)" in log
    assert "research follow-up (research" not in log


def test_reporter_keeps_the_partial_artifact_when_a_node_fails(
    tmp_path: Path,
) -> None:
    stream = io.StringIO()
    output_path = tmp_path / "output-test.json"
    reporter = ProgressReporter(
        output_path=output_path, stream=stream, generated_at=GENERATED_AT
    )
    reporter.workflow_started("Add rate limiting")
    reporter.node_started("research", make_state())

    error = RuntimeError("provider unavailable")
    reporter.node_failed("research", make_state(), error)
    reporter.workflow_failed(error)

    log = stream.getvalue()
    assert "research sub-agent: failed after" in log
    assert "provider unavailable" in log
    assert f"partial result kept in {output_path}" in log
    document = json.loads(output_path.read_text(encoding="utf-8"))
    assert document["metadata"]["progress"]["stage"] == "research_failed"


def test_reporter_failure_does_not_abort_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stream = io.StringIO()
    reporter = ProgressReporter(
        output_path=tmp_path / "output-test.json", stream=stream
    )

    def broken_write(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("issue_planner.progress.write_output", broken_write)

    reporter.workflow_started("Add rate limiting")

    assert "could not write the partial result" in stream.getvalue()
    assert reporter.partial_written is False


def test_run_agent_reports_every_sub_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = make_plan()

    def fake_research(state: WorkflowState, **kwargs: Any) -> dict[str, Any]:
        return {
            "research": ResearchSummary(recommended_approach="Token buckets"),
            "sources": [],
            "research_error": None,
        }

    def fake_planner(state: WorkflowState, **kwargs: Any) -> dict[str, Any]:
        return {"plan": plan, "approved": False}

    def fake_critic(state: WorkflowState, **kwargs: Any) -> dict[str, Any]:
        return {
            "critique": Critique(approved=True, summary="Ready"),
            "evaluations": 1,
            "approved": True,
        }

    monkeypatch.setattr(graph_module, "research_node", fake_research)
    monkeypatch.setattr(graph_module, "planner_node", fake_planner)
    monkeypatch.setattr(graph_module, "critic_node", fake_critic)

    stream = io.StringIO()
    output_path = tmp_path / "output-test.json"
    reporter = ProgressReporter(
        output_path=output_path, stream=stream, generated_at=GENERATED_AT
    )

    final_state = run_agent(
        "Add rate limiting",
        model=object(),  # type: ignore[arg-type]
        reporter=reporter,
    )

    assert final_state["approved"] is True
    log = stream.getvalue()
    for label in ("research sub-agent", "planner sub-agent", "critic sub-agent"):
        assert f"{label}: running..." in log
        assert f"{label}: done" in log
    assert "planning finished" in log
    assert "subtasks=2" in log

    document = json.loads(output_path.read_text(encoding="utf-8"))
    progress = document["metadata"]["progress"]
    assert progress["completed_stages"] == ["research", "planner", "critic"]
    assert progress["subtask_titles"] == [
        "Add a token bucket",
        "Document the limits",
    ]


def test_run_agent_reports_a_failure_and_keeps_the_partial_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_research(state: WorkflowState, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("search exploded")

    monkeypatch.setattr(graph_module, "research_node", broken_research)

    stream = io.StringIO()
    output_path = tmp_path / "output-test.json"
    reporter = ProgressReporter(
        output_path=output_path, stream=stream, generated_at=GENERATED_AT
    )

    with pytest.raises(RuntimeError, match="search exploded"):
        run_agent(
            "Add rate limiting",
            model=object(),  # type: ignore[arg-type]
            reporter=reporter,
        )

    log = stream.getvalue()
    assert "research sub-agent: failed after" in log
    assert "planning failed" in log
    assert f"partial result kept in {output_path}" in log
    document = json.loads(output_path.read_text(encoding="utf-8"))
    assert document["status"] == "running"
    assert document["metadata"]["progress"]["stage"] == "research_failed"



def test_cli_writes_progress_then_the_final_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent import main

    final_state = make_state(
        plan=make_plan(),
        critique=Critique(approved=True, summary="Ready"),
        evaluations=1,
        approved=True,
    )

    def fake_run_agent(**kwargs: Any) -> WorkflowState:
        reporter: ProgressReporter = kwargs["reporter"]
        reporter.workflow_started(kwargs["issue"])
        reporter.node_started("planner", make_state())
        reporter.node_finished("planner", final_state)
        reporter.workflow_finished(final_state)
        return final_state

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("agent.build_model", lambda provider, model: object())
    monkeypatch.setattr("agent.run_agent", fake_run_agent)

    exit_code = main(
        [
            "Add rate limiting",
            "--skip-research",
            "--output-dir",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    output_path = Path(captured.out.strip().splitlines()[-1])
    assert output_path.parent == tmp_path

    document = json.loads(output_path.read_text(encoding="utf-8"))
    assert document["status"] == "approved"
    assert document["approved"] is True
    assert document["metadata"].get("progress") is None
    assert document["subtasks"][0]["title"] == "Add a token bucket"

    assert "planning started" in captured.err
    assert "partial result file" in captured.err
    assert "subtasks=2" in captured.err
    assert "Plan written to" in captured.err

