"""Tests for continuing an unfinished run from its artifact."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import pytest

from issue_planner import graph as graph_module
from issue_planner.budget import RunBudget
from issue_planner.graph import run_agent
from issue_planner.models import Critique, WorkflowState
from issue_planner.nodes import route_start
from issue_planner.output import build_document, document_status
from issue_planner.progress import ProgressReporter
from issue_planner.resume import (
    latest_artifact,
    load_resume_state,
    resolve_resume_path,
)

GENERATED_AT = "2026-10-25T12:22:40+00:00"


def artifact_document(**overrides: Any) -> dict[str, Any]:
    """Return an artifact shaped like ``build_document`` output."""
    document: dict[str, Any] = {
        "issue": "Add rate limiting to the public API",
        "domain": "software",
        "generated_at": GENERATED_AT,
        "status": "max_iterations_reached",
        "approved": False,
        "summary": "Harden the public API",
        "subtasks": [
            {
                "id": "T1",
                "title": "Add a token bucket",
                "description": "Introduce a token bucket limiter.",
                "implementation_prompt": "Add a token bucket limiter with tests.",
                "depends_on": [],
                "files": ["src/ratelimit.py"],
                "acceptance_criteria": ["Requests above the limit are rejected"],
                "verification": "pytest tests/test_ratelimit.py",
                "complexity": "small",
                "risks": [],
            },
            {
                "id": "T2",
                "title": "Wire the limiter into the API",
                "description": "Apply the limiter to the public routes.",
                "implementation_prompt": "Wire the limiter into the API middleware.",
                "depends_on": ["T1"],
                "files": [],
                "acceptance_criteria": ["Limited routes return 429"],
                "verification": "pytest tests/test_api.py",
                "complexity": "medium",
                "risks": [],
            },
        ],
        "assumptions": ["Python 3.10"],
        "open_questions": [],
        "definition_of_done": ["Rate limiting is enabled"],
        "review": {
            "approved": False,
            "summary": "Close, but the burst case is missing",
            "blocking_issues": ["T1 must cover burst traffic"],
            "suggestions": ["Document the limits"],
        },
        "research": {
            "summary": {
                "key_facts": ["Token buckets are a common limiter"],
                "constraints": [],
                "recommended_approach": "Use token buckets",
                "assumptions": [],
            },
            "sources": [
                {
                    "title": "Tavily docs",
                    "url": "https://example.com/tavily",
                    "snippet": "Search API reference",
                }
            ],
            "error": None,
        },
        "metadata": {
            "provider": "openai",
            "model": "gpt-5-mini",
            "evaluations": 2,
            "max_iterations": 2,
            "research_calls": 1,
            "max_research_calls": 2,
        },
    }
    document.update(overrides)
    return document


def write_artifact(
    directory: Path, name: str = "output-abc123def456.json", **overrides: Any
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(artifact_document(**overrides)), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Loading a previous artifact
# ---------------------------------------------------------------------------
def test_load_resume_state_restores_the_plan_review_and_counters(
    tmp_path: Path,
) -> None:
    path = write_artifact(tmp_path)

    resume = load_resume_state(path)

    assert resume.issue == "Add rate limiting to the public API"
    assert resume.domain == "software"
    assert [task.id for task in resume.plan.subtasks] == ["T1", "T2"]
    assert resume.plan.subtasks[1].depends_on == ["T1"]
    assert resume.plan.assumptions == ["Python 3.10"]
    assert resume.critique is not None
    assert resume.critique.blocking_issues == ["T1 must cover burst traffic"]
    assert resume.research is not None
    assert resume.research.recommended_approach == "Use token buckets"
    assert [source.url for source in resume.sources] == [
        "https://example.com/tavily"
    ]
    assert resume.evaluations == 2
    assert resume.max_iterations == 2
    assert resume.research_calls == 1
    assert resume.max_research_calls == 2
    assert resume.runs == 1
    assert resume.status == "max_iterations_reached"
    assert resume.path == path.resolve()


def test_resumed_state_keeps_the_starting_point_of_the_workflow(
    tmp_path: Path,
) -> None:
    resume = load_resume_state(write_artifact(tmp_path))

    state = resume.as_initial_state()

    assert state["issue"] == resume.issue
    assert state["plan"] is resume.plan
    assert state["critique"] is resume.critique
    assert state["evaluations"] == 2
    assert state["research_calls"] == 1
    assert state["research_requests"] == []
    # The critic has to approve the revised plan again.
    assert state["approved"] is False


def test_load_resume_state_counts_a_chained_artifact(tmp_path: Path) -> None:
    document = artifact_document()
    document["metadata"].update({"runs": 3})
    path = tmp_path / "output-chained.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    assert load_resume_state(path).runs == 3


def test_load_resume_state_rejects_a_markdown_report(tmp_path: Path) -> None:
    path = tmp_path / "output-abc123def456.md"
    path.write_text("# Implementation plan\n", encoding="utf-8")

    with pytest.raises(ValueError, match="only JSON artifacts can be resumed"):
        load_resume_state(path)


def test_load_resume_state_rejects_a_run_that_is_still_running(
    tmp_path: Path,
) -> None:
    path = write_artifact(tmp_path, status="running")

    with pytest.raises(ValueError, match="still marked as 'running'"):
        load_resume_state(path)


def test_load_resume_state_rejects_an_artifact_without_a_plan(
    tmp_path: Path,
) -> None:
    path = write_artifact(tmp_path, subtasks=[])

    with pytest.raises(ValueError, match="no subtasks"):
        load_resume_state(path)


def test_load_resume_state_reports_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "output-broken.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="not a valid JSON artifact"):
        load_resume_state(path)


def test_load_resume_state_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Resume artifact not found"):
        load_resume_state(tmp_path / "output-missing.json")


# ---------------------------------------------------------------------------
# Picking the artifact
# ---------------------------------------------------------------------------
def make_two_artifacts(directory: Path) -> tuple[Path, Path]:
    older = write_artifact(directory, name="output-aaaaaaaaaaaa.json")
    newer = write_artifact(
        directory, name="output-bbbbbbbbbbbb.json", status="needs_improvement"
    )
    os.utime(older, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))
    return older, newer


def test_resolve_resume_path_accepts_a_file(tmp_path: Path) -> None:
    older, newer = make_two_artifacts(tmp_path)

    assert resolve_resume_path(older) == older
    assert resolve_resume_path(str(newer)) == newer


def test_resolve_resume_path_accepts_a_directory_and_latest(tmp_path: Path) -> None:
    _, newer = make_two_artifacts(tmp_path)

    assert resolve_resume_path(tmp_path) == newer
    assert resolve_resume_path("latest", tmp_path) == newer


def test_latest_artifact_ignores_markdown_and_temporary_files(
    tmp_path: Path,
) -> None:
    _, newer = make_two_artifacts(tmp_path)
    report = tmp_path / "output-cccccccccccc.md"
    report.write_text("# Implementation plan\n", encoding="utf-8")
    temporary = tmp_path / "output-dddddddddddd.json.tmp"
    temporary.write_text("{}", encoding="utf-8")
    os.utime(report, (3_000_000, 3_000_000))
    os.utime(temporary, (3_000_000, 3_000_000))

    assert latest_artifact(tmp_path) == newer


def test_resolve_resume_path_reports_a_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Resume artifact not found"):
        resolve_resume_path(tmp_path / "output-missing.json")


def test_latest_artifact_reports_an_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No JSON artifact to resume from"):
        latest_artifact(tmp_path)


def test_resolve_resume_path_rejects_an_empty_value(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="JSON artifact, a directory, or 'latest'"):
        resolve_resume_path("  ", tmp_path)


# ---------------------------------------------------------------------------
# Budget arithmetic
# ---------------------------------------------------------------------------
def test_fresh_budget_describes_a_single_run() -> None:
    budget = RunBudget.fresh(max_iterations=2, max_research_calls=2)

    metadata = budget.to_metadata(evaluations=1, research_calls=2)

    assert budget.continued_run is False
    assert metadata["evaluations"] == 1
    assert metadata["max_iterations"] == 2
    assert metadata["iterations_this_run"] == 1
    assert metadata["max_iterations_this_run"] == 2
    assert metadata["research_calls_this_run"] == 2
    assert metadata["runs"] == 1
    assert metadata["resumed_from"] is None
    assert metadata["previous_status"] is None


def test_continued_budget_adds_a_new_budget_to_the_spent_one(
    tmp_path: Path,
) -> None:
    resume = load_resume_state(write_artifact(tmp_path))

    budget = RunBudget.continued(resume, max_iterations=3, max_research_calls=2)

    # Two reviews and one research call were already spent by the artifact.
    assert budget.max_iterations == 5
    assert budget.max_research_calls == 3
    assert budget.iterations_before == 2
    assert budget.research_calls_before == 1
    assert budget.iterations_available == 3
    assert budget.research_calls_available == 2
    assert budget.continued_run is True
    assert budget.runs == 2
    assert budget.resumed_from == str(resume.path)
    assert budget.previous_status == "max_iterations_reached"

    metadata = budget.to_metadata(evaluations=4, research_calls=2)

    assert metadata["evaluations"] == 4
    assert metadata["max_iterations"] == 5
    assert metadata["iterations_this_run"] == 2
    assert metadata["max_iterations_this_run"] == 3
    assert metadata["research_calls_this_run"] == 1
    assert metadata["max_research_calls_this_run"] == 2
    assert metadata["runs"] == 2


def test_the_default_budget_continues_instead_of_ending_at_the_start(
    tmp_path: Path,
) -> None:
    """``--max-iterations 2`` must add reviews to an artifact that used two."""
    resume = load_resume_state(write_artifact(tmp_path))

    budget = RunBudget.continued(resume, max_iterations=2, max_research_calls=2)

    assert budget.max_iterations > resume.evaluations
    assert budget.iterations_available == 2


# ---------------------------------------------------------------------------
# Workflow continuation
# ---------------------------------------------------------------------------
def test_route_start_skips_research_when_a_plan_is_restored(
    tmp_path: Path,
) -> None:
    resume = load_resume_state(write_artifact(tmp_path))

    assert route_start(resume.as_initial_state()) == "planner"
    assert route_start({"issue": "Add rate limiting"}) == "research"


def test_resumed_run_spends_only_the_new_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resume = load_resume_state(write_artifact(tmp_path))
    budget = RunBudget.continued(resume, max_iterations=2, max_research_calls=1)
    planner_reviews: list[int] = []
    critic_reviews: list[int] = []

    def fake_planner(state: WorkflowState, **_kwargs: Any) -> dict[str, Any]:
        planner_reviews.append(state.get("evaluations", 0))
        return {"plan": resume.plan, "research_requests": [], "approved": False}

    def fake_critic(state: WorkflowState, **_kwargs: Any) -> dict[str, Any]:
        evaluations = state.get("evaluations", 0) + 1
        critic_reviews.append(evaluations)
        return {
            "critique": Critique(
                approved=False,
                summary="Still not ready",
                blocking_issues=["T1 must cover burst traffic"],
            ),
            "evaluations": evaluations,
            "approved": False,
        }

    monkeypatch.setattr(graph_module, "planner_node", fake_planner)
    monkeypatch.setattr(graph_module, "critic_node", fake_critic)

    final_state = run_agent(
        resume.issue,
        model=object(),  # type: ignore[arg-type]
        skip_research=True,
        max_iterations=budget.max_iterations,
        max_research_calls=budget.max_research_calls,
        initial_state=resume.as_initial_state(),
    )

    # The restored review is not repeated: two more reviews run, from 3 to 4.
    assert planner_reviews == [2, 3]
    assert critic_reviews == [3, 4]
    assert final_state["evaluations"] == 4
    assert final_state["approved"] is False
    assert final_state["research_calls"] == 1
    assert document_status(final_state, budget.max_iterations) == (
        "max_iterations_reached"
    )


def test_resumed_run_does_not_repeat_the_initial_research(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resume = load_resume_state(write_artifact(tmp_path))
    research_calls: list[list[str]] = []

    def fake_research(state: WorkflowState, **_kwargs: Any) -> dict[str, Any]:
        research_calls.append(list(state.get("research_requests") or []))
        return {"research_calls": state.get("research_calls", 0) + 1}

    def fake_planner(_state: WorkflowState, **_kwargs: Any) -> dict[str, Any]:
        # Ask for follow-up research so the research node would run if the
        # budget allowed it.
        return {
            "plan": resume.plan,
            "research_requests": ["Current rate-limit guidance"],
            "approved": False,
        }

    def fake_critic(state: WorkflowState, **_kwargs: Any) -> dict[str, Any]:
        return {
            "critique": Critique(approved=True, summary="Ready"),
            "evaluations": state.get("evaluations", 0) + 1,
            "approved": True,
        }

    monkeypatch.setattr(graph_module, "research_node", fake_research)
    monkeypatch.setattr(graph_module, "planner_node", fake_planner)
    monkeypatch.setattr(graph_module, "critic_node", fake_critic)

    final_state = run_agent(
        resume.issue,
        model=object(),  # type: ignore[arg-type]
        search_client=object(),
        max_iterations=3,
        max_research_calls=2,
        initial_state=resume.as_initial_state(),
    )

    # max_research_calls=2 is cumulative and the artifact already spent one
    # call, so exactly one follow-up research call runs.
    assert research_calls == [["Current rate-limit guidance"]]
    assert final_state["research_calls"] == 2
    assert final_state["approved"] is True


# ---------------------------------------------------------------------------
# Artifact metadata and console output
# ---------------------------------------------------------------------------
def test_artifact_records_the_cumulative_totals(tmp_path: Path) -> None:
    resume = load_resume_state(write_artifact(tmp_path))
    budget = RunBudget.continued(resume, max_iterations=2, max_research_calls=1)
    state = resume.as_initial_state()
    state["evaluations"] = budget.max_iterations
    state["critique"] = Critique(
        approved=False, summary="Still not ready", blocking_issues=["Add load tests"]
    )

    document = build_document(
        state,
        provider="openai",
        model_name="gpt-5-mini",
        max_iterations=budget.max_iterations,
        generated_at=GENERATED_AT,
        max_research_calls=budget.max_research_calls,
        budget=budget,
    )

    metadata = document["metadata"]
    assert document["status"] == "max_iterations_reached"
    assert metadata["evaluations"] == 4
    assert metadata["max_iterations"] == 4
    assert metadata["iterations_this_run"] == 2
    assert metadata["max_iterations_this_run"] == 2
    assert metadata["research_calls"] == 1
    assert metadata["max_research_calls"] == 2
    assert metadata["runs"] == 2
    assert str(resume.path) == metadata["resumed_from"]
    assert metadata["previous_status"] == "max_iterations_reached"


def test_reporter_announces_the_continuation_and_the_totals(
    tmp_path: Path,
) -> None:
    resume = load_resume_state(write_artifact(tmp_path))
    budget = RunBudget.continued(resume, max_iterations=2, max_research_calls=1)
    stream = io.StringIO()
    output_path = tmp_path / "output-continued.json"
    reporter = ProgressReporter(
        output_path=output_path,
        max_iterations=budget.max_iterations,
        max_research_calls=budget.max_research_calls,
        budget=budget,
        stream=stream,
        generated_at=GENERATED_AT,
    )

    reporter.workflow_started(resume.issue)
    partial = json.loads(output_path.read_text(encoding="utf-8"))
    assert partial["metadata"]["runs"] == 2
    assert partial["metadata"]["evaluations"] == 2

    final_state = resume.as_initial_state()
    final_state["evaluations"] = 4
    final_state["research_calls"] = 1
    reporter.workflow_finished(final_state)

    log = stream.getvalue()
    assert "max iterations=4" in log
    assert "| run 2" in log
    assert f"continuing {resume.path}" in log
    assert "already used: iterations=2, research calls=1" in log
    assert "this run may add 2 iteration(s) and 1 research call(s)" in log
    assert "iterations=4/4" in log
    assert "research calls=1/2" in log
    assert "runs=2" in log


def test_reporter_omits_the_continuation_banner_for_a_fresh_run(
    tmp_path: Path,
) -> None:
    stream = io.StringIO()
    reporter = ProgressReporter(
        output_path=tmp_path / "output-fresh.json",
        max_iterations=2,
        stream=stream,
        generated_at=GENERATED_AT,
    )

    reporter.workflow_started("Add rate limiting")
    reporter.workflow_finished({"evaluations": 2, "research_calls": 1})

    log = stream.getvalue()
    assert "continuing" not in log
    assert "| run 2" not in log
    assert "iterations=2/2" in log
    assert "runs=1" in log


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_resumes_the_newest_artifact_with_a_larger_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent import main

    artifact = write_artifact(tmp_path)
    captured: dict[str, Any] = {}

    def fake_run_agent(**kwargs: Any) -> WorkflowState:
        captured.update(kwargs)
        state = dict(kwargs["initial_state"] or {})
        state["issue"] = kwargs["issue"]
        state["domain"] = "software"
        state["evaluations"] = kwargs["max_iterations"]
        state["approved"] = False
        return state  # type: ignore[return-value]

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("agent.build_model", lambda provider, model: object())
    monkeypatch.setattr("agent.run_agent", fake_run_agent)

    exit_code = main(
        [
            "--resume",
            str(tmp_path),
            "--max-iterations",
            "2",
            "--max-research-calls",
            "1",
            "--skip-research",
            "--output-dir",
            str(tmp_path),
        ]
    )

    captured_io = capsys.readouterr()
    assert exit_code == 0
    assert captured["issue"] == "Add rate limiting to the public API"
    assert captured["domain"] == "software"
    assert captured["max_iterations"] == 4
    assert captured["max_research_calls"] == 2
    assert captured["initial_state"]["evaluations"] == 2
    assert [task.id for task in captured["initial_state"]["plan"].subtasks] == [
        "T1",
        "T2",
    ]
    assert "info: continuing output-abc123def456.json" in captured_io.err
    assert "this run adds 2 iteration(s) and 1 research call(s)" in captured_io.err
    assert "runs=2" in captured_io.err

    output_path = Path(captured_io.out.strip().splitlines()[-1])
    assert output_path != artifact
    document = json.loads(output_path.read_text(encoding="utf-8"))
    assert document["status"] == "max_iterations_reached"
    assert document["metadata"]["runs"] == 2
    assert document["metadata"]["max_iterations"] == 4
    assert document["metadata"]["resumed_from"] == str(artifact.resolve())


def test_cli_reports_an_artifact_that_cannot_be_resumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from agent import main

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("agent.build_model", lambda provider, model: object())

    exit_code = main(["--resume", str(tmp_path / "output-missing.json")])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "error: Resume artifact not found" in captured.err
