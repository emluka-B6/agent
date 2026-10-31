"""Tests for artifact naming, persistence, and plan-level section cleanup."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any

import pytest

from issue_planner.config import DEFAULT_OUTPUT_DIR
from issue_planner.models import ImplementationPlan, SubTask, WorkflowState
from issue_planner.output import (
    build_document,
    dedupe_sections,
    render_markdown,
    unique_output_path,
    write_output,
)


def make_task() -> SubTask:
    return SubTask(
        id="T1",
        title="Task T1",
        description="Do the first thing",
        implementation_prompt="Do the first thing with these constraints.",
        acceptance_criteria=["The first thing is done"],
        verification="pytest",
    )


def make_plan(**overrides: Any) -> ImplementationPlan:
    data: dict[str, Any] = {
        "summary": "A plan",
        "assumptions": [],
        "subtasks": [make_task()],
        "open_questions": [],
        "definition_of_done": ["Everything works"],
    }
    data.update(overrides)
    return ImplementationPlan(**data)


def plan_state(plan: ImplementationPlan) -> WorkflowState:
    return {
        "issue": "Add rate limiting to the public API",
        "research": None,
        "sources": [],
        "research_error": None,
        "plan": plan,
        "critique": None,
        "evaluations": 0,
        "approved": False,
    }


def plan_document(plan: ImplementationPlan) -> dict[str, Any]:
    return build_document(
        plan_state(plan),
        provider="openai",
        model_name="gpt-5-mini",
        max_iterations=2,
    )


def make_document(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "issue": "Add rate limiting to the public API",
        "generated_at": "2026-10-18T09:31:00+00:00",
        "status": "approved",
        "subtasks": [],
    }
    document.update(overrides)
    return document


def test_json_output_name_is_hashed(tmp_path: Path) -> None:
    path, output_format = unique_output_path(make_document(), tmp_path, "json")

    assert output_format == "json"
    assert path.parent == tmp_path
    assert re.fullmatch(r"output-[0-9a-f]{12}\.json", path.name)


def test_markdown_output_name_is_hashed(tmp_path: Path) -> None:
    path, output_format = unique_output_path(
        make_document(), tmp_path, "markdown"
    )

    assert output_format == "markdown"
    assert re.fullmatch(r"output-[0-9a-f]{12}\.md", path.name)


def test_auto_format_falls_back_to_json(tmp_path: Path) -> None:
    path, output_format = unique_output_path(make_document(), tmp_path, "auto")

    assert (output_format, path.suffix) == ("json", ".json")


def test_default_output_directory_is_shared_with_the_viewer() -> None:
    path, _ = unique_output_path(make_document())

    assert path.parent == DEFAULT_OUTPUT_DIR


def test_cli_defaults_to_the_shared_output_directory() -> None:
    from agent import build_parser

    args = build_parser().parse_args(["Add rate limiting"])

    assert args.output_dir == DEFAULT_OUTPUT_DIR


def test_repeated_runs_do_not_overwrite_previous_artifacts(
    tmp_path: Path,
) -> None:
    first, first_format = unique_output_path(make_document(), tmp_path, "json")
    write_output(make_document(), first, first_format)
    first_content = first.read_text(encoding="utf-8")

    second, second_format = unique_output_path(
        make_document(), tmp_path, "json"
    )
    write_output(
        make_document(status="needs_improvement"), second, second_format
    )

    assert first != second
    assert first.read_text(encoding="utf-8") == first_content
    assert json.loads(second.read_text(encoding="utf-8"))["status"] == (
        "needs_improvement"
    )


def test_resamples_when_the_hash_is_already_taken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = make_document()
    fixed = uuid.UUID(int=0)
    monkeypatch.setattr("issue_planner.output.uuid.uuid4", lambda: fixed)

    seed = "\x00".join(
        (document["issue"], document["generated_at"], fixed.hex, "0")
    )
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
    taken = tmp_path / f"output-{digest}.json"
    taken.write_text("{}", encoding="utf-8")

    path, _ = unique_output_path(document, tmp_path, "json")

    assert path != taken
    assert not path.exists()


def test_dedupe_sections_lets_the_definition_of_done_win() -> None:
    document: dict[str, Any] = {
        "definition_of_done": ["Same item"],
        "assumptions": ["Same item"],
        "open_questions": ["Same item"],
    }

    assert dedupe_sections(document) == {
        "definition_of_done": ["Same item"],
        "assumptions": [],
        "open_questions": [],
    }


def test_repeated_plan_items_end_up_in_one_section() -> None:
    plan = make_plan(
        assumptions=["Python 3.10 is available."],
        open_questions=["Python 3.10 is available."],
        definition_of_done=["Tests pass", "Python 3.10 is available."],
    )

    document = plan_document(plan)

    assert document["definition_of_done"] == [
        "Tests pass",
        "Python 3.10 is available.",
    ]
    assert document["assumptions"] == []
    assert document["open_questions"] == []


def test_assumptions_win_over_open_questions() -> None:
    plan = make_plan(
        assumptions=["Single user only."],
        open_questions=["single user only"],
    )

    document = plan_document(plan)

    assert document["assumptions"] == ["Single user only."]
    assert document["open_questions"] == []


def test_section_dedupe_ignores_case_spacing_and_punctuation() -> None:
    plan = make_plan(
        assumptions=["Use SQLite."],
        definition_of_done=["use   sqlite", "Backups are documented"],
    )

    document = plan_document(plan)

    assert document["assumptions"] == []
    assert document["definition_of_done"] == [
        "use   sqlite",
        "Backups are documented",
    ]


def test_repeated_and_blank_items_are_removed_within_a_section() -> None:
    plan = make_plan(
        assumptions=["", "   "],
        definition_of_done=["Tests pass", "  tests   pass ", "Docs updated", "  "],
    )

    document = plan_document(plan)

    assert document["assumptions"] == []
    assert document["definition_of_done"] == ["Tests pass", "Docs updated"]


def test_definition_of_done_renders_as_a_markdown_bullet_list() -> None:
    plan = make_plan(definition_of_done=["Tests pass", "Docs updated"])

    rendered = render_markdown(plan_document(plan))

    assert "## Definition of done\n\n- Tests pass\n- Docs updated" in rendered


def test_empty_definition_of_done_renders_none_in_markdown() -> None:
    document = plan_document(make_plan())
    document["definition_of_done"] = []

    assert "## Definition of done\n\n- none" in render_markdown(document)
