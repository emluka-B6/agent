"""Tests for plan repair and the plan size warnings."""

from __future__ import annotations

import pytest

from issue_planner.config import (
    DEFAULT_MAX_SUBTASKS,
    OVERSIZED_IMPLEMENTATION_PROMPT_CHARS,
)
from issue_planner.models import ImplementationPlan, SubTask
from issue_planner.plan_repair import normalize_plan, plan_warnings


def make_task(
    task_id: str,
    depends_on: tuple[str, ...] = (),
    prompt: str = "Do the work.",
) -> SubTask:
    return SubTask(
        id=task_id,
        title=f"Task {task_id}",
        description=f"Description for {task_id}",
        implementation_prompt=prompt,
        depends_on=list(depends_on),
        acceptance_criteria=[f"{task_id} works"],
        verification="pytest",
    )


def make_plan(*tasks: SubTask) -> ImplementationPlan:
    return ImplementationPlan(
        summary="A plan",
        subtasks=list(tasks),
        definition_of_done=["Everything works"],
    )


def test_normalize_plan_warns_about_unknown_dependencies(
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan = make_plan(make_task("T1"), make_task("T2", ("T9",)))

    normalized = normalize_plan(plan)

    assert normalized.subtasks[1].depends_on == []
    assert "unknown or self dependency T9" in capsys.readouterr().err


def test_plan_warnings_report_a_plan_that_is_too_large() -> None:
    tasks = [
        make_task(f"T{index}") for index in range(1, DEFAULT_MAX_SUBTASKS + 2)
    ]

    warnings = plan_warnings(make_plan(*tasks))

    assert len(warnings) == 1
    assert f"{DEFAULT_MAX_SUBTASKS + 1} subtasks" in warnings[0]
    assert f"maximum is {DEFAULT_MAX_SUBTASKS}" in warnings[0]


def test_plan_warnings_report_oversized_prompts() -> None:
    oversized = "x" * (OVERSIZED_IMPLEMENTATION_PROMPT_CHARS + 1)

    warnings = plan_warnings(make_plan(make_task("T1", prompt=oversized)))

    assert len(warnings) == 1
    assert "'T1'" in warnings[0]
    assert "implementation_prompt" in warnings[0]


def test_plan_warnings_accept_a_plan_within_the_limits() -> None:
    assert plan_warnings(make_plan(make_task("T1"))) == []
