"""Tests for the UI-agnostic plan editing helpers."""

from __future__ import annotations

import pytest

from issue_planner.models import ImplementationPlan, SubTask
from issue_planner.plan_editor import (
    RemovalError,
    apply_plan_to_document,
    edit_subtask,
    find_dependents,
    plan_from_document,
    remove_subtasks,
)


def make_task(
    task_id: str,
    depends_on: tuple[str, ...] = (),
    complexity: str = "small",
) -> SubTask:
    return SubTask(
        id=task_id,
        title=f"Task {task_id}",
        description=f"Description for {task_id}",
        implementation_prompt=f"Prompt for {task_id}",
        depends_on=list(depends_on),
        files=["agent.py"],
        acceptance_criteria=[f"{task_id} works"],
        verification="pytest",
        complexity=complexity,
    )


def make_plan(*tasks: SubTask) -> ImplementationPlan:
    return ImplementationPlan(
        summary="A plan",
        assumptions=["Python 3.10"],
        subtasks=list(tasks),
        open_questions=["None"],
        definition_of_done=["Everything works"],
    )


def by_id(plan: ImplementationPlan) -> dict[str, SubTask]:
    return {task.id: task for task in plan.subtasks}


def test_find_dependents_reports_remaining_tasks() -> None:
    plan = make_plan(
        make_task("T1"),
        make_task("T2", ("T1",)),
        make_task("T3", ("T1", "T2")),
    )

    assert find_dependents(plan.subtasks, {"T1"}) == {"T2": ["T1"], "T3": ["T1"]}
    assert find_dependents(plan.subtasks, {"T1", "T2"}) == {"T3": ["T1", "T2"]}
    assert find_dependents(plan.subtasks, {"T3"}) == {}


def test_rewire_repoints_dependencies_to_removed_task_parents() -> None:
    plan = make_plan(
        make_task("T1"),
        make_task("T2", ("T1",)),
        make_task("T3", ("T2",)),
    )

    updated = remove_subtasks(plan, ["T2"], strategy="rewire")

    assert [task.id for task in updated.subtasks] == ["T1", "T3"]
    assert by_id(updated)["T3"].depends_on == ["T1"]


def test_rewire_deduplicates_repointed_dependencies() -> None:
    plan = make_plan(
        make_task("T1"),
        make_task("T2", ("T1",)),
        make_task("T3", ("T1", "T2")),
    )

    updated = remove_subtasks(plan, ["T2"], strategy="rewire")

    assert by_id(updated)["T3"].depends_on == ["T1"]


def test_rewire_handles_chains_of_removed_tasks() -> None:
    plan = make_plan(
        make_task("T1"),
        make_task("T2", ("T1",)),
        make_task("T3", ("T2",)),
        make_task("T4", ("T3",)),
    )

    updated = remove_subtasks(plan, ["T2", "T3"], strategy="rewire")

    assert [task.id for task in updated.subtasks] == ["T1", "T4"]
    assert by_id(updated)["T4"].depends_on == ["T1"]


def test_rewire_drops_dependencies_that_cannot_be_repointed() -> None:
    plan = make_plan(
        make_task("T1"),
        make_task("T2"),
        make_task("T3", ("T1", "T2")),
    )

    updated = remove_subtasks(plan, ["T1", "T2"], strategy="rewire")

    assert [task.id for task in updated.subtasks] == ["T3"]
    assert by_id(updated)["T3"].depends_on == []


def test_cascade_removes_transitive_dependents() -> None:
    plan = make_plan(
        make_task("T1"),
        make_task("T2", ("T1",)),
        make_task("T3", ("T2",)),
        make_task("T4"),
    )

    updated = remove_subtasks(plan, ["T2"], strategy="cascade")

    assert [task.id for task in updated.subtasks] == ["T1", "T4"]


def test_block_refuses_while_dependents_remain() -> None:
    plan = make_plan(
        make_task("T1"),
        make_task("T2", ("T1",)),
    )

    with pytest.raises(RemovalError) as excinfo:
        remove_subtasks(plan, ["T1"], strategy="block")

    assert excinfo.value.blockers == ["T2"]
    assert "T2 depends on T1" in str(excinfo.value)


def test_block_succeeds_when_nothing_depends_on_the_selection() -> None:
    plan = make_plan(
        make_task("T1"),
        make_task("T2", ("T1",)),
    )

    updated = remove_subtasks(plan, ["T2"], strategy="block")

    assert [task.id for task in updated.subtasks] == ["T1"]


def test_unknown_ids_are_rejected() -> None:
    plan = make_plan(make_task("T1"))

    with pytest.raises(RemovalError) as excinfo:
        remove_subtasks(plan, ["T9"], strategy="rewire")

    assert excinfo.value.unknown_ids == ["T9"]


def test_removing_every_subtask_is_rejected() -> None:
    plan = make_plan(make_task("T1"), make_task("T2"))

    with pytest.raises(RemovalError, match="At least one subtask"):
        remove_subtasks(plan, ["T1", "T2"], strategy="rewire")


def test_removal_returns_topological_order() -> None:
    plan = make_plan(
        make_task("T3", ("T2",)),
        make_task("T1"),
        make_task("T2", ("T1",)),
        make_task("T4"),
    )

    updated = remove_subtasks(plan, ["T4"], strategy="rewire")

    ids = [task.id for task in updated.subtasks]
    assert ids.index("T1") < ids.index("T2") < ids.index("T3")


def test_removal_breaks_existing_cycles() -> None:
    plan = make_plan(
        make_task("T1", ("T2",)),
        make_task("T2", ("T1",)),
        make_task("T3"),
    )

    updated = remove_subtasks(plan, ["T3"], strategy="rewire")

    assert len(updated.subtasks) == 2
    assert updated.subtasks  # acyclic after normalization


def test_rewire_never_creates_a_self_dependency() -> None:
    plan = make_plan(
        make_task("T1"),
        make_task("T2", ("T3",)),
        make_task("T3", ("T2",)),
    )

    updated = remove_subtasks(plan, ["T2"], strategy="rewire")

    assert by_id(updated)["T3"].depends_on == []


def test_edit_subtask_updates_fields_and_normalizes() -> None:
    plan = make_plan(make_task("T1"), make_task("T2", ("T1",)))

    updated = edit_subtask(
        plan,
        "T2",
        {"title": "New title", "complexity": "large", "depends_on": []},
    )

    assert by_id(updated)["T2"].title == "New title"
    assert by_id(updated)["T2"].complexity == "large"
    assert by_id(updated)["T2"].depends_on == []


def test_edit_subtask_rejects_unknown_ids_and_fields() -> None:
    plan = make_plan(make_task("T1"))

    with pytest.raises(RemovalError) as excinfo:
        edit_subtask(plan, "T9", {"title": "x"})
    assert excinfo.value.unknown_ids == ["T9"]

    with pytest.raises(RemovalError, match="cannot be edited"):
        edit_subtask(plan, "T1", {"id": "T7"})


def test_document_round_trip_preserves_extra_fields() -> None:
    plan = make_plan(
        make_task("T1"),
        make_task("T2", ("T1",)),
    )
    document = {
        "issue": "Add rate limiting",
        "status": "approved",
        "approved": True,
        "review": {"approved": True, "summary": "ok"},
        "metadata": {"provider": "openai"},
        **plan.model_dump(),
    }

    parsed = plan_from_document(document)
    updated = apply_plan_to_document(document, parsed)

    assert updated["issue"] == "Add rate limiting"
    assert updated["review"] == {"approved": True, "summary": "ok"}
    assert updated["metadata"] == {"provider": "openai"}
    assert [task["id"] for task in updated["subtasks"]] == ["T1", "T2"]


def test_plan_from_document_requires_plan_fields() -> None:
    with pytest.raises(ValueError, match="missing plan field"):
        plan_from_document({"issue": "x"})
