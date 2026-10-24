"""Repair model-generated plans so that they can be scheduled reliably."""

from __future__ import annotations

import sys

from issue_planner.models import ImplementationPlan, SubTask


def break_dependency_cycles(subtasks: list[SubTask]) -> list[SubTask]:
    """Remove back edges so that the dependency graph is acyclic."""
    by_id = {task.id: task for task in subtasks}
    dependencies = {task.id: list(task.depends_on) for task in subtasks}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        visiting.add(task_id)
        for dependency in list(dependencies[task_id]):
            if dependency not in by_id:
                continue
            if dependency in visiting:
                dependencies[task_id].remove(dependency)
                print(
                    f"warning: removed cyclic dependency {task_id} -> {dependency}.",
                    file=sys.stderr,
                )
            elif dependency not in visited:
                visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task in subtasks:
        if task.id not in visited:
            visit(task.id)

    return [
        task.model_copy(update={"depends_on": dependencies[task.id]})
        for task in subtasks
    ]


def topological_order(subtasks: list[SubTask]) -> list[SubTask]:
    """Order subtasks so that dependencies come first, where possible."""
    by_id = {task.id: task for task in subtasks}
    ordered: list[SubTask] = []
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visited:
            return
        if task_id in visiting:
            print(
                f"warning: dependency cycle detected at {task_id}; "
                "keeping the original relative order.",
                file=sys.stderr,
            )
            return
        visiting.add(task_id)
        for dependency in by_id[task_id].depends_on:
            if dependency in by_id:
                visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)
        ordered.append(by_id[task_id])

    for task in subtasks:
        visit(task.id)
    for task in subtasks:
        if task.id not in visited:
            ordered.append(task)
    return ordered


def normalize_plan(plan: ImplementationPlan) -> ImplementationPlan:
    """Repair duplicate IDs, unknown dependencies, and dependency order."""
    seen: set[str] = set()
    normalized: list[SubTask] = []

    for index, task in enumerate(plan.subtasks, start=1):
        task_id = task.id.strip() or f"T{index}"
        if task_id in seen:
            candidate = f"T{index}"
            while candidate in seen:
                index += 1
                candidate = f"T{index}"
            print(
                f"warning: duplicate subtask id '{task_id}' renamed to '{candidate}'.",
                file=sys.stderr,
            )
            task_id = candidate
        seen.add(task_id)
        normalized.append(task.model_copy(update={"id": task_id}))

    valid_ids = {task.id for task in normalized}
    cleaned: list[SubTask] = []
    for task in normalized:
        dependencies: list[str] = []
        for dependency in task.depends_on:
            if (
                dependency in valid_ids
                and dependency != task.id
                and dependency not in dependencies
            ):
                dependencies.append(dependency)
        cleaned.append(task.model_copy(update={"depends_on": dependencies}))

    acyclic = break_dependency_cycles(cleaned)
    return plan.model_copy(update={"subtasks": topological_order(acyclic)})
