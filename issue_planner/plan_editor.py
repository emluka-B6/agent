"""Edit implementation plans while keeping ids and dependencies consistent.

The functions here are deliberately UI-agnostic: the CLI, the web viewer, and
any future tooling can share the same removal and editing semantics.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

from issue_planner.models import ImplementationPlan, SubTask
from issue_planner.plan_repair import normalize_plan

RemoveStrategy = Literal["rewire", "block", "cascade"]

#: Top-level artifact keys that make up the :class:`ImplementationPlan`.
PLAN_FIELDS: tuple[str, ...] = (
    "summary",
    "assumptions",
    "subtasks",
    "open_questions",
    "definition_of_done",
)

#: Subtask fields that :func:`edit_subtask` accepts. Renaming an ``id`` would
#: require rewriting every dependency, so it is intentionally excluded.
EDITABLE_FIELDS = frozenset(
    {
        "title",
        "description",
        "implementation_prompt",
        "depends_on",
        "files",
        "acceptance_criteria",
        "verification",
        "complexity",
        "risks",
    }
)


class RemovalError(ValueError):
    """A removal or edit request that cannot be applied as specified."""

    def __init__(
        self,
        message: str,
        *,
        blockers: Iterable[str] = (),
        unknown_ids: Iterable[str] = (),
    ) -> None:
        super().__init__(message)
        #: Remaining subtasks that still depend on the selection.
        self.blockers = sorted(set(blockers))
        #: Ids that are not part of the plan.
        self.unknown_ids = sorted(set(unknown_ids))


def find_dependents(
    subtasks: Iterable[SubTask], ids: Iterable[str]
) -> dict[str, list[str]]:
    """Map remaining subtask ids to the removed ids they depend on."""
    removed = set(ids)
    dependents: dict[str, list[str]] = {}
    for task in subtasks:
        if task.id in removed:
            continue
        blocking = sorted({dep for dep in task.depends_on if dep in removed})
        if blocking:
            dependents[task.id] = blocking
    return dependents


def _cascade_ids(subtasks: Iterable[SubTask], ids: Iterable[str]) -> set[str]:
    """Expand a selection with every subtask that transitively depends on it."""
    removed = set(ids)
    changed = True
    while changed:
        changed = False
        for task in subtasks:
            if task.id in removed:
                continue
            if any(dep in removed for dep in task.depends_on):
                removed.add(task.id)
                changed = True
    return removed


def _rewire_tasks(
    remaining: Iterable[SubTask],
    removed_ids: set[str],
    removed_dependencies: Mapping[str, Iterable[str]],
) -> list[SubTask]:
    """Replace dependencies on removed tasks with their own dependencies."""

    def expand(task_id: str, seen: set[str]) -> list[str]:
        if task_id not in removed_ids:
            return [task_id]
        expanded: list[str] = []
        for dependency in removed_dependencies.get(task_id, ()):
            if dependency in seen:
                continue
            for item in expand(dependency, seen | {dependency}):
                if item not in expanded:
                    expanded.append(item)
        return expanded

    rewired: list[SubTask] = []
    for task in remaining:
        dependencies: list[str] = []
        for dependency in task.depends_on:
            for item in expand(dependency, {dependency}):
                if item != task.id and item not in dependencies:
                    dependencies.append(item)
        rewired.append(task.model_copy(update={"depends_on": dependencies}))
    return rewired


def remove_subtasks(
    plan: ImplementationPlan,
    ids: Iterable[str],
    strategy: RemoveStrategy = "rewire",
) -> ImplementationPlan:
    """Remove subtasks and return a repaired, topologically ordered plan.

    Strategies:

    - ``rewire`` (default): replace dependencies on removed tasks with the
      removed tasks' own dependencies.
    - ``cascade``: also remove every subtask that transitively depends on the
      selection.
    - ``block``: refuse the removal while other subtasks still depend on the
      selection.
    """
    requested = {str(task_id).strip() for task_id in ids if str(task_id).strip()}
    if not requested:
        raise RemovalError("Select at least one subtask to remove.")

    known = {task.id for task in plan.subtasks}
    unknown = requested - known
    if unknown:
        raise RemovalError(
            f"Unknown subtask id(s): {', '.join(sorted(unknown))}.",
            unknown_ids=unknown,
        )

    if strategy == "block":
        blockers = find_dependents(plan.subtasks, requested)
        if blockers:
            details = "; ".join(
                f"{task_id} depends on {', '.join(dependencies)}"
                for task_id, dependencies in blockers.items()
            )
            raise RemovalError(
                "These subtasks still depend on the selection: " + details + ".",
                blockers=blockers,
            )
        remove_ids = requested
    elif strategy == "cascade":
        remove_ids = _cascade_ids(plan.subtasks, requested)
    elif strategy == "rewire":
        remove_ids = requested
    else:
        raise RemovalError(f"Unknown removal strategy '{strategy}'.")

    remaining = [task for task in plan.subtasks if task.id not in remove_ids]
    if not remaining:
        raise RemovalError("At least one subtask must remain in the plan.")

    if strategy == "rewire":
        removed_dependencies = {
            task.id: list(task.depends_on)
            for task in plan.subtasks
            if task.id in remove_ids
        }
        remaining = _rewire_tasks(remaining, remove_ids, removed_dependencies)

    updated = plan.model_copy(update={"subtasks": remaining})
    return normalize_plan(updated)


def edit_subtask(
    plan: ImplementationPlan, task_id: str, patch: Mapping[str, Any]
) -> ImplementationPlan:
    """Apply a field patch to one subtask and repair the plan."""
    matches = [task for task in plan.subtasks if task.id == task_id]
    if not matches:
        raise RemovalError(f"Unknown subtask id '{task_id}'.", unknown_ids={task_id})

    unsupported = set(patch) - EDITABLE_FIELDS
    if unsupported:
        raise RemovalError(
            "These fields cannot be edited: " + ", ".join(sorted(unsupported)) + "."
        )

    updated_task = matches[0].model_copy(update=dict(patch))
    # Re-validate so that an edit cannot produce an invalid subtask.
    updated_task = SubTask.model_validate(updated_task.model_dump())
    subtasks = [updated_task if task.id == task_id else task for task in plan.subtasks]
    return normalize_plan(plan.model_copy(update={"subtasks": subtasks}))


def plan_from_document(document: Mapping[str, Any]) -> ImplementationPlan:
    """Extract and validate the plan part of an output artifact."""
    missing = [field for field in PLAN_FIELDS if field not in document]
    if missing:
        raise ValueError(
            "Document is missing plan field(s): " + ", ".join(missing) + "."
        )
    return ImplementationPlan.model_validate(
        {field: document[field] for field in PLAN_FIELDS}
    )


def apply_plan_to_document(
    document: Mapping[str, Any], plan: ImplementationPlan
) -> dict[str, Any]:
    """Merge an edited plan back into the full artifact."""
    updated = dict(document)
    updated.update(plan.model_dump())
    return updated
