"""Edit implementation plans while keeping ids and dependencies consistent.

The functions here are deliberately UI-agnostic: the CLI, the web viewer, and
any future tooling can share the same removal and editing semantics.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

from issue_planner.models import ImplementationPlan, PlanRevision, SubTask
from issue_planner.plan_repair import normalize_plan

RemoveStrategy = Literal["rewire", "block", "cascade"]

#: Matches generated subtask ids so new ids can continue the numbering.
_TASK_ID_PATTERN = re.compile(r"^T(\d+)$")

#: Ordering used to keep the largest complexity when subtasks are merged.
_COMPLEXITY_ORDER: dict[str, int] = {"small": 0, "medium": 1, "large": 2}

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


def _dedupe(items: Iterable[str]) -> list[str]:
    """Return non-empty items without duplicates, preserving their order."""
    unique: list[str] = []
    for item in items:
        if item and item not in unique:
            unique.append(item)
    return unique


def _next_task_id(used: set[str]) -> str:
    """Return the next free ``T<n>`` identifier for a set of ids."""
    highest = 0
    for task_id in used:
        match = _TASK_ID_PATTERN.match(task_id.strip())
        if match:
            highest = max(highest, int(match.group(1)))
    candidate = f"T{highest + 1}"
    while candidate in used:
        highest += 1
        candidate = f"T{highest + 1}"
    return candidate


def next_task_id(subtasks: Iterable[SubTask]) -> str:
    """Return the next free ``T<n>`` identifier for a subtask collection."""
    return _next_task_id({task.id for task in subtasks})


def _require_known_dependencies(task: SubTask, valid_ids: set[str]) -> None:
    """Reject an edit that references a missing id or the subtask itself."""
    unknown = sorted(
        {dependency for dependency in task.depends_on if dependency not in valid_ids}
    )
    if unknown:
        raise RemovalError(
            f"Subtask '{task.id}' depends on unknown id(s): {', '.join(unknown)}.",
            unknown_ids=unknown,
        )
    if task.id in task.depends_on:
        raise RemovalError(f"Subtask '{task.id}' cannot depend on itself.")


def _removal_ids(
    subtasks: Iterable[SubTask], ids: Iterable[str], strategy: str
) -> set[str]:
    """Validate a removal request and return every id that must be removed."""
    subtasks = list(subtasks)
    requested = {str(task_id).strip() for task_id in ids if str(task_id).strip()}
    known = {task.id for task in subtasks}
    unknown = requested - known
    if unknown:
        raise RemovalError(
            f"Unknown subtask id(s): {', '.join(sorted(unknown))}.",
            unknown_ids=unknown,
        )

    if strategy == "block":
        blockers = find_dependents(subtasks, requested)
        if blockers:
            details = "; ".join(
                f"{task_id} depends on {', '.join(dependencies)}"
                for task_id, dependencies in blockers.items()
            )
            raise RemovalError(
                "These subtasks still depend on the selection: " + details + ".",
                blockers=blockers,
            )
        return requested
    if strategy == "cascade":
        return _cascade_ids(subtasks, requested)
    if strategy == "rewire":
        return requested
    raise RemovalError(f"Unknown removal strategy '{strategy}'.")


def _drop_subtasks(
    subtasks: Iterable[SubTask], remove_ids: set[str], strategy: str
) -> list[SubTask]:
    """Remove subtasks and rewire dependents without validating the result."""
    subtasks = list(subtasks)
    remaining = [task for task in subtasks if task.id not in remove_ids]
    if strategy == "rewire" and remove_ids:
        removed_dependencies = {
            task.id: list(task.depends_on)
            for task in subtasks
            if task.id in remove_ids
        }
        remaining = _rewire_tasks(remaining, remove_ids, removed_dependencies)
    return remaining


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
    selection = [str(task_id).strip() for task_id in ids if str(task_id).strip()]
    if not selection:
        raise RemovalError("Select at least one subtask to remove.")

    remove_ids = _removal_ids(plan.subtasks, selection, strategy)
    remaining = _drop_subtasks(plan.subtasks, remove_ids, strategy)
    if not remaining:
        raise RemovalError("At least one subtask must remain in the plan.")

    updated = plan.model_copy(update={"subtasks": remaining})
    return normalize_plan(updated)


def insert_subtask(
    plan: ImplementationPlan,
    task: SubTask,
    *,
    after: str | None = None,
) -> ImplementationPlan:
    """Add one subtask, optionally directly after an existing subtask.

    An empty ``task.id`` is replaced with the next free ``T<n>`` identifier. A
    duplicate id or an unknown dependency is rejected so that an edit cannot
    silently produce a plan that no longer schedules.
    """
    subtasks = list(plan.subtasks)
    used = {existing.id for existing in subtasks}
    task_id = task.id.strip()
    if task_id and task_id in used:
        raise RemovalError(f"Subtask id '{task_id}' already exists.")
    if not task_id:
        task_id = _next_task_id(used)

    inserted = task.model_copy(update={"id": task_id})
    _require_known_dependencies(inserted, used | {task_id})

    if after is None:
        subtasks.append(inserted)
    else:
        ids = [existing.id for existing in subtasks]
        if after not in ids:
            raise RemovalError(f"Unknown subtask id '{after}'.", unknown_ids={after})
        subtasks.insert(ids.index(after) + 1, inserted)

    return normalize_plan(plan.model_copy(update={"subtasks": subtasks}))


def split_subtask(
    plan: ImplementationPlan,
    task_id: str,
    parts: Sequence[SubTask],
) -> ImplementationPlan:
    """Replace one subtask with two or more chained subtasks.

    The replacements take the position of the original subtask and run in the
    given order: each part depends on the previous one, the first part inherits
    the original dependencies, and every subtask that depended on the original
    now depends on the last part.
    """
    matches = [task for task in plan.subtasks if task.id == task_id]
    if not matches:
        raise RemovalError(f"Unknown subtask id '{task_id}'.", unknown_ids={task_id})
    original = matches[0]

    replacements = list(parts)
    if len(replacements) < 2:
        raise RemovalError("A split needs at least two replacement subtasks.")

    remaining = [task for task in plan.subtasks if task.id != task_id]
    used = {task.id for task in remaining}
    assigned: list[SubTask] = []
    for index, part in enumerate(replacements):
        part_id = part.id.strip()
        if not part_id or part_id in used:
            part_id = _next_task_id(used)
        used.add(part_id)
        dependencies = list(part.depends_on)
        if index == 0 and not dependencies:
            dependencies = list(original.depends_on)
        assigned.append(
            part.model_copy(
                update={"id": part_id, "depends_on": _dedupe(dependencies)}
            )
        )

    chained: list[SubTask] = []
    for index, part in enumerate(assigned):
        dependencies = list(part.depends_on)
        if index:
            dependencies = _dedupe([*dependencies, assigned[index - 1].id])
        chained.append(part.model_copy(update={"depends_on": dependencies}))
    for part in chained:
        _require_known_dependencies(part, used | {part.id})

    last_id = chained[-1].id
    rewired = {
        task.id: task.model_copy(
            update={
                "depends_on": _dedupe(
                    last_id if dependency == task_id else dependency
                    for dependency in task.depends_on
                )
            }
        )
        for task in remaining
    }

    subtasks: list[SubTask] = []
    for task in plan.subtasks:
        if task.id == task_id:
            subtasks.extend(chained)
        else:
            subtasks.append(rewired[task.id])
    return normalize_plan(plan.model_copy(update={"subtasks": subtasks}))


def merge_subtasks(
    plan: ImplementationPlan,
    ids: Iterable[str],
    *,
    title: str,
    description: str,
    implementation_prompt: str,
    verification: str | None = None,
) -> ImplementationPlan:
    """Merge two or more subtasks into one that keeps the earliest id.

    Files, acceptance criteria, risks, and dependencies are unioned and the
    merged subtask takes the largest complexity. The caller supplies the new
    title, description, and prompt because the merged scope has to be rewritten.
    """
    selection = {str(task_id).strip() for task_id in ids if str(task_id).strip()}
    if len(selection) < 2:
        raise RemovalError("A merge needs at least two subtask ids.")
    known = {task.id for task in plan.subtasks}
    unknown = selection - known
    if unknown:
        raise RemovalError(
            f"Unknown subtask id(s): {', '.join(sorted(unknown))}.",
            unknown_ids=unknown,
        )

    ordered = [task for task in plan.subtasks if task.id in selection]
    kept = ordered[0]
    dependencies = [
        dependency
        for dependency in _dedupe(
            dependency for task in ordered for dependency in task.depends_on
        )
        if dependency not in selection
    ]
    criteria = _dedupe(
        criterion for task in ordered for criterion in task.acceptance_criteria
    )
    merged = kept.model_copy(
        update={
            "title": title,
            "description": description,
            "implementation_prompt": implementation_prompt,
            "depends_on": dependencies,
            "files": _dedupe(item for task in ordered for item in task.files),
            "acceptance_criteria": criteria or list(kept.acceptance_criteria),
            "verification": verification
            or "; ".join(
                _dedupe(task.verification for task in ordered if task.verification)
            ),
            "complexity": max(
                (task.complexity for task in ordered),
                key=lambda value: _COMPLEXITY_ORDER[value],
            ),
            "risks": _dedupe(item for task in ordered for item in task.risks),
        }
    )
    _require_known_dependencies(merged, (known - selection) | {merged.id})

    subtasks: list[SubTask] = []
    for task in plan.subtasks:
        if task.id == kept.id:
            subtasks.append(merged)
        elif task.id in selection:
            continue
        else:
            subtasks.append(
                task.model_copy(
                    update={
                        "depends_on": _dedupe(
                            kept.id if dependency in selection else dependency
                            for dependency in task.depends_on
                        )
                    }
                )
            )
    return normalize_plan(plan.model_copy(update={"subtasks": subtasks}))


def apply_revision(
    plan: ImplementationPlan, revision: PlanRevision
) -> ImplementationPlan:
    """Apply a planner patch to an existing plan.

    Removals use the ``rewire`` strategy so dependents keep a valid execution
    order, new or changed subtasks are upserted by id and appended when their id
    is new, and scalar fields are replaced only when the patch supplies them.
    Untouched subtasks keep their ids, content, and position.
    """
    remove_ids = [
        str(task_id).strip()
        for task_id in revision.remove_subtasks
        if str(task_id).strip()
    ]
    subtasks = list(plan.subtasks)
    if remove_ids:
        known = {task.id for task in subtasks}
        unknown = {task_id for task_id in remove_ids if task_id not in known}
        if unknown:
            raise RemovalError(
                f"Unknown subtask id(s): {', '.join(sorted(unknown))}.",
                unknown_ids=unknown,
            )
        subtasks = _drop_subtasks(subtasks, set(remove_ids), "rewire")

    upserted = list(subtasks)
    for incoming in revision.upsert_subtasks:
        task_id = incoming.id.strip()
        if not task_id:
            task_id = _next_task_id({task.id for task in upserted})
            incoming = incoming.model_copy(update={"id": task_id})
        for index, existing in enumerate(upserted):
            if existing.id == task_id:
                upserted[index] = incoming
                break
        else:
            upserted.append(incoming)

    if not upserted:
        raise RemovalError("At least one subtask must remain in the plan.")

    updates: dict[str, Any] = {
        "subtasks": upserted,
        "research_requests": list(revision.research_requests),
    }
    if revision.summary is not None:
        updates["summary"] = revision.summary
    if revision.assumptions is not None:
        updates["assumptions"] = list(revision.assumptions)
    if revision.definition_of_done is not None:
        updates["definition_of_done"] = list(revision.definition_of_done)
    if revision.open_questions is not None:
        updates["open_questions"] = list(revision.open_questions)

    return normalize_plan(plan.model_copy(update=updates))


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
    _require_known_dependencies(updated_task, {task.id for task in plan.subtasks})
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
