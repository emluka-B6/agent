"""Load a previous artifact so a run can continue where the last one stopped.

A run ends either because the critic approved the plan or because the revision
budget ran out. ``status: max_iterations_reached`` therefore means "not approved
yet, but no budget left" -- the plan, the last review, and the research that
supported them are still on disk. This module reads such an artifact back into
the workflow inputs so ``agent.py --resume`` can continue the analysis instead
of regenerating the plan from scratch.

Only JSON artifacts are resumable: the Markdown rendering is a report, not a
machine-readable state. Artifacts still marked ``running`` are rejected because
a concurrent or interrupted run may not have written a consistent state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from issue_planner.config import DEFAULT_OUTPUT_DIR
from issue_planner.domains import DEFAULT_DOMAIN, get_domain
from issue_planner.models import (
    Critique,
    ImplementationPlan,
    ResearchSummary,
    Source,
    WorkflowState,
)
from issue_planner.output import RUNNING_STATUS

#: Value accepted by ``--resume`` that selects the newest artifact on disk.
RESUME_LATEST = "latest"

#: Artifact naming scheme written by :func:`issue_planner.output.unique_output_path`.
_ARTIFACT_GLOB = "output-*"
_MARKDOWN_SUFFIXES = {".md", ".markdown"}


@dataclass(frozen=True)
class ResumeState:
    """A previous artifact restored into workflow inputs."""

    path: Path
    issue: str
    domain: str
    plan: ImplementationPlan
    critique: Critique | None
    research: ResearchSummary | None
    sources: tuple[Source, ...]
    research_error: str | None
    evaluations: int
    max_iterations: int
    research_calls: int
    max_research_calls: int
    runs: int
    status: str

    @property
    def name(self) -> str:
        """File name of the artifact, for console messages."""
        return self.path.name

    def as_initial_state(self) -> WorkflowState:
        """Return the state a continued workflow starts from.

        ``approved`` is reset so the critic has to approve the revised plan
        again; the restored plan keeps its subtask ids, which keeps them stable
        across runs and in the viewer.
        """
        state: WorkflowState = {
            "issue": self.issue,
            "domain": self.domain,
            "research": self.research,
            "sources": list(self.sources),
            "research_error": self.research_error,
            "research_calls": self.research_calls,
            "research_requests": [],
            "plan": self.plan,
            "critique": self.critique,
            "evaluations": self.evaluations,
            "approved": False,
        }
        return state


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def latest_artifact(directory: Path | str = DEFAULT_OUTPUT_DIR) -> Path:
    """Return the most recently modified JSON artifact in ``directory``."""
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"No artifact directory to resume from: {directory}")

    candidates = sorted(
        (
            path
            for path in directory.glob(_ARTIFACT_GLOB)
            if path.suffix.lower() == ".json" and path.is_file()
        ),
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    if not candidates:
        raise FileNotFoundError(f"No JSON artifact to resume from in {directory}.")
    return candidates[-1]


def resolve_resume_path(
    value: str | Path, output_dir: Path | str = DEFAULT_OUTPUT_DIR
) -> Path:
    """Resolve a ``--resume`` value into the artifact to continue.

    The value may be a JSON artifact, a directory (the newest artifact inside it
    is used), or the word ``latest`` (the newest artifact in ``output_dir``).
    """
    raw = str(value).strip()
    if not raw:
        raise ValueError("Resume needs a JSON artifact, a directory, or 'latest'.")
    if raw.lower() == RESUME_LATEST:
        return latest_artifact(output_dir)

    path = Path(raw).expanduser()
    if path.is_dir():
        return latest_artifact(path)
    if not path.exists():
        raise FileNotFoundError(f"Resume artifact not found: {path}")
    return path


# ---------------------------------------------------------------------------
# Artifact parsing
# ---------------------------------------------------------------------------
def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _as_count(value: Any, default: int = 0) -> int:
    """Return a non-negative integer, falling back to ``default``."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number >= 0 else default


def _validated(schema: type[BaseModel], data: Any, label: str, path: Path) -> Any:
    """Validate one part of an artifact, reporting only the first problem."""
    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(part) for part in first.get("loc", ())) or label
        raise ValueError(
            f"{path.name} cannot be resumed: {label} is invalid "
            f"({location}: {first.get('msg', 'validation failed')})."
        ) from exc


def _plan_from(document: dict[str, Any], path: Path) -> ImplementationPlan:
    """Rebuild the plan from the fields written by ``build_document``."""
    subtasks = _as_list(document.get("subtasks"))
    if not subtasks:
        raise ValueError(
            f"{path.name} has no subtasks, so there is no plan to continue."
        )
    data = {
        "summary": str(document.get("summary") or ""),
        "assumptions": _as_list(document.get("assumptions")),
        "subtasks": subtasks,
        "open_questions": _as_list(document.get("open_questions")),
        "definition_of_done": _as_list(document.get("definition_of_done")),
    }
    return _validated(ImplementationPlan, data, "plan", path)


def load_resume_state(path: Path | str) -> ResumeState:
    """Read a previous JSON artifact into a :class:`ResumeState`.

    Raises :class:`ValueError` for artifacts that cannot be continued (Markdown
    reports, partial artifacts still marked ``running``, files without a plan)
    and :class:`FileNotFoundError` when the file does not exist.
    """
    path = Path(path).expanduser()
    if path.suffix.lower() in _MARKDOWN_SUFFIXES:
        raise ValueError(
            f"{path.name} is a Markdown report; only JSON artifacts can be resumed."
        )
    if not path.exists():
        raise FileNotFoundError(f"Resume artifact not found: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} is not a valid JSON artifact: {exc}") from exc
    except OSError as exc:
        raise OSError(f"could not read {path}: {exc}") from exc

    if not isinstance(document, dict):
        raise ValueError(f"{path.name} does not contain a planning artifact.")

    status = str(document.get("status") or "")
    if status == RUNNING_STATUS:
        raise ValueError(
            f"{path.name} is still marked as '{RUNNING_STATUS}'; "
            "wait for the run to finish before resuming it."
        )

    issue = str(document.get("issue") or "").strip()
    if not issue:
        raise ValueError(f"{path.name} has no issue, so there is nothing to continue.")

    raw_domain = str(document.get("domain") or DEFAULT_DOMAIN)
    try:
        domain = get_domain(raw_domain).key
    except ValueError as exc:
        raise ValueError(f"{path.name} cannot be resumed: {exc}") from exc

    plan = _plan_from(document, path)

    review = document.get("review")
    critique = (
        _validated(Critique, review, "review", path)
        if isinstance(review, dict)
        else None
    )

    research_block = document.get("research")
    if not isinstance(research_block, dict):
        research_block = {}
    summary_data = research_block.get("summary")
    research = (
        _validated(ResearchSummary, summary_data, "research summary", path)
        if isinstance(summary_data, dict)
        else None
    )
    sources = tuple(
        _validated(Source, source, "research source", path)
        for source in _as_list(research_block.get("sources"))
        if isinstance(source, dict)
    )
    raw_error = research_block.get("error")
    research_error = str(raw_error) if raw_error else None

    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    evaluations = _as_count(metadata.get("evaluations"))
    research_calls = _as_count(metadata.get("research_calls"))
    # Older artifacts may have spent more than their recorded cap (for example
    # after a manual edit); never hand back a budget that is already exhausted.
    max_iterations = max(
        _as_count(metadata.get("max_iterations"), evaluations), evaluations
    )
    max_research_calls = max(
        _as_count(metadata.get("max_research_calls"), research_calls), research_calls
    )

    return ResumeState(
        path=path.resolve(),
        issue=issue,
        domain=domain,
        plan=plan,
        critique=critique,
        research=research,
        sources=sources,
        research_error=research_error,
        evaluations=evaluations,
        max_iterations=max_iterations,
        research_calls=research_calls,
        max_research_calls=max_research_calls,
        runs=max(1, _as_count(metadata.get("runs"), 1)),
        status=status or "unknown",
    )
