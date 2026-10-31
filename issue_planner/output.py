"""Build, render, and persist the final planning artifact."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from issue_planner.budget import RunBudget
from issue_planner.config import DEFAULT_MAX_RESEARCH_CALLS, DEFAULT_OUTPUT_DIR
from issue_planner.domains import DEFAULT_DOMAIN
from issue_planner.models import WorkflowState

#: Prefix and suffix used for automatically generated artifact names.
OUTPUT_PREFIX = "output"
OUTPUT_HASH_LENGTH = 12
_FORMAT_SUFFIXES = {"json": ".json", "markdown": ".md"}

#: Status recorded by partial artifacts while the workflow is still running.
RUNNING_STATUS = "running"


def _dump_model(value: Any) -> dict[str, Any] | None:
    """Return a JSON-serialisable mapping for a Pydantic model or dict."""
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump()
    return None


def document_status(state: WorkflowState, max_iterations: int) -> str:
    """Return the final artifact status for a workflow state."""
    if state.get("approved"):
        return "approved"
    if state.get("evaluations", 0) >= max_iterations:
        return "max_iterations_reached"
    return "needs_improvement"


def _one_line(text: str, limit: int = 300) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


#: Plan-level bullet sections, in the order that decides which section keeps a
#: repeated item. The definition of done wins because it is the completion
#: checklist the reader relies on; after it a premise stays an assumption and a
#: decision stays an open question. ``summary`` is prose, so it is not part of
#: this cleanup.
_BULLET_SECTIONS: tuple[str, ...] = (
    "definition_of_done",
    "assumptions",
    "open_questions",
)


def _item_key(item: str) -> str:
    """Return a comparison key for a bullet (case, spacing, punctuation free)."""
    return re.sub(r"[\W_]+", " ", str(item).casefold()).strip()


def _unique_items(items: Iterable[str], seen: set[str]) -> list[str]:
    """Strip items and drop empty or already ``seen`` ones, updating ``seen``."""
    unique: list[str] = []
    for item in items:
        text = str(item).strip()
        key = _item_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(text)
    return unique


def dedupe_sections(document: dict[str, Any]) -> dict[str, Any]:
    """Remove repeated bullets from the plan-level sections, in place.

    The planner is instructed to keep the definition of done, the assumptions,
    and the open questions distinct, but structured output can still restate the
    same item in two places. Exact repeats (ignoring case, punctuation, and
    spacing) are dropped here so that every item appears in exactly one section.
    The definition of done wins, then assumptions, then open questions.
    """
    seen: set[str] = set()
    for key in _BULLET_SECTIONS:
        if key not in document:
            continue
        original = [str(item).strip() for item in (document.get(key) or [])]
        unique = _unique_items(original, seen)
        # The plan schema requires at least one definition-of-done entry, so an
        # all-blank list is left as it is instead of becoming empty.
        document[key] = unique or (original if key == "definition_of_done" else [])
    return document


def build_document(
    state: WorkflowState,
    provider: str,
    model_name: str,
    max_iterations: int,
    *,
    generated_at: str | None = None,
    max_research_calls: int = DEFAULT_MAX_RESEARCH_CALLS,
    budget: RunBudget | None = None,
) -> dict[str, Any]:
    """Build the serializable artifact written to the output file.

    ``budget`` describes the cumulative iteration/research budget of a run that
    may continue an earlier artifact. When it is given it supersedes
    ``max_iterations`` and ``max_research_calls`` and adds the totals a chained
    run needs (``runs``, ``resumed_from``, and the per-run counters).
    """
    plan = state.get("plan")
    if plan is None:
        raise RuntimeError("The workflow finished without producing a plan.")

    run_budget = budget or RunBudget.fresh(max_iterations, max_research_calls)
    plan_data = plan.model_dump()
    critique = state.get("critique")
    research = state.get("research")

    document = {
        "issue": state.get("issue", ""),
        "domain": state.get("domain", DEFAULT_DOMAIN),
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "status": document_status(state, run_budget.max_iterations),
        "approved": bool(state.get("approved")),
        "summary": plan_data["summary"],
        "subtasks": plan_data["subtasks"],
        "assumptions": plan_data["assumptions"],
        "open_questions": plan_data["open_questions"],
        "definition_of_done": plan_data["definition_of_done"],
        "review": critique.model_dump() if critique else None,
        "research": {
            "summary": research.model_dump() if research else None,
            "sources": [source.model_dump() for source in state.get("sources", [])],
            "error": state.get("research_error"),
        },
        "metadata": {
            "provider": provider,
            "model": model_name,
            **run_budget.to_metadata(
                state.get("evaluations", 0), state.get("research_calls", 0)
            ),
        },
    }
    return dedupe_sections(document)


def build_progress_document(
    state: WorkflowState,
    provider: str,
    model_name: str,
    max_iterations: int,
    *,
    stage: str,
    completed_stages: Iterable[str] = (),
    running_stage: str | None = None,
    generated_at: str | None = None,
    max_research_calls: int = DEFAULT_MAX_RESEARCH_CALLS,
    budget: RunBudget | None = None,
) -> dict[str, Any]:
    """Build a partial artifact for an in-progress workflow.

    The document has the same shape as :func:`build_document` so a run that is
    interrupted still leaves a readable JSON/Markdown file. Missing plan data is
    replaced with empty values and ``metadata.progress`` records which stage is
    running, which stages finished, and how many subtasks exist so far.
    """
    run_budget = budget or RunBudget.fresh(max_iterations, max_research_calls)
    plan_data = _dump_model(state.get("plan")) or {}
    critique = _dump_model(state.get("critique"))
    research = _dump_model(state.get("research"))
    subtasks = list(plan_data.get("subtasks") or [])
    completed = list(completed_stages)

    document = {
        "issue": state.get("issue", ""),
        "domain": state.get("domain", DEFAULT_DOMAIN),
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "status": RUNNING_STATUS,
        "approved": bool(state.get("approved")),
        "summary": plan_data.get("summary", ""),
        "subtasks": subtasks,
        "assumptions": list(plan_data.get("assumptions") or []),
        "open_questions": list(plan_data.get("open_questions") or []),
        "definition_of_done": list(plan_data.get("definition_of_done") or []),
        "review": critique,
        "research": {
            "summary": research,
            "sources": [
                _dump_model(source) or {} for source in state.get("sources", [])
            ],
            "error": state.get("research_error"),
        },
        "metadata": {
            "provider": provider,
            "model": model_name,
            **run_budget.to_metadata(
                state.get("evaluations", 0), state.get("research_calls", 0)
            ),
            "progress": {
                "partial": True,
                "stage": stage,
                "running_stage": running_stage,
                "completed_stages": completed,
                "subtask_count": len(subtasks),
                "subtask_titles": [
                    str(task.get("title", "")) for task in subtasks
                ],
            },
        },
    }
    return dedupe_sections(document)


def render_markdown(document: dict[str, Any]) -> str:
    """Render the artifact as a human-readable Markdown plan."""
    lines: list[str] = [
        "# Implementation plan",
        "",
        f"**Status:** {document['status']}",
        f"**Approved:** {'yes' if document['approved'] else 'no'}",
        f"**Domain:** {document.get('domain', DEFAULT_DOMAIN)}",
        "",
        "## Issue",
        "",
        document["issue"],
        "",
        "## Summary",
        "",
        document["summary"],
        "",
        "## Subtasks",
        "",
    ]

    for index, task in enumerate(document["subtasks"], start=1):
        lines.extend(
            [
                f"### {index}. {task['id']} - {task['title']}",
                "",
                f"**Description:** {task['description']}",
                "",
                "**Depends on:** "
                + (", ".join(task["depends_on"]) if task["depends_on"] else "none"),
                "**Complexity:** " + task["complexity"],
                "**Expected artifacts / files:** "
                + (", ".join(task["files"]) if task["files"] else "not specified"),
                "",
                "**Acceptance criteria:**",
            ]
        )
        lines.extend(f"- {criterion}" for criterion in task["acceptance_criteria"])
        lines.extend(
            [
                "",
                f"**Verification:** {task['verification']}",
                "",
            ]
        )
        if task["risks"]:
            lines.append("**Risks:**")
            lines.extend(f"- {risk}" for risk in task["risks"])
            lines.append("")
        lines.extend(
            [
                "**Implementation prompt:**",
                "",
                "````text",
                task["implementation_prompt"],
                "````",
                "",
            ]
        )

    lines.extend(["## Assumptions", ""])
    if document["assumptions"]:
        lines.extend(f"- {item}" for item in document["assumptions"])
    else:
        lines.append("- none")
    lines.append("")

    lines.extend(["## Open questions", ""])
    if document["open_questions"]:
        lines.extend(f"- {item}" for item in document["open_questions"])
    else:
        lines.append("- none")
    lines.append("")

    lines.extend(["## Definition of done", ""])
    if document["definition_of_done"]:
        lines.extend(f"- {item}" for item in document["definition_of_done"])
    else:
        lines.append("- none")
    lines.append("")

    review = document.get("review")
    if review:
        lines.extend(
            [
                "## Review",
                "",
                f"**Approved:** {'yes' if review['approved'] else 'no'}",
                "",
                review["summary"],
                "",
            ]
        )
        if review["blocking_issues"]:
            lines.append("**Blocking issues:**")
            lines.extend(f"- {item}" for item in review["blocking_issues"])
            lines.append("")
        if review["suggestions"]:
            lines.append("**Suggestions:**")
            lines.extend(f"- {item}" for item in review["suggestions"])
            lines.append("")

    research = document.get("research") or {}
    lines.extend(["## Research", ""])
    if research.get("error"):
        lines.extend([f"**Limitation:** {research['error']}", ""])

    summary = research.get("summary")
    if summary:
        if summary["key_facts"]:
            lines.append("**Key facts:**")
            lines.extend(f"- {item}" for item in summary["key_facts"])
            lines.append("")
        if summary["constraints"]:
            lines.append("**Constraints:**")
            lines.extend(f"- {item}" for item in summary["constraints"])
            lines.append("")
        lines.extend(
            [
                "**Recommended approach:**",
                "",
                summary["recommended_approach"],
                "",
            ]
        )
        if summary["assumptions"]:
            lines.append("**Research assumptions:**")
            lines.extend(f"- {item}" for item in summary["assumptions"])
            lines.append("")

    sources = research.get("sources") or []
    if sources:
        lines.append("**Sources:**")
        for source in sources:
            title = source.get("title") or source.get("url") or "source"
            if source.get("url"):
                lines.append(f"- [{title}]({source['url']})")
            else:
                lines.append(f"- {title}")
            if source.get("snippet"):
                lines.append(f"  - {_one_line(source['snippet'])}")
        lines.append("")

    metadata = document["metadata"]
    lines.extend(
        [
            "## Metadata",
            "",
            f"- Provider: {metadata.get('provider', 'unknown')}",
            f"- Model: {metadata.get('model', 'unknown')}",
            f"- Evaluations: {metadata.get('evaluations', 0)}",
            f"- Max iterations: {metadata.get('max_iterations', 0)}",
            f"- Research calls: {metadata.get('research_calls', 0)}",
            f"- Max research calls: {metadata.get('max_research_calls', 0)}",
            f"- Runs: {metadata.get('runs', 1)}",
        ]
    )
    if metadata.get("resumed_from"):
        lines.append(f"- Resumed from: {metadata['resumed_from']}")
    progress = metadata.get("progress")
    if progress:
        completed = ", ".join(progress.get("completed_stages") or []) or "none"
        lines.extend(
            [
                f"- Stage: {progress.get('stage', 'unknown')}",
                f"- Running sub-agent: {progress.get('running_stage') or 'none'}",
                f"- Completed stages: {completed}",
                f"- Subtasks so far: {progress.get('subtask_count', 0)}",
            ]
        )
        for title in progress.get("subtask_titles") or []:
            lines.append(f"  - {title}")
    lines.append("")
    return "\n".join(lines)


def resolve_output_format(output_path: Path, requested: str) -> str:
    """Resolve the output format, inferring it from the file extension if needed."""
    if requested != "auto":
        return requested
    if output_path.suffix.lower() in {".md", ".markdown"}:
        return "markdown"
    return "json"


def unique_output_path(
    document: dict[str, Any],
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    requested_format: str = "json",
) -> tuple[Path, str]:
    """Return an unused ``output-<hash>.<ext>`` path and its format.

    The hash covers the issue, the generation timestamp, and a random nonce, so
    running the agent twice never reuses (and therefore never overwrites) an
    existing artifact. ``requested_format="auto"`` falls back to JSON because
    there is no user-supplied file extension to inspect.
    """
    output_format = "json" if requested_format == "auto" else requested_format
    if output_format not in _FORMAT_SUFFIXES:
        raise ValueError(f"Unsupported output format: {requested_format!r}")

    suffix = _FORMAT_SUFFIXES[output_format]
    directory = Path(output_dir).expanduser()
    issue = str(document.get("issue", ""))
    generated_at = str(document.get("generated_at", ""))

    for attempt in range(100):
        seed = "\x00".join(
            (issue, generated_at, uuid.uuid4().hex, str(attempt))
        )
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[
            :OUTPUT_HASH_LENGTH
        ]
        candidate = directory / f"{OUTPUT_PREFIX}-{digest}{suffix}"
        if not candidate.exists():
            return candidate, output_format
    raise RuntimeError("Could not find an unused output file name.")


def write_output(
    document: dict[str, Any], output_path: Path, requested_format: str = "auto"
) -> tuple[Path, str]:
    """Atomically write the final artifact and return its path and format."""
    output_path = output_path.expanduser().resolve()
    output_format = resolve_output_format(output_path, requested_format)

    if output_format == "markdown":
        text = render_markdown(document)
        if not text.endswith("\n"):
            text += "\n"
    else:
        text = json.dumps(document, indent=2, ensure_ascii=False) + "\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    try:
        temporary_path.write_text(text, encoding="utf-8")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path, output_format

