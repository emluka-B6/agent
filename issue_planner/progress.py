"""Live console progress and partial artifact persistence.

Long planning runs spend most of their time inside a single LLM call. This
module reports which workflow node ("sub-agent") is running, how many subtasks
have been produced so far, and their titles. It also rewrites the final output
path after every step, so an interrupted run still leaves a partial artifact on
disk instead of nothing.

All console output goes to stderr, keeping stdout reserved for the final file
path printed by ``agent.py``.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from issue_planner.budget import RunBudget
from issue_planner.config import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_RESEARCH_CALLS,
)
from issue_planner.domains import DEFAULT_DOMAIN
from issue_planner.models import WorkflowState
from issue_planner.output import (
    build_progress_document,
    document_status,
    write_output,
)

#: Display order and one-based index of each workflow node.
NODE_STEPS: dict[str, int] = {"research": 1, "planner": 2, "critic": 3}

#: Human-readable name used for each workflow node in console messages.
NODE_LABELS: dict[str, str] = {
    "research": "research sub-agent",
    "planner": "planner sub-agent",
    "critic": "critic sub-agent",
}

TOTAL_STEPS = len(NODE_STEPS)


def _label(node: str) -> str:
    return NODE_LABELS.get(node, f"{node} sub-agent")


def _prefix(node: str) -> str:
    step = NODE_STEPS.get(node)
    return f"[{step}/{TOTAL_STEPS}]" if step else "[agent]"


def _one_line(text: str, limit: int = 120) -> str:
    """Collapse whitespace and shorten ``text`` for a single console line."""
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _duration(seconds: float) -> str:
    if seconds >= 60:
        minutes, remainder = divmod(seconds, 60)
        return f"{int(minutes)}m {remainder:.0f}s"
    return f"{seconds:.1f}s"


def _plan_data(state: WorkflowState) -> dict[str, Any]:
    """Return the current plan as a mapping, tolerating missing plan data."""
    plan = state.get("plan")
    if plan is None:
        return {}
    if isinstance(plan, dict):
        return plan
    dump = getattr(plan, "model_dump", None)
    return dump() if callable(dump) else {}


def _subtasks(state: WorkflowState) -> list[dict[str, Any]]:
    return list(_plan_data(state).get("subtasks") or [])


def _critique_data(state: WorkflowState) -> dict[str, Any]:
    critique = state.get("critique")
    if critique is None:
        return {}
    if isinstance(critique, dict):
        return critique
    dump = getattr(critique, "model_dump", None)
    return dump() if callable(dump) else {}


@dataclass
class ProgressReporter:
    """Print workflow progress and persist partial artifacts.

    ``output_path`` is optional. When it is set, the reporter writes a partial
    artifact after every workflow step. ``stream`` defaults to ``sys.stderr``
    and is resolved at call time so test capture works as expected.

    ``budget`` carries the cumulative iteration/research budget. When it is
    given it supersedes ``max_iterations`` and ``max_research_calls``, which
    keeps the console and the artifacts consistent for a resumed run.
    """

    output_path: Path | str | None = None
    output_format: str = "json"
    provider: str = "openai"
    model_name: str = "unknown"
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    domain: str = DEFAULT_DOMAIN
    generated_at: str | None = None
    stream: TextIO | None = None
    max_research_calls: int = DEFAULT_MAX_RESEARCH_CALLS
    research_available: bool = True
    budget: RunBudget | None = None

    _started_at: float = field(
        default_factory=time.monotonic, init=False, repr=False
    )
    _stage_started_at: float | None = field(default=None, init=False, repr=False)
    _running_stage: str | None = field(default=None, init=False, repr=False)
    _completed_stages: list[str] = field(
        default_factory=list, init=False, repr=False
    )
    _partial_announced: bool = field(default=False, init=False, repr=False)
    _partial_written: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.budget is not None:
            self.max_iterations = self.budget.max_iterations
            self.max_research_calls = self.budget.max_research_calls

    @property
    def run_budget(self) -> RunBudget:
        """The cumulative budget, defaulting to a single-run budget."""
        return self.budget or RunBudget.fresh(
            self.max_iterations, self.max_research_calls
        )

    @property
    def partial_written(self) -> bool:
        """True once at least one partial artifact was written successfully."""
        return self._partial_written

    # -- console helpers ---------------------------------------------------
    def _emit(self, message: str) -> None:
        print(message, file=self.stream or sys.stderr, flush=True)

    def _save_partial(self, state: WorkflowState, stage: str) -> None:
        """Atomically write a partial artifact, never failing the workflow."""
        if self.output_path is None:
            return
        output_path = Path(self.output_path).expanduser()
        document = build_progress_document(
            state,
            provider=self.provider,
            model_name=self.model_name,
            max_iterations=self.max_iterations,
            stage=stage,
            completed_stages=self._completed_stages,
            running_stage=self._running_stage,
            generated_at=self.generated_at,
            max_research_calls=self.max_research_calls,
            budget=self.run_budget,
        )
        try:
            write_output(document, output_path, self.output_format)
        except Exception as exc:  # noqa: BLE001 - partial writes are best-effort
            self._emit(
                f"[agent] warning: could not write the partial result to "
                f"{output_path}: {exc}"
            )
            return
        self._partial_written = True
        if not self._partial_announced:
            self._partial_announced = True
            self._emit(
                f"[agent] partial result file: {output_path} "
                "(updated after every step)"
            )

    # -- lifecycle ---------------------------------------------------------
    def workflow_started(self, issue: str) -> None:
        """Announce the run and create the first partial artifact."""
        self._started_at = time.monotonic()
        self.generated_at = self.generated_at or datetime.now(
            timezone.utc
        ).isoformat()
        budget = self.run_budget
        workflow = " -> ".join(_label(node) for node in NODE_STEPS)
        run_suffix = f" | run {budget.runs}" if budget.continued_run else ""
        self._emit(
            f"[agent] planning started | domain={self.domain} | "
            f"workflow={workflow} | max iterations={self.max_iterations} | "
            f"max research calls={self.max_research_calls}{run_suffix}"
        )
        if budget.continued_run:
            self._emit(
                f"[agent] continuing {budget.resumed_from} "
                f"(previous status={budget.previous_status}) | "
                f"already used: iterations={budget.iterations_before}, "
                f"research calls={budget.research_calls_before} | "
                f"this run may add {budget.iterations_available} iteration(s) "
                f"and {budget.research_calls_available} research call(s)"
            )
        self._emit(f"[agent] issue: {_one_line(issue)}")
        initial_state: WorkflowState = {"issue": issue, "domain": self.domain}
        if budget.continued_run:
            # Keep the partial artifact honest about what earlier runs spent.
            initial_state["evaluations"] = budget.iterations_before
            initial_state["research_calls"] = budget.research_calls_before
        self._save_partial(initial_state, "starting")

    def node_started(self, node: str, state: WorkflowState) -> None:
        """Report that a workflow node (sub-agent) started running."""
        self._running_stage = node
        self._stage_started_at = time.monotonic()
        self._emit(
            f"{_prefix(node)} {_label(node)}: running..."
            f"{self._start_context(node, state)}"
        )
        self._save_partial(state, node)

    def node_finished(self, node: str, state: WorkflowState) -> None:
        """Report the result of one workflow node and refresh the artifact."""
        elapsed = (
            time.monotonic() - self._stage_started_at
            if self._stage_started_at is not None
            else 0.0
        )
        self._emit(
            f"{_prefix(node)} {_label(node)}: done in {_duration(elapsed)} | "
            f"{self._result_summary(node, state)}"
        )

        if node == "planner":
            for task in _subtasks(state):
                task_id = task.get("id", "?")
                title = task.get("title", "(untitled)")
                self._emit(f"    - {task_id}: {title}")

        if node not in self._completed_stages:
            self._completed_stages.append(node)
        self._running_stage = None
        self._stage_started_at = None
        self._save_partial(state, f"{node}_done")
        self._announce_next_step(node, state)

    def node_failed(
        self, node: str, state: WorkflowState, error: BaseException
    ) -> None:
        """Report a failing workflow node and persist the state reached so far."""
        elapsed = (
            time.monotonic() - self._stage_started_at
            if self._stage_started_at is not None
            else 0.0
        )
        self._emit(
            f"{_prefix(node)} {_label(node)}: failed after "
            f"{_duration(elapsed)} | {_one_line(str(error), 200)}"
        )
        self._running_stage = None
        self._stage_started_at = None
        self._save_partial(state, f"{node}_failed")

    def workflow_finished(self, state: WorkflowState) -> None:
        """Print the final one-line summary for a completed workflow.

        The line reports the cumulative counters, so a user who chained several
        runs sees what the whole analysis cost before deciding on a new
        ``--max-iterations`` value for the next continuation.
        """
        elapsed = time.monotonic() - self._started_at
        tasks = _subtasks(state)
        status = document_status(state, self.max_iterations)
        self._emit(
            f"[agent] planning finished in {_duration(elapsed)} | "
            f"status={status} | "
            f"approved={'yes' if state.get('approved') else 'no'} | "
            f"subtasks={len(tasks)} | "
            f"iterations={state.get('evaluations', 0)}/{self.max_iterations} | "
            f"research calls={state.get('research_calls', 0)}/"
            f"{self.max_research_calls} | "
            f"runs={self.run_budget.runs}"
        )

    def workflow_failed(self, error: BaseException) -> None:
        """Print a failure summary and point at the retained partial artifact."""
        elapsed = time.monotonic() - self._started_at
        self._emit(
            f"[agent] planning failed after {_duration(elapsed)}: "
            f"{_one_line(str(error), 200)}"
        )
        if self._partial_written and self.output_path is not None:
            self._emit(f"[agent] partial result kept in {self.output_path}")

    # -- message fragments -------------------------------------------------
    def _start_context(self, node: str, state: WorkflowState) -> str:
        if node == "research":
            questions = [
                request
                for request in (state.get("research_requests") or [])
                if str(request).strip()
            ]
            if not questions:
                return ""
            call = state.get("research_calls", 0) + 1
            return (
                f" | follow-up research ({call}/{self.max_research_calls})"
                f" | questions={len(questions)}"
            )
        if node == "planner":
            review = state.get("evaluations", 0) + 1
            parts = [f"review {review}/{self.max_iterations}"]
            tasks = _subtasks(state)
            if tasks:
                parts.append(f"revising {len(tasks)} subtasks")
            return " | " + " | ".join(parts)
        if node == "critic":
            return f" | reviewing {len(_subtasks(state))} subtasks"
        return ""

    def _result_summary(self, node: str, state: WorkflowState) -> str:
        if node == "research":
            calls = state.get("research_calls", 0)
            parts = [f"sources={len(state.get('sources') or [])}"]
            if calls > 1:
                parts.append(
                    f"research calls={calls}/{self.max_research_calls}"
                )
            research_error = state.get("research_error")
            if research_error:
                parts.append(f"note: {_one_line(str(research_error), 100)}")
            return " | ".join(parts)
        if node == "planner":
            return f"subtasks={len(_subtasks(state))}"
        if node == "critic":
            blocking = _critique_data(state).get("blocking_issues") or []
            approved = "yes" if state.get("approved") else "no"
            return f"approved={approved} | blocking issues={len(blocking)}"
        return "done"

    def _announce_next_step(self, node: str, state: WorkflowState) -> None:
        if node == "planner":
            requests = [
                request
                for request in (state.get("research_requests") or [])
                if str(request).strip()
            ]
            if not requests:
                return
            if not self.research_available:
                self._emit(
                    "[agent] next sub-agent: critic "
                    "(follow-up research unavailable)"
                )
                return
            calls = state.get("research_calls", 0)
            if calls < self.max_research_calls:
                self._emit(
                    f"[agent] next sub-agent: research follow-up "
                    f"(research {calls + 1}/{self.max_research_calls}) | "
                    f"questions={len(requests)}"
                )
            else:
                self._emit(
                    f"[agent] research budget exhausted "
                    f"({calls}/{self.max_research_calls}); continuing to critic"
                )
            return
        if node != "critic" or state.get("approved"):
            return
        evaluations = state.get("evaluations", 0)
        if evaluations < self.max_iterations:
            self._emit(
                f"[agent] next sub-agent: planner revision "
                f"(review {evaluations + 1}/{self.max_iterations})"
            )
        else:
            self._emit(
                f"[agent] revision budget exhausted "
                f"({evaluations}/{self.max_iterations}); finishing"
            )

