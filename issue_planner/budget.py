"""Iteration and research budgets that survive across chained runs.

One run of ``agent.py`` may end without an approval because its revision budget
was exhausted (``status=max_iterations_reached``). Continuing such a run means
adding a *new* budget on top of the reviews that already happened, so the critic
loop does not stop before it starts.

:class:`RunBudget` keeps both views:

- the cumulative caps (what the artifact records as ``max_iterations`` and
  ``max_research_calls``), and
- the per-run caps (how much this invocation may add), together with the number
  of chained runs and where the state was picked up from.

The workflow itself only needs the cumulative caps; the per-run numbers and the
run counter exist so a user can see how many iterations the whole analysis cost
before deciding on the next ``--max-iterations`` value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from issue_planner.resume import ResumeState


@dataclass(frozen=True)
class RunBudget:
    """Cumulative iteration and research budget for one invocation."""

    max_iterations: int
    max_research_calls: int
    iterations_before: int = 0
    research_calls_before: int = 0
    runs: int = 1
    resumed_from: str | None = None
    previous_status: str | None = None

    @classmethod
    def fresh(cls, max_iterations: int, max_research_calls: int) -> RunBudget:
        """Return the budget of a run that starts from scratch."""
        return cls(
            max_iterations=max_iterations,
            max_research_calls=max_research_calls,
        )

    @classmethod
    def continued(
        cls,
        resume: ResumeState,
        max_iterations: int,
        max_research_calls: int,
    ) -> RunBudget:
        """Extend the budget a previous artifact already spent.

        ``max_iterations`` and ``max_research_calls`` are the amounts this run
        may add, so the cumulative caps become ``spent + requested``. That is
        what lets ``--max-iterations 2`` continue a run that already used two
        reviews instead of finishing immediately.
        """
        return cls(
            max_iterations=resume.evaluations + max_iterations,
            max_research_calls=resume.research_calls + max_research_calls,
            iterations_before=resume.evaluations,
            research_calls_before=resume.research_calls,
            runs=resume.runs + 1,
            resumed_from=str(resume.path),
            previous_status=resume.status,
        )

    @property
    def continued_run(self) -> bool:
        """True when this invocation continues an earlier artifact."""
        return self.runs > 1

    @property
    def iterations_available(self) -> int:
        """Number of critic reviews this invocation may still spend."""
        return max(0, self.max_iterations - self.iterations_before)

    @property
    def research_calls_available(self) -> int:
        """Number of research sub-agent calls this invocation may still spend."""
        return max(0, self.max_research_calls - self.research_calls_before)

    def to_metadata(self, evaluations: int, research_calls: int) -> dict[str, Any]:
        """Return the ``metadata`` counters for the artifact.

        The cumulative values keep the same names as before, so a reader that
        only knows ``evaluations``/``max_iterations`` keeps working. The
        ``*_this_run`` values and ``runs`` describe the current invocation, and
        ``resumed_from``/``previous_status`` point at the artifact it continues.
        """
        return {
            "evaluations": evaluations,
            "max_iterations": self.max_iterations,
            "iterations_this_run": max(0, evaluations - self.iterations_before),
            "max_iterations_this_run": self.iterations_available,
            "research_calls": research_calls,
            "max_research_calls": self.max_research_calls,
            "research_calls_this_run": max(
                0, research_calls - self.research_calls_before
            ),
            "max_research_calls_this_run": self.research_calls_available,
            "runs": self.runs,
            "resumed_from": self.resumed_from,
            "previous_status": self.previous_status,
        }
