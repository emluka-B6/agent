"""LangGraph node implementations: research, planner, and critic."""

from __future__ import annotations

import sys
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from issue_planner.config import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_RESEARCH_CALLS,
    MAX_FOLLOW_UP_QUERIES,
)
from issue_planner.llm import as_model, invoke_structured, invoke_with_retry
from issue_planner.models import (
    Critique,
    ImplementationPlan,
    PlanRevision,
    ResearchSummary,
    Source,
    WorkflowState,
)
from issue_planner.plan_editor import RemovalError, apply_revision
from issue_planner.plan_repair import normalize_plan, plan_warnings
from issue_planner.prompts import prompts_for
from issue_planner.research import (
    extract_sources,
    format_research,
    format_research_questions,
    format_research_status,
    format_source_list,
    format_sources_for_prompt,
    merge_sources,
)


def _search_web(
    search_client: Any,
    query: str,
    max_search_results: int,
    max_retries: int,
) -> tuple[list[Source], str | None]:
    """Run one web search and normalise its result into sources plus an error."""
    try:
        raw_results = invoke_with_retry(
            search_client, query, "web search", max_retries
        )
    except Exception as exc:  # noqa: BLE001 - research is best-effort
        message = f"Web search failed: {exc}"
        print(f"warning: {message}", file=sys.stderr)
        return [], message

    if isinstance(raw_results, dict) and raw_results.get("error"):
        return [], f"Web search failed: {raw_results['error']}"

    sources = extract_sources(raw_results, max_search_results)
    if not sources:
        return [], "Web search returned no usable results."
    return sources, None


def research_node(
    state: WorkflowState,
    model: BaseChatModel,
    provider: str,
    search_client: Any,
    max_search_results: int,
    skip_research: bool,
    max_retries: int,
) -> dict[str, Any]:
    """Collect web evidence and condense it into a ResearchSummary.

    The first invocation searches the issue. Later invocations are requested by
    the planner through ``research_requests``; each request becomes a focused
    search and the new evidence is merged with the sources and summary that
    were collected earlier.
    """
    issue = state["issue"]
    previous_research = state.get("research")
    existing_sources = list(state.get("sources") or [])
    requests = [
        str(request).strip()
        for request in (state.get("research_requests") or [])
        if str(request).strip()
    ][:MAX_FOLLOW_UP_QUERIES]

    new_sources: list[Source] = []
    research_error: str | None = None

    if skip_research:
        research_error = "Web research was skipped with --skip-research."
    elif search_client is None:
        research_error = "Web research is unavailable because no search client is configured."
    else:
        queries = requests or [issue]
        errors: list[str] = []
        for query in queries:
            found, error = _search_web(
                search_client, query, max_search_results, max_retries
            )
            new_sources.extend(found)
            if error:
                errors.append(
                    error if len(queries) == 1 else f"'{query}': {error}"
                )
        new_sources = merge_sources([], new_sources)
        if errors:
            research_error = "; ".join(errors)

    sources = merge_sources(existing_sources, new_sources)
    prompts = prompts_for(state.get("domain"))
    messages = prompts.research.format_messages(
        issue=issue,
        research_questions=format_research_questions(requests),
        previous_research=format_research(previous_research),
        research_data=format_sources_for_prompt(new_sources),
        research_error=research_error or "none",
    )
    raw_summary = invoke_structured(
        model,
        ResearchSummary,
        provider,
        messages,
        "research synthesis",
        max_retries,
    )
    summary = as_model(ResearchSummary, raw_summary)

    return {
        "research": summary,
        "sources": sources,
        "research_error": research_error,
        "research_calls": state.get("research_calls", 0) + 1,
        "research_requests": [],
    }


def _review_context(evaluations: int, max_iterations: int) -> str:
    """Describe where a review sits in the revision budget."""
    remaining = max(0, max_iterations - evaluations)
    if remaining:
        return (
            f"review {evaluations} of {max_iterations}; "
            f"{remaining} revision(s) remain after this one"
        )
    return (
        f"review {evaluations} of {max_iterations}; this is the final review, "
        "so block only on issues that make the plan unexecutable"
    )


def planner_node(
    state: WorkflowState,
    model: BaseChatModel,
    provider: str,
    max_retries: int,
    max_research_calls: int = DEFAULT_MAX_RESEARCH_CALLS,
) -> dict[str, Any]:
    """Create a new plan or revise the existing plan using the critic feedback.

    The first call generates a complete :class:`ImplementationPlan`. Every later
    call (after follow-up research or a critic review) requests a
    :class:`PlanRevision` patch instead: unchanged subtasks keep their ids and are
    not regenerated, and the response stays far below the provider's
    output-token budget.
    """
    current_plan = state.get("plan")
    critique = state.get("critique")

    prompts = prompts_for(state.get("domain"))
    template = (
        prompts.planner if current_plan is None else prompts.planner_revision
    )
    messages = template.format_messages(
        issue=state["issue"],
        research=format_research(state.get("research")),
        sources=format_source_list(state.get("sources") or []),
        research_status=format_research_status(
            state.get("research_error"),
            state.get("research_calls", 0),
            max_research_calls,
        ),
        current_plan=(
            current_plan.model_dump_json(indent=2)
            if current_plan is not None
            else "(none; create the first plan)"
        ),
        critique=(critique.model_dump_json(indent=2) if critique else "(none)"),
    )

    if current_plan is None:
        raw_plan = invoke_structured(
            model,
            ImplementationPlan,
            provider,
            messages,
            "plan generation",
            max_retries,
        )
        plan = normalize_plan(as_model(ImplementationPlan, raw_plan))
    else:
        raw_revision = invoke_structured(
            model,
            PlanRevision,
            provider,
            messages,
            "plan revision",
            max_retries,
        )
        revision = as_model(PlanRevision, raw_revision)
        if revision.is_empty():
            print(
                "warning: the planner returned an empty revision; keeping the "
                "previous plan.",
                file=sys.stderr,
            )
        try:
            plan = apply_revision(current_plan, revision)
        except RemovalError as exc:
            print(
                f"warning: could not apply the plan revision ({exc}); keeping the "
                "previous plan.",
                file=sys.stderr,
            )
            # Drop the already-consumed queries so the workflow does not search
            # for them a second time.
            plan = current_plan.model_copy(update={"research_requests": []})

    for warning in plan_warnings(plan):
        print(f"warning: {warning}", file=sys.stderr)

    requests: list[str] = []
    for request in plan.research_requests:
        text = str(request).strip()
        if text and text not in requests:
            requests.append(text)

    return {
        "plan": plan,
        "research_requests": requests[:MAX_FOLLOW_UP_QUERIES],
        "approved": False,
    }


def critic_node(
    state: WorkflowState,
    model: BaseChatModel,
    provider: str,
    max_retries: int,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_research_calls: int = DEFAULT_MAX_RESEARCH_CALLS,
) -> dict[str, Any]:
    """Review the plan and record whether it is ready to be scheduled.

    The review receives the research limitation and remaining budget, a compact
    source list, its position in the revision budget, and its own previous
    review so that it can confirm earlier blocking issues are resolved.
    """
    plan = state.get("plan")
    if plan is None:
        raise RuntimeError("Cannot review a plan because none was generated.")

    previous = state.get("critique")
    evaluations = state.get("evaluations", 0) + 1

    prompts = prompts_for(state.get("domain"))
    messages = prompts.critic.format_messages(
        issue=state["issue"],
        review_context=_review_context(evaluations, max_iterations),
        research=format_research(state.get("research")),
        sources=format_source_list(state.get("sources") or []),
        research_status=format_research_status(
            state.get("research_error"),
            state.get("research_calls", 0),
            max_research_calls,
        ),
        plan=plan.model_dump_json(indent=2),
        previous_critique=(
            previous.model_dump_json(indent=2)
            if previous is not None
            else "(none; this is the first review)"
        ),
    )
    raw_critique = invoke_structured(
        model,
        Critique,
        provider,
        messages,
        "plan review",
        max_retries,
    )
    critique = as_model(Critique, raw_critique)

    if critique.blocking_issues and critique.approved:
        critique = critique.model_copy(update={"approved": False})

    return {
        "critique": critique,
        "evaluations": evaluations,
        "approved": critique.approved,
    }


def route_after_planner(
    state: WorkflowState,
    max_research_calls: int,
    research_available: bool = True,
) -> str:
    """Route to follow-up research while the planner asks and budget remains."""
    if not research_available:
        return "critic"
    requests = [
        request
        for request in (state.get("research_requests") or [])
        if str(request).strip()
    ]
    if requests and state.get("research_calls", 0) < max_research_calls:
        return "research"
    return "critic"


def route_after_critic(state: WorkflowState, max_iterations: int) -> str:
    """Stop when the plan is approved or the revision budget is exhausted."""
    critique = state.get("critique")
    if critique is not None and critique.approved:
        return "finish"
    if state.get("evaluations", 0) >= max_iterations:
        return "finish"
    return "revise"
