"""LangGraph assembly and the high-level workflow runner."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from issue_planner.config import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_RESEARCH_CALLS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_SEARCH_RESULTS,
)
from issue_planner.domains import DEFAULT_DOMAIN, get_domain
from issue_planner.models import WorkflowState
from issue_planner.nodes import (
    critic_node,
    planner_node,
    research_node,
    route_after_critic,
    route_after_planner,
)
from issue_planner.progress import ProgressReporter


def build_graph(
    model: BaseChatModel,
    provider: str = "openai",
    search_client: Any = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_search_results: int = DEFAULT_MAX_SEARCH_RESULTS,
    skip_research: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
    reporter: ProgressReporter | None = None,
    max_research_calls: int = DEFAULT_MAX_RESEARCH_CALLS,
) -> Any:
    """Build and compile the research -> planner -> critic workflow.

    The planner may route back to research when it asks for targeted follow-up
    information while revising the plan. At most ``max_research_calls`` research
    invocations (including the initial one) are allowed per run.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")
    if max_research_calls < 1:
        raise ValueError("max_research_calls must be at least 1.")
    if max_search_results < 1:
        raise ValueError("max_search_results must be at least 1.")

    graph = StateGraph(WorkflowState)

    def research(state: WorkflowState) -> dict[str, Any]:
        return research_node(
            state,
            model=model,
            provider=provider,
            search_client=search_client,
            max_search_results=max_search_results,
            skip_research=skip_research,
            max_retries=max_retries,
        )

    def planner(state: WorkflowState) -> dict[str, Any]:
        return planner_node(
            state,
            model=model,
            provider=provider,
            max_retries=max_retries,
            max_research_calls=max_research_calls,
        )

    def critic(state: WorkflowState) -> dict[str, Any]:
        return critic_node(
            state,
            model=model,
            provider=provider,
            max_retries=max_retries,
            max_iterations=max_iterations,
            max_research_calls=max_research_calls,
        )

    def tracked(
        name: str, node: Callable[[WorkflowState], dict[str, Any]]
    ) -> Callable[[WorkflowState], dict[str, Any]]:
        """Wrap a node so the optional reporter sees start/finish/failure."""

        def run(state: WorkflowState) -> dict[str, Any]:
            if reporter is not None:
                reporter.node_started(name, state)
            try:
                result = node(state)
            except Exception as exc:
                if reporter is not None:
                    reporter.node_failed(name, state, exc)
                raise
            if reporter is not None:
                reporter.node_finished(
                    name, cast(WorkflowState, {**state, **result})
                )
            return result

        return run

    graph.add_node("research", tracked("research", research))
    graph.add_node("planner", tracked("planner", planner))
    graph.add_node("critic", tracked("critic", critic))
    research_available = search_client is not None and not skip_research
    graph.add_edge(START, "research")
    graph.add_edge("research", "planner")
    graph.add_conditional_edges(
        "planner",
        lambda state: route_after_planner(
            state, max_research_calls, research_available
        ),
        {"research": "research", "critic": "critic"},
    )
    graph.add_conditional_edges(
        "critic",
        lambda state: route_after_critic(state, max_iterations),
        {"revise": "planner", "finish": END},
    )
    return graph.compile()


def run_agent(
    issue: str,
    model: BaseChatModel,
    provider: str = "openai",
    search_client: Any = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_search_results: int = DEFAULT_MAX_SEARCH_RESULTS,
    skip_research: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
    domain: str = DEFAULT_DOMAIN,
    reporter: ProgressReporter | None = None,
    max_research_calls: int = DEFAULT_MAX_RESEARCH_CALLS,
) -> WorkflowState:
    """Run the workflow and return its final state.

    ``reporter`` is optional. When supplied, it receives a callback for every
    workflow node and can persist partial artifacts while the run is in
    progress.
    """
    profile = get_domain(domain)
    issue = issue.strip()
    if not issue:
        raise ValueError("The issue must not be empty.")

    graph = build_graph(
        model=model,
        provider=provider,
        search_client=search_client,
        max_iterations=max_iterations,
        max_research_calls=max_research_calls,
        max_search_results=max_search_results,
        skip_research=skip_research,
        max_retries=max_retries,
        reporter=reporter,
    )
    initial_state: WorkflowState = {
        "issue": issue,
        "domain": profile.key,
        "research": None,
        "sources": [],
        "research_error": None,
        "research_calls": 0,
        "research_requests": [],
        "plan": None,
        "critique": None,
        "evaluations": 0,
        "approved": False,
    }
    if reporter is not None:
        reporter.workflow_started(issue)
    try:
        final_state = graph.invoke(initial_state)
    except Exception as exc:
        if reporter is not None:
            reporter.workflow_failed(exc)
        raise
    if reporter is not None:
        reporter.workflow_finished(final_state)
    return final_state
