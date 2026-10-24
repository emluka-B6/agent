"""Pydantic schemas used for structured LLM output and the workflow state."""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field


class Source(BaseModel):
    """A single web search result used as research evidence."""

    title: str = Field(default="", description="Title of the web page, if available.")
    url: str = Field(default="", description="URL of the web page.")
    snippet: str = Field(
        default="", description="Text snippet returned by the search engine."
    )


class ResearchSummary(BaseModel):
    """Condensed research that is relevant for resolving the issue."""

    key_facts: list[str] = Field(
        default_factory=list,
        description="Verified facts from the supplied search snippets.",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Domain, technical, or compatibility constraints that the plan must respect.",
    )
    recommended_approach: str = Field(
        description="A concise recommended approach supported by the research."
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Things that are not verified by the sources and must be treated as assumptions.",
    )


class SubTask(BaseModel):
    """One schedulable unit of work for an execution LLM."""

    id: str = Field(description="Stable identifier, for example T1 or T2.")
    title: str = Field(description="Short imperative title of the subtask.")
    description: str = Field(
        description="What must be changed or produced, including relevant context and scope."
    )
    implementation_prompt: str = Field(
        description=(
            "A self-contained prompt that can be sent directly to an execution LLM "
            "(a coding, analysis, or operations assistant). It must include the goal, "
            "constraints, artifacts, and acceptance criteria."
        )
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="IDs of subtasks that must be completed before this one.",
    )
    files: list[str] = Field(
        default_factory=list,
        description=(
            "Files, documents, configurations, or other artifacts expected to be "
            "created or modified. May be empty when the domain has no concrete paths."
        ),
    )
    acceptance_criteria: list[str] = Field(
        min_length=1,
        description="Observable conditions that prove the subtask is complete.",
    )
    verification: str = Field(
        description=(
            "Concrete commands, tests, calculations, or manual checks that verify "
            "the subtask."
        )
    )
    complexity: Literal["small", "medium", "large"] = Field(
        default="medium", description="Estimated size of the subtask."
    )
    risks: list[str] = Field(
        default_factory=list, description="Known risks or edge cases for this subtask."
    )


class ImplementationPlan(BaseModel):
    """The complete plan produced by the planner and reviewed by the critic."""

    summary: str = Field(description="Short summary of the proposed strategy.")
    assumptions: list[str] = Field(
        default_factory=list, description="Assumptions made while preparing the plan."
    )
    subtasks: list[SubTask] = Field(
        min_length=1, description="Ordered list of schedulable implementation subtasks."
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Questions that still need a human decision, if any.",
    )
    definition_of_done: list[str] = Field(
        min_length=1,
        description="High-level conditions that define a complete solution.",
    )
    research_requests: list[str] = Field(
        default_factory=list,
        exclude=True,
        description=(
            "Focused web search queries the planner needs answered before the "
            "plan can be finalized. Leave empty when the existing research is "
            "sufficient. This is workflow control data and is not written to the "
            "output artifact."
        ),
    )


class Critique(BaseModel):
    """Structured review of an implementation plan."""

    approved: bool = Field(
        description=(
            "True only when the plan can be handed to an execution LLM without "
            "clarification."
        )
    )
    summary: str = Field(description="Short assessment of the plan.")
    blocking_issues: list[str] = Field(
        default_factory=list,
        description="Concrete problems that must be fixed before the plan is approved.",
    )
    suggestions: list[str] = Field(
        default_factory=list, description="Non-blocking improvements."
    )


class WorkflowState(TypedDict, total=False):
    """State passed between LangGraph nodes."""

    issue: str
    domain: str
    research: ResearchSummary | None
    sources: list[Source]
    research_error: str | None
    research_calls: int
    research_requests: list[str]
    plan: ImplementationPlan | None
    critique: Critique | None
    evaluations: int
    approved: bool
