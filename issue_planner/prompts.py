"""Shared prompt templates used by the research, planner, and critic nodes.

The templates below are domain-agnostic. :mod:`issue_planner.domains` supplies
the values that specialise them for software engineering, AI, finance, and
cybersecurity, so all domains keep the same workflow and output contract while
receiving different roles, focus areas, verification guidance, and safety
rules.

The module-level ``RESEARCH_PROMPT``, ``PLANNER_PROMPT``, and ``CRITIC_PROMPT``
objects are the software defaults and remain available for backwards
compatibility. Use :func:`prompts_for` to obtain the set for another domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from string import Template

from langchain_core.prompts import ChatPromptTemplate

from issue_planner.domains import DEFAULT_DOMAIN, get_domain, prompt_values

RESEARCH_SYSTEM_TEMPLATE = Template(
    """You are a $research_role acting as a research agent.
Extract $research_objective from the supplied web search snippets for the given issue.

Focus areas:
$research_focus

Rules:
- Use only the supplied snippets as external evidence. Do not invent URLs, versions, or APIs.
- When follow-up research questions and a previous research summary are supplied, answer the questions from the new snippets and merge the result with the previous summary instead of discarding it.
- Put verified evidence in key_facts and constraints.
- Put anything that is not supported by the snippets in assumptions.
- If snippets are missing or irrelevant, state that in assumptions and rely only on well-known $practice_label.
- Be concise and $research_style.
- Write in the same language as the issue.$research_rules"""
)

PLANNER_SYSTEM_TEMPLATE = Template(
    """You are a $planner_role creating $planner_objective.
Break the issue into small, ordered, independently schedulable $work_unit.

Focus areas:
$planner_focus

Rules:
- Use stable IDs such as T1, T2, T3.
- Every subtask must have a self-contained implementation_prompt with the goal, context, constraints, expected $artifact_noun, and acceptance criteria.
- Dependencies must reference existing subtask IDs and must not form cycles.
- List only $artifact_scope. Mark uncertain items as suggestions.
- Include concrete acceptance criteria and a verification step for every subtask. $verification_hint
- Keep the scope aligned with the issue; do not add unrelated work.
- Never include secrets, credentials, or private personal data.
- When a critique is provided, address every blocking issue and preserve the correct parts of the previous plan.
- When a blocking issue needs external or current information that is missing from the research summary, put up to three focused web search queries in research_requests. Leave research_requests empty when the existing research is sufficient; research calls are limited.
- Write in the same language as the issue.$planner_rules"""
)

CRITIC_SYSTEM_TEMPLATE = Template(
    """You are $critic_role.
Evaluate whether the plan can be handed to $executor_label without further clarification.

Check:
- completeness relative to the issue;
- atomicity and schedulability of every subtask;
- correct dependencies and execution order;
- observable acceptance criteria and concrete verification;
- realistic $artifact_noun and scope;
- self-contained implementation prompts.$critic_checks

Set approved to true only if there are no blocking issues.
Blocking issues must be concrete and actionable.
If a blocking issue depends on facts that are missing from the research summary, say so explicitly so the planner can request more research.
Write in the same language as the issue."""
)


@dataclass(frozen=True)
class PromptSet:
    """The rendered system prompts and chat templates for one domain."""

    research_system_prompt: str
    planner_system_prompt: str
    critic_system_prompt: str
    research: ChatPromptTemplate
    planner: ChatPromptTemplate
    critic: ChatPromptTemplate


@lru_cache(maxsize=None)
def build_prompts(domain: str = DEFAULT_DOMAIN) -> PromptSet:
    """Render the shared templates for ``domain`` (cached per domain key)."""
    values = prompt_values(get_domain(domain))
    research_system_prompt = RESEARCH_SYSTEM_TEMPLATE.substitute(**values)
    planner_system_prompt = PLANNER_SYSTEM_TEMPLATE.substitute(**values)
    critic_system_prompt = CRITIC_SYSTEM_TEMPLATE.substitute(**values)

    research = ChatPromptTemplate.from_messages(
        [
            ("system", research_system_prompt),
            (
                "user",
                "Issue:\n{issue}\n\nFollow-up research questions:\n"
                "{research_questions}\n\nPrevious research summary:\n"
                "{previous_research}\n\nWeb search snippets:\n{research_data}\n\n"
                "Search error or limitation:\n{research_error}",
            ),
        ]
    )
    planner = ChatPromptTemplate.from_messages(
        [
            ("system", planner_system_prompt),
            (
                "user",
                "Issue:\n{issue}\n\nResearch summary:\n{research}\n\n"
                "Current plan:\n{current_plan}\n\nCritique:\n{critique}",
            ),
        ]
    )
    critic = ChatPromptTemplate.from_messages(
        [
            ("system", critic_system_prompt),
            (
                "user",
                "Issue:\n{issue}\n\nResearch summary:\n{research}\n\nPlan:\n{plan}",
            ),
        ]
    )

    return PromptSet(
        research_system_prompt=research_system_prompt,
        planner_system_prompt=planner_system_prompt,
        critic_system_prompt=critic_system_prompt,
        research=research,
        planner=planner,
        critic=critic,
    )


def prompts_for(domain: str | None) -> PromptSet:
    """Return the prompt set for ``domain``, falling back to the default."""
    return build_prompts(domain or DEFAULT_DOMAIN)


# Software defaults kept under their historical names.
_DEFAULT_PROMPTS = build_prompts(DEFAULT_DOMAIN)
RESEARCH_SYSTEM_PROMPT = _DEFAULT_PROMPTS.research_system_prompt
PLANNER_SYSTEM_PROMPT = _DEFAULT_PROMPTS.planner_system_prompt
CRITIC_SYSTEM_PROMPT = _DEFAULT_PROMPTS.critic_system_prompt
RESEARCH_PROMPT = _DEFAULT_PROMPTS.research
PLANNER_PROMPT = _DEFAULT_PROMPTS.planner
CRITIC_PROMPT = _DEFAULT_PROMPTS.critic
