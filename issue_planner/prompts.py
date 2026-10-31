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

from issue_planner.config import (
    DEFAULT_MAX_SUBTASKS,
    TARGET_IMPLEMENTATION_PROMPT_CHARS,
)
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

_PLANNER_RULES = """You are a $planner_role creating $planner_objective.
Break the issue into ordered, independently schedulable $work_unit.

Focus areas:
$planner_focus

Size policy:
- small: one file or artifact, one behaviour, verifiable in a single step;
- medium: two to five files or artifacts, one reviewable change;
- large: more than five files, a schema or contract change, or several independent behaviours.
Split every subtask you would label large. A plan has at most $max_subtasks subtasks.
Keep each implementation_prompt under about $implementation_prompt_budget characters: state the goal, the interfaces to use, the constraints, the expected $artifact_noun, and the acceptance criteria. Do not paste whole files.

Top-level fields (keep them distinct and never restate one in another):
- summary: two to four sentences about the strategy and why it fits the issue; no subtask-by-subtask recap and no completion checklist.
- assumptions: unverified premises the plan depends on, one per item and stated as a premise; no tasks, checks, or questions.
- open_questions: decisions that only a human can make; none of them may block execution.
- definition_of_done: a bullet list of observable, testable end-state conditions that prove the whole issue is solved, one condition per item; no assumptions, no questions, and no per-subtask acceptance criteria.

Rules:
- Use stable IDs such as T1, T2, T3.
- Every subtask must have a self-contained implementation_prompt with the goal, context, constraints, expected $artifact_noun, and acceptance criteria.
- Dependencies must reference existing subtask IDs and must not form cycles.
- List only $artifact_scope. Record uncertain items in assumptions, risks, or open_questions.
- Include concrete acceptance criteria and a verification step for every subtask. $verification_hint
- Keep the scope aligned with the issue; do not add unrelated work.
- Never include secrets, credentials, or private personal data.
- When a critique is provided, address every blocking issue and preserve the correct parts of the previous plan.
- When a blocking issue needs external or current information that is missing from the research summary, put up to three focused web search queries in research_requests. Leave research_requests empty when the existing research is sufficient; the request states how much research budget remains.
"""

_PLANNER_REVISION_RULES = """

Revision mode:
- The plan already exists. Return only what changes: put new or modified subtasks in upsert_subtasks (reuse an id to replace it, use a new id to add one), list deleted ids in remove_subtasks, and omit untouched subtasks entirely.
- Leave a field at null when it does not change; an empty list clears it.
- Change only what the critique or the new research requires.
"""

_PLANNER_LANGUAGE_RULE = "- Write in the same language as the issue.$planner_rules"

PLANNER_SYSTEM_TEMPLATE = Template(_PLANNER_RULES + _PLANNER_LANGUAGE_RULE)

PLANNER_REVISION_SYSTEM_TEMPLATE = Template(
    _PLANNER_RULES + _PLANNER_LANGUAGE_RULE + _PLANNER_REVISION_RULES
)

CRITIC_SYSTEM_TEMPLATE = Template(
    """You are $critic_role.
Evaluate whether the plan can be handed to $executor_label without further clarification.

Check:
- completeness relative to the issue;
- atomicity and schedulability of every subtask;
- sizing: every subtask is small or medium, its complexity matches the described scope, and the plan has at most $max_subtasks subtasks;
- correct dependencies and execution order;
- observable acceptance criteria and concrete verification;
- no repeated content between summary, assumptions, open_questions, and definition_of_done, and a definition_of_done written as one observable condition per bullet;
- realistic $artifact_noun and scope;
- self-contained implementation prompts that respect the size policy;
- open_questions contains only decisions that never block execution.$critic_checks

Set approved to true only if there are no blocking issues.
Blocking issues must be concrete and actionable: name the affected subtask ids and the required action (rewrite, split, merge, remove, add, or research).
If a previous review is supplied, confirm that every earlier blocking issue is resolved and report any regression.
If a blocking issue depends on facts that are missing from the research summary, say so explicitly so the planner can request more research.
Do not block on stylistic preferences or wording; block only when the plan cannot be executed as written.
Write in the same language as the issue."""
)


@dataclass(frozen=True)
class PromptSet:
    """The rendered system prompts and chat templates for one domain."""

    research_system_prompt: str
    planner_system_prompt: str
    planner_revision_system_prompt: str
    critic_system_prompt: str
    research: ChatPromptTemplate
    planner: ChatPromptTemplate
    planner_revision: ChatPromptTemplate
    critic: ChatPromptTemplate


@lru_cache(maxsize=None)
def build_prompts(domain: str = DEFAULT_DOMAIN) -> PromptSet:
    """Render the shared templates for ``domain`` (cached per domain key)."""
    values = prompt_values(get_domain(domain))
    # Workflow-wide values that are not domain specific but must be substituted
    # into the same templates so every domain keeps an identical contract.
    values.update(
        {
            "max_subtasks": str(DEFAULT_MAX_SUBTASKS),
            "implementation_prompt_budget": (
                f"{TARGET_IMPLEMENTATION_PROMPT_CHARS:,}"
            ),
        }
    )
    research_system_prompt = RESEARCH_SYSTEM_TEMPLATE.substitute(**values)
    planner_system_prompt = PLANNER_SYSTEM_TEMPLATE.substitute(**values)
    planner_revision_system_prompt = PLANNER_REVISION_SYSTEM_TEMPLATE.substitute(
        **values
    )
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
    planner_messages = [
        ("system", planner_system_prompt),
        (
            "user",
            "Issue:\n{issue}\n\nResearch summary:\n{research}\n\n"
            "Research sources (titles and URLs):\n{sources}\n\n"
            "Research status:\n{research_status}\n\n"
            "Current plan:\n{current_plan}\n\n"
            "Critique to address:\n{critique}",
        ),
    ]
    planner_revision_messages = [
        ("system", planner_revision_system_prompt),
        (
            "user",
            "Issue:\n{issue}\n\nResearch summary:\n{research}\n\n"
            "Research sources (titles and URLs):\n{sources}\n\n"
            "Research status:\n{research_status}\n\n"
            "Current plan:\n{current_plan}\n\n"
            "Critique to address:\n{critique}",
        ),
    ]
    critic = ChatPromptTemplate.from_messages(
        [
            ("system", critic_system_prompt),
            (
                "user",
                "Issue:\n{issue}\n\nReview:\n{review_context}\n\n"
                "Research summary:\n{research}\n\n"
                "Research sources (titles and URLs):\n{sources}\n\n"
                "Research status:\n{research_status}\n\n"
                "Plan:\n{plan}\n\nPrevious review:\n{previous_critique}",
            ),
        ]
    )

    return PromptSet(
        research_system_prompt=research_system_prompt,
        planner_system_prompt=planner_system_prompt,
        planner_revision_system_prompt=planner_revision_system_prompt,
        critic_system_prompt=critic_system_prompt,
        research=research,
        planner=ChatPromptTemplate.from_messages(planner_messages),
        planner_revision=ChatPromptTemplate.from_messages(
            planner_revision_messages
        ),
        critic=critic,
    )


def prompts_for(domain: str | None) -> PromptSet:
    """Return the prompt set for ``domain``, falling back to the default."""
    return build_prompts(domain or DEFAULT_DOMAIN)


# Software defaults kept under their historical names.
_DEFAULT_PROMPTS = build_prompts(DEFAULT_DOMAIN)
RESEARCH_SYSTEM_PROMPT = _DEFAULT_PROMPTS.research_system_prompt
PLANNER_SYSTEM_PROMPT = _DEFAULT_PROMPTS.planner_system_prompt
PLANNER_REVISION_SYSTEM_PROMPT = (
    _DEFAULT_PROMPTS.planner_revision_system_prompt
)
CRITIC_SYSTEM_PROMPT = _DEFAULT_PROMPTS.critic_system_prompt
RESEARCH_PROMPT = _DEFAULT_PROMPTS.research
PLANNER_PROMPT = _DEFAULT_PROMPTS.planner
PLANNER_REVISION_PROMPT = _DEFAULT_PROMPTS.planner_revision
CRITIC_PROMPT = _DEFAULT_PROMPTS.critic
