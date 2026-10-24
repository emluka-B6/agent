"""Tests for domain profiles and the shared prompt templates."""

from __future__ import annotations

import pytest

from agent import build_parser
from issue_planner import nodes as nodes_module
from issue_planner.domains import DEFAULT_DOMAIN, DOMAIN_KEYS, DOMAINS, get_domain
from issue_planner.graph import run_agent
from issue_planner.models import ImplementationPlan, SubTask, WorkflowState
from issue_planner.nodes import planner_node
from issue_planner.output import build_document, render_markdown
from issue_planner.prompts import (
    CRITIC_PROMPT,
    PLANNER_PROMPT,
    RESEARCH_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
    build_prompts,
    prompts_for,
)

EXPECTED_DOMAINS = ("software", "ai", "finance", "cybersecurity")


def make_plan() -> ImplementationPlan:
    return ImplementationPlan(
        summary="A plan",
        subtasks=[
            SubTask(
                id="T1",
                title="Task T1",
                description="Do the first thing",
                implementation_prompt="Do the first thing with these constraints.",
                files=["notes.md"],
                acceptance_criteria=["The first thing is done"],
                verification="pytest",
            )
        ],
        definition_of_done=["Everything works"],
    )


def test_domain_registry_contains_the_supported_areas() -> None:
    assert DOMAIN_KEYS == EXPECTED_DOMAINS
    assert DEFAULT_DOMAIN == "software"
    assert set(DOMAINS) == set(EXPECTED_DOMAINS)


@pytest.mark.parametrize("domain", EXPECTED_DOMAINS)
def test_shared_templates_keep_the_same_input_variables(domain: str) -> None:
    prompts = prompts_for(domain)

    assert set(prompts.research.input_variables) == {
        "issue",
        "research_questions",
        "previous_research",
        "research_data",
        "research_error",
    }
    assert set(prompts.planner.input_variables) == {
        "issue",
        "research",
        "current_plan",
        "critique",
    }
    assert set(prompts.critic.input_variables) == {"issue", "research", "plan"}


@pytest.mark.parametrize("domain", EXPECTED_DOMAINS)
def test_each_domain_fills_the_shared_template_without_leftovers(domain: str) -> None:
    prompts = prompts_for(domain)
    profile = DOMAINS[domain]

    assert profile.research_role in prompts.research_system_prompt
    assert profile.planner_role in prompts.planner_system_prompt
    assert profile.critic_role in prompts.critic_system_prompt
    for system_prompt in (
        prompts.research_system_prompt,
        prompts.planner_system_prompt,
        prompts.critic_system_prompt,
    ):
        assert "$" not in system_prompt
        assert "{issue}" not in system_prompt


def test_domain_specific_rules_are_rendered() -> None:
    assert (
        "personalized financial advice"
        in prompts_for("finance").planner_system_prompt
    )
    assert (
        "explicitly authorized scope"
        in prompts_for("cybersecurity").planner_system_prompt
    )
    assert "offline evaluations" in prompts_for("ai").planner_system_prompt
    assert "automated tests" in prompts_for("software").planner_system_prompt


def test_planner_prompt_asks_for_follow_up_research_only_when_needed() -> None:
    planner_prompt = prompts_for(DEFAULT_DOMAIN).planner_system_prompt

    assert "research_requests" in planner_prompt
    assert "missing from the research summary" in planner_prompt
    assert "research calls are limited" in planner_prompt


def test_research_prompt_accepts_follow_up_context() -> None:
    messages = prompts_for(DEFAULT_DOMAIN).research.format_messages(
        issue="Add rate limiting",
        research_questions="- What is the current guidance?",
        previous_research='{"recommended_approach": "Token buckets"}',
        research_data="new snippet",
        research_error="none",
    )

    rendered = "\n".join(str(message.content) for message in messages)
    assert "What is the current guidance?" in rendered
    assert "Token buckets" in rendered
    assert "new snippet" in rendered


def test_domains_produce_distinct_prompts() -> None:
    system_prompts = {
        domain: prompts_for(domain).research_system_prompt
        for domain in EXPECTED_DOMAINS
    }

    assert len(set(system_prompts.values())) == len(EXPECTED_DOMAINS)


def test_prompts_are_cached_and_domain_lookup_is_case_insensitive() -> None:
    assert build_prompts("finance") is build_prompts("finance")
    assert (
        prompts_for("Finance").planner_system_prompt
        == prompts_for("finance").planner_system_prompt
    )


def test_unknown_domains_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported domain"):
        get_domain("medicine")
    with pytest.raises(ValueError, match="Unsupported domain"):
        prompts_for("medicine")
    with pytest.raises(ValueError, match="Unsupported domain"):
        run_agent("Do something", model=object(), domain="medicine")  # type: ignore[arg-type]


def test_default_prompt_constants_stay_on_the_software_domain() -> None:
    software = prompts_for(DEFAULT_DOMAIN)
    messages = software.research.format_messages(
        issue="Add caching",
        research_questions="(none; initial research)",
        previous_research="(no research summary)",
        research_data="snippet",
        research_error="none",
    )

    assert RESEARCH_SYSTEM_PROMPT == software.research_system_prompt
    assert RESEARCH_PROMPT.format_messages(
        issue="Add caching",
        research_questions="(none; initial research)",
        previous_research="(no research summary)",
        research_data="snippet",
        research_error="none",
    ) == messages
    assert PLANNER_PROMPT.input_variables == software.planner.input_variables
    assert CRITIC_PROMPT.input_variables == software.critic.input_variables


@pytest.mark.parametrize(
    ("domain", "expected_role"),
    [
        ("finance", "senior investment analyst"),
        ("cybersecurity", "senior security architect"),
        (None, "senior software architect"),
    ],
)
def test_planner_node_uses_the_domain_from_state(
    monkeypatch: pytest.MonkeyPatch,
    domain: str | None,
    expected_role: str,
) -> None:
    captured: dict[str, str] = {}

    def fake_invoke_structured(
        model: object,
        schema: type[ImplementationPlan],
        provider: str,
        messages: object,
        label: str,
        max_retries: int,
    ) -> ImplementationPlan:
        captured["system"] = messages[0].content  # type: ignore[index]
        return make_plan()

    monkeypatch.setattr(nodes_module, "invoke_structured", fake_invoke_structured)

    state: WorkflowState = {"issue": "Do the work", "plan": None, "critique": None}
    if domain is not None:
        state["domain"] = domain

    planner_node(state, model=object(), provider="openai", max_retries=1)  # type: ignore[arg-type]

    assert expected_role in captured["system"]


def test_output_records_the_selected_domain() -> None:
    state: WorkflowState = {
        "issue": "Rebalance a portfolio",
        "domain": "finance",
        "research": None,
        "sources": [],
        "research_error": None,
        "plan": make_plan(),
        "critique": None,
        "evaluations": 0,
        "approved": False,
    }

    document = build_document(
        state, provider="openai", model_name="gpt-5-mini", max_iterations=2
    )

    assert document["domain"] == "finance"
    assert "**Domain:** finance" in render_markdown(document)


def test_output_defaults_to_the_software_domain_for_older_states() -> None:
    state: WorkflowState = {
        "issue": "Add caching",
        "plan": make_plan(),
        "evaluations": 0,
        "approved": False,
    }

    document = build_document(
        state, provider="openai", model_name="gpt-5-mini", max_iterations=2
    )

    assert document["domain"] == "software"


def test_cli_exposes_the_domain_choice() -> None:
    args = build_parser().parse_args(["Do something", "--domain", "ai"])

    assert args.domain == "ai"
