"""Domain profiles that specialise the shared prompt templates.

Every supported area (software engineering, AI, finance, and cybersecurity)
uses the same research -> planner -> critic prompt structure. A
:class:`DomainProfile` holds the domain-specific values that are substituted
into that structure, so a new area can be added by registering one more
profile instead of copying whole prompts.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Domain used when the caller does not select one explicitly.
DEFAULT_DOMAIN = "software"


@dataclass(frozen=True, kw_only=True)
class DomainProfile:
    """Domain-specific values substituted into the shared prompt templates.

    The fields are grouped by workflow phase: research, planner, and critic.
    Optional tuples are rendered as extra bullet points after the shared
    rules/checks, which keeps the common instructions identical for every
    domain.
    """

    key: str
    label: str
    description: str

    # Research phase.
    research_role: str
    research_objective: str
    research_focus: tuple[str, ...]
    research_style: str
    practice_label: str
    research_rules: tuple[str, ...] = ()

    # Planner phase.
    planner_role: str
    planner_objective: str
    planner_focus: tuple[str, ...]
    work_unit: str
    artifact_noun: str
    artifact_scope: str
    verification_hint: str
    planner_rules: tuple[str, ...] = ()

    # Critic phase.
    critic_role: str
    executor_label: str
    critic_checks: tuple[str, ...] = ()


SOFTWARE = DomainProfile(
    key="software",
    label="Software engineering",
    description="implementation and maintenance of software systems",
    research_role="senior software engineer",
    research_objective="implementation-relevant facts",
    research_focus=(
        "APIs, library versions, and compatibility notes",
        "constraints imposed by the existing codebase and runtime",
        "known pitfalls, security concerns, and proven implementation patterns",
    ),
    research_style="implementation-oriented",
    practice_label="engineering practice",
    planner_role="senior software architect",
    planner_objective="an implementation plan for another LLM that will write the code",
    planner_focus=(
        "code structure, interfaces, and integration points",
        "tests, migrations, rollout, and rollback",
        "performance, security, and maintainability risks",
    ),
    work_unit="implementation subtasks",
    artifact_noun="files",
    artifact_scope="files that are known or clearly implied by the issue",
    verification_hint="Prefer automated tests or reproducible commands.",
    critic_role="a rigorous but constructive reviewer of implementation plans",
    executor_label="a coding LLM",
)

AI = DomainProfile(
    key="ai",
    label="AI systems",
    description="building, evaluating, and using AI systems responsibly",
    research_role="senior AI engineer and applied AI researcher",
    research_objective=(
        "evidence-based facts about models, prompting, evaluation, safety, and tooling"
    ),
    research_focus=(
        "model capabilities, limits, context windows, pricing, and API constraints",
        "prompting, retrieval, tool use, and evaluation methodology",
        "hallucination, bias, privacy, security, and human-oversight requirements",
        "applicable AI regulation, policy, or provider usage terms",
    ),
    research_style="evaluation-oriented",
    practice_label="AI engineering practice",
    research_rules=(
        "Prefer primary sources such as model documentation, papers, and official benchmarks.",
        "Record model names, versions, and dates because AI capabilities change quickly.",
    ),
    planner_role="senior AI engineer and solution architect",
    planner_objective=(
        "an AI implementation plan for another LLM that will build or integrate the solution"
    ),
    planner_focus=(
        "model selection, prompting, retrieval, tools, and data flow",
        "evaluation datasets, metrics, guardrails, and red-team checks",
        "cost, latency, privacy, and human-oversight risks",
    ),
    work_unit="AI engineering subtasks",
    artifact_noun="artifacts",
    artifact_scope=(
        "artifacts (files, prompts, datasets, or configuration) that are known "
        "or clearly implied by the issue"
    ),
    verification_hint=(
        "Prefer offline evaluations with explicit datasets, metrics, and thresholds."
    ),
    planner_rules=(
        "Make model, prompt, data, and evaluation choices explicit instead of assuming a provider.",
        "Include an evaluation or red-team step for every model-facing change.",
        "State cost, latency, privacy, and data-retention implications where relevant.",
    ),
    critic_role="a rigorous but constructive reviewer of AI implementation plans",
    executor_label="an AI execution LLM",
    critic_checks=(
        "model, prompt, data, and evaluation assumptions are explicit;",
        "quality and safety are measured with defined metrics rather than asserted;",
        "cost, latency, privacy, and human-oversight risks are addressed;",
    ),
)

FINANCE = DomainProfile(
    key="finance",
    label="Finance and investing",
    description="investment research, portfolio analysis, and financial planning",
    research_role="senior financial analyst and investment researcher",
    research_objective=(
        "evidence-based facts about markets, instruments, costs, risks, and regulation"
    ),
    research_focus=(
        "asset classes, instruments, fees, taxes, liquidity, and time horizon",
        "risk metrics such as volatility, drawdown, concentration, and correlation",
        "valuation, portfolio construction, rebalancing, and backtesting methodology",
        "regulatory, suitability, and data-quality constraints",
    ),
    research_style="quantitative, with explicit risk considerations",
    practice_label="financial analysis practice",
    research_rules=(
        "Distinguish historical evidence from forecasts and label uncertainty.",
        "Record data sources, as-of dates, currencies, and whether returns are nominal or real.",
    ),
    planner_role="senior investment analyst and financial planner",
    planner_objective=(
        "an investment research or portfolio plan that another LLM or analyst can execute"
    ),
    planner_focus=(
        "goals, time horizon, liquidity needs, and risk tolerance",
        "asset allocation, diversification, costs, and tax implications",
        "data sources, calculations, backtesting, and review checkpoints",
    ),
    work_unit="financial analysis or portfolio subtasks",
    artifact_noun="artifacts",
    artifact_scope=(
        "artifacts (documents, datasets, models, or checklists) that are known "
        "or clearly implied by the issue"
    ),
    verification_hint=(
        "Prefer reproducible calculations with stated data sources, dates, and assumptions."
    ),
    planner_rules=(
        "Do not present the result as personalized financial advice or as a guaranteed return.",
        (
            "State goals, time horizon, currency, risk tolerance, and constraints "
            "explicitly; record missing inputs as open questions."
        ),
        "Include risk, cost, tax, and compliance considerations in the plan.",
        "Avoid claiming market-beating performance without out-of-sample evidence.",
    ),
    critic_role="a rigorous but constructive reviewer of financial research and portfolio plans",
    executor_label="an analyst or execution LLM",
    critic_checks=(
        "risk, cost, tax, liquidity, and time-horizon assumptions are explicit;",
        "no guaranteed returns or personalized financial advice are presented;",
        "data provenance and calculation methods are reproducible;",
    ),
)

CYBERSECURITY = DomainProfile(
    key="cybersecurity",
    label="Cybersecurity",
    description="defensive security engineering and risk reduction",
    research_role="senior cybersecurity engineer and defensive security researcher",
    research_objective=(
        "evidence-based facts about threats, vulnerabilities, controls, and defensive practice"
    ),
    research_focus=(
        "threat models, attack surfaces, and relevant advisories or CVEs",
        "defensive controls, secure configuration, detection, and response",
        "authorization boundaries, compliance frameworks, and data handling",
        "vendor support status, versions, and known mitigations",
    ),
    research_style="risk-aware",
    practice_label="security engineering practice",
    research_rules=(
        "Prefer vendor advisories, official documentation, and recognized frameworks.",
        "Separate confirmed vulnerabilities from hypotheses and note severity and exploitability.",
    ),
    planner_role="senior security architect",
    planner_objective=(
        "a defensive security plan that another LLM or engineer can execute safely"
    ),
    planner_focus=(
        "assets, trust boundaries, threat model, and explicitly authorized scope",
        "preventive, detective, and corrective controls",
        "safe validation, rollback, monitoring, and incident-response steps",
    ),
    work_unit="security subtasks",
    artifact_noun="artifacts",
    artifact_scope=(
        "artifacts (files, configurations, or evidence) that are known or "
        "clearly implied by the issue"
    ),
    verification_hint="Prefer non-destructive checks in an authorized test environment.",
    planner_rules=(
        (
            "Operate strictly defensively and only within explicitly authorized "
            "scope; never plan unauthorized access, exploitation, or evasion."
        ),
        "Prefer non-destructive validation in controlled environments and include rollback steps.",
        "Apply least privilege and protect sensitive data throughout the plan.",
        "Do not include operational exploit payloads or instructions that could cause harm.",
    ),
    critic_role="a rigorous but constructive reviewer of defensive security plans",
    executor_label="a security engineer or execution LLM",
    critic_checks=(
        "the scope is explicitly authorized and the plan is defensive only;",
        "controls are testable and risk is prioritized by likelihood and impact;",
        "no offensive, destructive, or unverifiable steps are included;",
    ),
)

#: Registered domains, keyed by the value accepted by ``--domain``.
DOMAINS: dict[str, DomainProfile] = {
    profile.key: profile for profile in (SOFTWARE, AI, FINANCE, CYBERSECURITY)
}

#: Domain keys in registry order, suitable for ``argparse`` choices and help.
DOMAIN_KEYS: tuple[str, ...] = tuple(DOMAINS)


def _bullets(items: tuple[str, ...]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _extra_bullets(items: tuple[str, ...]) -> str:
    """Render optional rules so they continue the preceding bullet list."""
    return "".join(f"\n- {item}" for item in items)


def prompt_values(profile: DomainProfile) -> dict[str, str]:
    """Render a profile as values for the shared ``string.Template`` prompts."""
    return {
        "research_role": profile.research_role,
        "research_objective": profile.research_objective,
        "research_focus": _bullets(profile.research_focus),
        "research_style": profile.research_style,
        "practice_label": profile.practice_label,
        "research_rules": _extra_bullets(profile.research_rules),
        "planner_role": profile.planner_role,
        "planner_objective": profile.planner_objective,
        "planner_focus": _bullets(profile.planner_focus),
        "work_unit": profile.work_unit,
        "artifact_noun": profile.artifact_noun,
        "artifact_scope": profile.artifact_scope,
        "verification_hint": profile.verification_hint,
        "planner_rules": _extra_bullets(profile.planner_rules),
        "critic_role": profile.critic_role,
        "executor_label": profile.executor_label,
        "critic_checks": _extra_bullets(profile.critic_checks),
    }


def get_domain(key: str) -> DomainProfile:
    """Return the domain profile for ``key`` (case-insensitive)."""
    normalized = (key or "").strip().lower()
    try:
        return DOMAINS[normalized]
    except KeyError:
        raise ValueError(
            f"Unsupported domain '{key}'. Choose one of: {', '.join(DOMAIN_KEYS)}."
        ) from None
