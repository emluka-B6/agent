#!/usr/bin/env python3
"""AI work-planning agent (CLI entry point).

The agent accepts a brief from a selected domain (software engineering, AI,
finance, or cybersecurity), optionally researches it on the web, turns it into
an ordered list of self-contained subtasks, and asks a critic to review the
plan. The final artifact is written to a uniquely named JSON or Markdown file
(``output-<hash>.json`` or ``output-<hash>.md``) so that each subtask can later
be scheduled to an execution-oriented LLM and repeated runs never overwrite an
earlier plan.

This module keeps the command-line interface. The implementation lives in the
``work_planner`` package; the public names are re-exported here so existing
``import agent`` code keeps working.

An unfinished run (for example ``status=max_iterations_reached``) can be
continued with ``--resume``: the previous artifact supplies the brief, domain,
plan, review, and research, and the iteration/research budgets are extended
instead of being reset.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from langchain_tavily import TavilySearch

from work_planner.budget import RunBudget
from work_planner.config import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_RESEARCH_CALLS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_SEARCH_RESULTS,
    DEFAULT_OUTPUT_DIR,
    SUPPORTED_PROVIDERS,
)
from work_planner.domains import DEFAULT_DOMAIN, DOMAIN_KEYS, get_domain
from work_planner.graph import build_graph, run_agent
from work_planner.llm import (
    build_model,
    format_model_catalog,
    normalize_provider,
    resolve_model_name,
)
from work_planner.models import (
    Critique,
    ImplementationPlan,
    OpenQuestion,
    ResearchSummary,
    Source,
    SubTask,
    WorkflowState,
)
from work_planner.output import (
    build_document,
    build_progress_document,
    document_status,
    render_markdown,
    unique_output_path,
    write_output,
)
from work_planner.progress import ProgressReporter
from work_planner.questions import (
    apply_answers,
    format_questions,
    load_answer_file,
    parse_answer_arguments,
    pending_questions,
    prompt_answers,
)
from work_planner.resume import ResumeState, load_resume_state, resolve_resume_path

__all__ = [
    "Critique",
    "DEFAULT_DOMAIN",
    "DOMAIN_KEYS",
    "ImplementationPlan",
    "OpenQuestion",
    "ProgressReporter",
    "ResearchSummary",
    "ResumeState",
    "RunBudget",
    "Source",
    "SubTask",
    "WorkflowState",
    "build_document",
    "build_graph",
    "build_model",
    "build_parser",
    "build_progress_document",
    "document_status",
    "get_domain",
    "load_resume_state",
    "main",
    "render_markdown",
    "resolve_resume_path",
    "run_agent",
    "unique_output_path",
    "write_output",
]


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def _resolve_brief(args: argparse.Namespace) -> str:
    if args.brief_file:
        path = args.brief_file.expanduser()
        text = path.read_text(encoding="utf-8")
    elif args.brief:
        text = args.brief
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        raise ValueError(
            "No brief provided. Pass it as an argument, use --brief-file, "
            "or pipe it via stdin."
        )

    text = text.strip()
    if not text:
        raise ValueError("The brief must not be empty.")
    return text


def _model_name_for(provider: str, requested: str | None) -> str:
    """Return a validated model ID for the provider."""
    return resolve_model_name(provider, requested)


def _apply_cli_answers(
    plan: ImplementationPlan, args: argparse.Namespace
) -> ImplementationPlan:
    """Return ``plan`` with the answers supplied on the command line."""
    entries: list[tuple[str, str]] = []
    if args.answers is not None:
        entries.extend(load_answer_file(args.answers))
    entries.extend(parse_answer_arguments(args.answer or []))
    return apply_answers(plan, entries)


def _report_open_questions(items: Iterable[object], output_path: Path) -> None:
    """Print the pending questions of a finished artifact and how to answer them."""
    try:
        questions = [OpenQuestion.model_validate(item) for item in items]
    except Exception:  # noqa: BLE001 - reporting must never fail a finished run
        return
    if not pending_questions(questions):
        return
    print("", file=sys.stderr)
    print(format_questions(questions), file=sys.stderr)
    if output_path.suffix.lower() in {".md", ".markdown"}:
        print(
            "info: these decisions do not block execution. Only JSON artifacts "
            "can be resumed, so answer them from a run that writes JSON.",
            file=sys.stderr,
        )
        return
    print(
        "info: these decisions do not block execution; answer them with\n"
        f"  python agent.py --resume {output_path} --answers answers.json\n"
        '  where answers.json is {"<question number or text>": "<answer>"}, or\n'
        f"  python agent.py --resume {output_path} --answer 1=2 --answer 2=text",
        file=sys.stderr,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent.py",
        description=(
            "Turn a brief from software engineering, AI, finance, or cybersecurity "
            "into an ordered set of self-contained subtasks that can be scheduled "
            "to an execution LLM."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python agent.py \"Add rate limiting to the public API\"\n"
            "  python agent.py \"Audit a RAG assistant for prompt injection\" --domain ai\n"
            "  python agent.py \"Build a rebalancing checklist for a three-fund portfolio\" --domain finance\n"
            "  python agent.py \"Harden the public API against the OWASP API Top 10\" --domain cybersecurity\n"
            "  python agent.py --brief-file brief.md --format markdown --skip-research\n"
            "  python agent.py \"Add structured logging\" --output-dir plans\n"
            "  python agent.py --resume latest --max-iterations 2\n"
            "  python agent.py --resume latest --answer 1=2 --answer 2=PostgreSQL\n"
            "  python agent.py --resume latest --answers answers.json\n"
            "  python agent.py --list-models\n"
        ),
    )
    parser.add_argument(
        "brief",
        nargs="?",
        help="Brief or task description. If omitted, stdin is read.",
    )
    parser.add_argument(
        "--brief-file",
        "--issue-file",
        dest="brief_file",
        type=Path,
        help=(
            "Read the brief from a UTF-8 text file "
            "(--issue-file is a deprecated alias)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for the generated artifact. The file name is derived "
            "from a hash, so runs never overwrite each other "
            f"(default: {DEFAULT_OUTPUT_DIR})."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format (default: json).",
    )
    parser.add_argument(
        "--resume",
        metavar="ARTIFACT",
        help=(
            "Continue an earlier run from its JSON artifact instead of starting "
            "over. Pass a file, a directory (its newest artifact is used), or "
            "'latest' (the newest artifact in --output-dir). The brief, domain, "
            "plan, review, and research come from the artifact, and "
            "--max-iterations/--max-research-calls are added to the budget that "
            "artifact already spent."
        ),
    )
    parser.add_argument(
        "--answers",
        type=Path,
        metavar="FILE",
        help=(
            "Answer the open questions of the resumed artifact. FILE is a JSON "
            "object that maps a question number or text to an answer, or a "
            "list of {question, answer} objects. An answer may be an option "
            "number, an option text, or free text. Requires --resume."
        ),
    )
    parser.add_argument(
        "--answer",
        action="append",
        default=[],
        metavar="QUESTION=ANSWER",
        help=(
            "Answer one open question of the resumed artifact, for example "
            "--answer 1=2 or --answer 2=PostgreSQL; QUESTION is the number "
            "shown in the artifact. Repeat the flag for several questions. "
            "Requires --resume."
        ),
    )
    parser.add_argument(
        "--skip-questions",
        action="store_true",
        help=(
            "Do not ask for answers to the open questions of a resumed run. "
            "Without this flag the prompt appears only when stdin is a "
            "terminal and neither --answers nor --answer was given."
        ),
    )
    parser.add_argument(
        "--provider",
        choices=SUPPORTED_PROVIDERS,
        default=os.getenv("LLM_PROVIDER", "openai"),
        help="LLM provider (default: LLM_PROVIDER or openai).",
    )
    parser.add_argument(
        "--domain",
        choices=DOMAIN_KEYS,
        default=(
            os.getenv("PLAN_DOMAIN") or os.getenv("ISSUE_DOMAIN") or DEFAULT_DOMAIN
        ),
        help=(
            "Domain of the brief: software, ai, finance, or cybersecurity "
            "(default: PLAN_DOMAIN or software; ISSUE_DOMAIN is the "
            "deprecated name of that variable)."
        ),
    )
    parser.add_argument(
        "--model",
        help=(
            "Model ID from the provider allow-list. Defaults to OPENAI_MODEL or "
            "DEEPSEEK_MODEL, then to the provider default. "
            "Use --list-models to see every supported value."
        ),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List the supported models for every provider and exit.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=(
            "Maximum number of critic reviews in one run "
            f"(default: {DEFAULT_MAX_ITERATIONS}). With --resume this many "
            "reviews are added on top of the ones the artifact already spent."
        ),
    )
    parser.add_argument(
        "--max-research-calls",
        type=int,
        default=DEFAULT_MAX_RESEARCH_CALLS,
        help=(
            "Maximum number of research sub-agent calls, including the initial "
            f"pass (default: {DEFAULT_MAX_RESEARCH_CALLS}). With --resume these "
            "are added on top of the calls the artifact already spent."
        ),
    )
    parser.add_argument(
        "--max-search-results",
        type=int,
        default=DEFAULT_MAX_SEARCH_RESULTS,
        help=(
            "Maximum number of web results per search call "
            f"(default: {DEFAULT_MAX_SEARCH_RESULTS})."
        ),
    )
    parser.add_argument(
        "--skip-research",
        action="store_true",
        help="Do not call the web search provider.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Maximum attempts for a failing LLM/search call (default: {DEFAULT_MAX_RETRIES}).",
    )
    return parser



def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        print(format_model_catalog())
        return 0

    resume: ResumeState | None = None
    if args.resume is not None:
        try:
            resume = load_resume_state(
                resolve_resume_path(args.resume, args.output_dir)
            )
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    if resume is None and (args.answers is not None or args.answer):
        print(
            "error: --answers and --answer can only be used with --resume.",
            file=sys.stderr,
        )
        return 2

    if resume is not None:
        if args.brief or args.brief_file:
            print(
                "info: --resume ignores the brief argument; the brief and the "
                "domain are read from the artifact.",
                file=sys.stderr,
            )
        brief = resume.brief
        domain = resume.domain
    else:
        try:
            brief = _resolve_brief(args)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        try:
            domain = get_domain(args.domain).key
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    if args.max_iterations < 1:
        print("error: --max-iterations must be at least 1.", file=sys.stderr)
        return 2
    if args.max_research_calls < 1:
        print("error: --max-research-calls must be at least 1.", file=sys.stderr)
        return 2
    if args.max_search_results < 1:
        print("error: --max-search-results must be at least 1.", file=sys.stderr)
        return 2
    if args.max_retries < 1:
        print("error: --max-retries must be at least 1.", file=sys.stderr)
        return 2

    budget = (
        RunBudget.continued(resume, args.max_iterations, args.max_research_calls)
        if resume is not None
        else RunBudget.fresh(args.max_iterations, args.max_research_calls)
    )
    if resume is not None:
        print(
            f"info: continuing {resume.name} (status={resume.status}, "
            f"runs={resume.runs}, iterations={resume.evaluations}/"
            f"{resume.max_iterations}, research calls={resume.research_calls}/"
            f"{resume.max_research_calls}); this run adds "
            f"{budget.iterations_available} iteration(s) and "
            f"{budget.research_calls_available} research call(s), for a total "
            f"budget of {budget.max_iterations} iteration(s) and "
            f"{budget.max_research_calls} research call(s).",
            file=sys.stderr,
        )
        if resume.status == "approved":
            print(
                "note: the resumed artifact was already approved; this run will "
                "revise and review the plan again.",
                file=sys.stderr,
            )

    provider = args.provider
    try:
        provider = normalize_provider(provider)
        model_name = _model_name_for(provider, args.model)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    api_key_env = "DEEPSEEK_API_KEY" if provider == "deepseek" else "OPENAI_API_KEY"
    if not os.getenv(api_key_env):
        print(
            f"error: {api_key_env} is not set. Add it to .env or the environment.",
            file=sys.stderr,
        )
        return 2

    try:
        model = build_model(provider, model_name)
    except Exception as exc:  # noqa: BLE001 - configuration errors are user-facing
        print(f"error: could not create the chat model: {exc}", file=sys.stderr)
        return 2

    search_client: TavilySearch | None = None
    if args.skip_research:
        print("info: web research is disabled.", file=sys.stderr)
    elif not os.getenv("TAVILY_API_KEY"):
        print(
            "warning: TAVILY_API_KEY is not set; continuing without web research.",
            file=sys.stderr,
        )
    else:
        try:
            search_client = TavilySearch(
                max_results=args.max_search_results,
                search_depth="basic",
            )
        except Exception as exc:  # noqa: BLE001 - research is best-effort
            print(
                f"warning: could not initialize web search: {exc}. "
                "Continuing without web research.",
                file=sys.stderr,
            )

    initial_state: WorkflowState | None = None
    if resume is not None:
        try:
            answered_plan = _apply_cli_answers(resume.plan, args)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        interactive = (
            not args.skip_questions
            and args.answers is None
            and not args.answer
            and sys.stdin.isatty()
        )
        pending = pending_questions(answered_plan.open_questions)
        if pending and interactive:
            print(
                f"info: {len(pending)} open question(s) need a human decision; "
                "the plan runs without an answer, so press Enter to skip one.",
                file=sys.stderr,
            )
            answered_plan = apply_answers(
                answered_plan, prompt_answers(answered_plan)
            )
        elif pending:
            print(format_questions(answered_plan.open_questions), file=sys.stderr)
            print(
                "info: answer them with --answers FILE or --answer "
                "QUESTION=ANSWER; the plan runs without them.",
                file=sys.stderr,
            )

        answered = sum(
            1 for question in answered_plan.open_questions if question.answer
        )
        if answered:
            print(
                f"info: {answered} open question(s) answered; the planner will "
                "apply the decisions to the plan.",
                file=sys.stderr,
            )

        initial_state = resume.as_initial_state()
        initial_state["plan"] = answered_plan

    generated_at = datetime.now(timezone.utc).isoformat()
    try:
        output_path, output_format = unique_output_path(
            {"brief": brief, "generated_at": generated_at},
            args.output_dir,
            args.format,
        )
    except Exception as exc:  # noqa: BLE001 - report output failures cleanly
        print(f"error: could not prepare the output file: {exc}", file=sys.stderr)
        return 1

    reporter = ProgressReporter(
        output_path=output_path,
        output_format=output_format,
        provider=provider,
        model_name=model_name,
        max_iterations=budget.max_iterations,
        max_research_calls=budget.max_research_calls,
        research_available=search_client is not None and not args.skip_research,
        domain=domain,
        generated_at=generated_at,
        budget=budget,
    )

    try:
        state = run_agent(
            brief=brief,
            model=model,
            provider=provider,
            search_client=search_client,
            max_iterations=budget.max_iterations,
            max_research_calls=budget.max_research_calls,
            max_search_results=args.max_search_results,
            skip_research=args.skip_research,
            max_retries=args.max_retries,
            domain=domain,
            reporter=reporter,
            initial_state=initial_state,
        )
    except Exception as exc:  # noqa: BLE001 - report agent failures cleanly
        print(f"error: agent failed: {exc}", file=sys.stderr)
        return 1

    try:
        document = build_document(
            state,
            provider=provider,
            model_name=model_name,
            max_iterations=budget.max_iterations,
            generated_at=generated_at,
            max_research_calls=budget.max_research_calls,
            budget=budget,
        )
        output_path, output_format = write_output(
            document, output_path, output_format
        )
    except Exception as exc:  # noqa: BLE001 - report output failures cleanly
        print(f"error: could not write the result: {exc}", file=sys.stderr)
        if reporter.partial_written:
            print(
                f"error: the partial result is still available in {output_path}.",
                file=sys.stderr,
            )
        return 1

    metadata = document["metadata"]
    print(
        f"Plan written to {output_path} ({output_format}, "
        f"status={document['status']}, "
        f"subtasks={len(document['subtasks'])}, "
        f"iterations={metadata['evaluations']}/{metadata['max_iterations']}, "
        f"research calls={metadata['research_calls']}/"
        f"{metadata['max_research_calls']}, "
        f"runs={metadata['runs']}).",
        file=sys.stderr,
    )
    _report_open_questions(document.get("open_questions") or [], output_path)
    # Keep the final stdout line machine-readable: the generated file path.
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

