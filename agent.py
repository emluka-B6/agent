#!/usr/bin/env python3
"""AI issue-planning agent (CLI entry point).

The agent accepts an issue from a selected domain (software engineering, AI,
finance, or cybersecurity), optionally researches it on the web, turns it into
an ordered list of self-contained subtasks, and asks a critic to review the
plan. The final artifact is written to a uniquely named JSON or Markdown file
(``output-<hash>.json`` or ``output-<hash>.md``) so that each subtask can later
be scheduled to an execution-oriented LLM and repeated runs never overwrite an
earlier plan.

This module keeps the command-line interface. The implementation lives in the
``issue_planner`` package; the public names are re-exported here so existing
``import agent`` code keeps working.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from langchain_tavily import TavilySearch

from issue_planner.config import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_RESEARCH_CALLS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_SEARCH_RESULTS,
    DEFAULT_OUTPUT_DIR,
    SUPPORTED_PROVIDERS,
)
from issue_planner.domains import DEFAULT_DOMAIN, DOMAIN_KEYS, get_domain
from issue_planner.graph import build_graph, run_agent
from issue_planner.llm import (
    build_model,
    format_model_catalog,
    normalize_provider,
    resolve_model_name,
)
from issue_planner.models import (
    Critique,
    ImplementationPlan,
    ResearchSummary,
    Source,
    SubTask,
    WorkflowState,
)
from issue_planner.output import (
    build_document,
    build_progress_document,
    document_status,
    render_markdown,
    unique_output_path,
    write_output,
)
from issue_planner.progress import ProgressReporter

__all__ = [
    "Critique",
    "DEFAULT_DOMAIN",
    "DOMAIN_KEYS",
    "ImplementationPlan",
    "ProgressReporter",
    "ResearchSummary",
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
    "main",
    "render_markdown",
    "run_agent",
    "unique_output_path",
    "write_output",
]


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def _resolve_issue(args: argparse.Namespace) -> str:
    if args.issue_file:
        path = args.issue_file.expanduser()
        text = path.read_text(encoding="utf-8")
    elif args.issue:
        text = args.issue
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        raise ValueError(
            "No issue provided. Pass it as an argument, use --issue-file, or pipe it via stdin."
        )

    text = text.strip()
    if not text:
        raise ValueError("The issue must not be empty.")
    return text


def _model_name_for(provider: str, requested: str | None) -> str:
    """Return a validated model ID for the provider."""
    return resolve_model_name(provider, requested)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent.py",
        description=(
            "Turn an issue from software engineering, AI, finance, or cybersecurity "
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
            "  python agent.py --issue-file issue.md --format markdown --skip-research\n"
            "  python agent.py \"Add structured logging\" --output-dir plans\n"
            "  python agent.py --list-models\n"
        ),
    )
    parser.add_argument(
        "issue",
        nargs="?",
        help="Issue or task description. If omitted, stdin is read.",
    )
    parser.add_argument(
        "--issue-file",
        type=Path,
        help="Read the issue from a UTF-8 text file.",
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
        "--provider",
        choices=SUPPORTED_PROVIDERS,
        default=os.getenv("LLM_PROVIDER", "openai"),
        help="LLM provider (default: LLM_PROVIDER or openai).",
    )
    parser.add_argument(
        "--domain",
        choices=DOMAIN_KEYS,
        default=os.getenv("ISSUE_DOMAIN", DEFAULT_DOMAIN),
        help=(
            "Issue domain: software, ai, finance, or cybersecurity "
            "(default: ISSUE_DOMAIN or software)."
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
        help=f"Maximum number of critic reviews (default: {DEFAULT_MAX_ITERATIONS}).",
    )
    parser.add_argument(
        "--max-research-calls",
        type=int,
        default=DEFAULT_MAX_RESEARCH_CALLS,
        help=(
            "Maximum number of research sub-agent calls, including the initial "
            f"pass (default: {DEFAULT_MAX_RESEARCH_CALLS})."
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

    try:
        issue = _resolve_issue(args)
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

    generated_at = datetime.now(timezone.utc).isoformat()
    try:
        output_path, output_format = unique_output_path(
            {"issue": issue, "generated_at": generated_at},
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
        max_iterations=args.max_iterations,
        max_research_calls=args.max_research_calls,
        research_available=search_client is not None and not args.skip_research,
        domain=domain,
        generated_at=generated_at,
    )

    try:
        state = run_agent(
            issue=issue,
            model=model,
            provider=provider,
            search_client=search_client,
            max_iterations=args.max_iterations,
            max_research_calls=args.max_research_calls,
            max_search_results=args.max_search_results,
            skip_research=args.skip_research,
            max_retries=args.max_retries,
            domain=domain,
            reporter=reporter,
        )
    except Exception as exc:  # noqa: BLE001 - report agent failures cleanly
        print(f"error: agent failed: {exc}", file=sys.stderr)
        return 1

    try:
        document = build_document(
            state,
            provider=provider,
            model_name=model_name,
            max_iterations=args.max_iterations,
            generated_at=generated_at,
            max_research_calls=args.max_research_calls,
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

    print(
        f"Plan written to {output_path} ({output_format}, "
        f"status={document['status']}, "
        f"subtasks={len(document['subtasks'])}).",
        file=sys.stderr,
    )
    # Keep the final stdout line machine-readable: the generated file path.
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

