"""Helpers that convert web searches and research summaries into prompt context."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from issue_planner.models import ResearchSummary, Source


def format_research(research: ResearchSummary | None) -> str:
    """Render a research summary as JSON for a prompt."""
    if research is None:
        return "(no research summary)"
    return json.dumps(research.model_dump(), indent=2, ensure_ascii=False)


def format_sources_for_prompt(sources: list[Source], max_chars: int = 16000) -> str:
    """Render search results as compact prompt text, bounded by ``max_chars``."""
    if not sources:
        return "(no web search results)"

    chunks: list[str] = []
    total = 0
    for source in sources:
        chunk = (
            f"Title: {source.title}\n"
            f"URL: {source.url}\n"
            f"Snippet: {source.snippet.strip()}"
        )
        if total + len(chunk) > max_chars:
            break
        chunks.append(chunk)
        total += len(chunk) + len("\n---\n")
    return "\n---\n".join(chunks)


def format_research_questions(questions: list[str]) -> str:
    """Render the planner's follow-up questions for the research prompt."""
    cleaned = [question.strip() for question in questions if question.strip()]
    if not cleaned:
        return "(none; initial research)"
    return "\n".join(f"- {question}" for question in cleaned)


def format_source_list(
    sources: Iterable[Source], max_sources: int = 20, max_chars: int = 4000
) -> str:
    """Render sources as a numbered title/URL list without snippets.

    The planner and the critic receive this compact list instead of the full
    snippets: it lets them check where a claim came from without paying for the
    search-result text a second time. Entries are numbered so that research facts
    can cite them as ``[S1]``.
    """
    items = list(sources)
    lines: list[str] = []
    total = 0
    for index, source in enumerate(items, start=1):
        if index > max_sources:
            break
        label = source.title.strip() or source.url.strip() or "untitled source"
        line = f"[S{index}] {label}"
        if source.url.strip():
            line = f"{line} ({source.url.strip()})"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line) + 1

    if not lines:
        return "(no sources were collected)"
    omitted = len(items) - len(lines)
    if omitted > 0:
        lines.append(f"... {omitted} more source(s) omitted")
    return "\n".join(lines)


def format_research_status(
    research_error: str | None, research_calls: int, max_research_calls: int
) -> str:
    """Render the research limitation and the remaining follow-up budget.

    The planner and the critic need both facts to decide whether missing
    information justifies another research call or has to be recorded as an
    assumption.
    """
    limitation = (
        f"limitation: {research_error}"
        if research_error
        else "no research limitation was reported"
    )
    remaining = max(0, max_research_calls - research_calls)
    if remaining:
        budget = (
            f"follow-up research remaining: {remaining} of {max_research_calls}"
        )
    else:
        budget = (
            f"follow-up research budget exhausted ({research_calls} of "
            f"{max_research_calls}); do not request more research"
        )
    return f"{limitation}; {budget}"


def merge_sources(
    existing: Iterable[Source], new: Iterable[Source]
) -> list[Source]:
    """Merge source lists, keeping the first occurrence of each result."""
    merged: list[Source] = []
    seen: set[str] = set()
    for source in list(existing) + list(new):
        key = source.url.strip() or f"{source.title}\x00{source.snippet}"
        if key in seen:
            continue
        seen.add(key)
        merged.append(source)
    return merged


def extract_sources(raw: Any, max_results: int) -> list[Source]:
    """Normalise a provider-specific search response into ``Source`` objects."""
    if isinstance(raw, dict):
        items = raw.get("results", [])
    elif isinstance(raw, list):
        items = raw
    else:
        items = []

    sources: list[Source] = []
    for item in items[:max_results]:
        if not isinstance(item, dict):
            continue
        content = item.get("content") or item.get("snippet") or ""
        sources.append(
            Source(
                title=str(item.get("title") or ""),
                url=str(item.get("url") or ""),
                snippet=str(content)[:4000],
            )
        )
    return sources
