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
