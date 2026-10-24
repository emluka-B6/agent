"""Chat-model construction and structured-output invocation helpers."""

from __future__ import annotations

import os
import sys
import textwrap
import time
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableLambda
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from issue_planner.config import (
    DEEPSEEK_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODELS,
    MODEL_ENV_VARS,
    SUPPORTED_MODELS,
    SUPPORTED_PROVIDERS,
)


class StructuredOutputError(RuntimeError):
    """The model call did not produce the requested structured result."""


def normalize_provider(provider: str) -> str:
    """Return the canonical provider key, or raise ValueError."""
    normalized = provider.strip().lower()
    if normalized not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported provider '{provider}'. "
            f"Choose one of: {', '.join(SUPPORTED_PROVIDERS)}."
        )
    return normalized


def resolve_model_name(provider: str, model_name: str | None = None) -> str:
    """Return a validated model ID for the provider.

    ``model_name`` takes precedence over the provider-specific environment
    variable, which in turn takes precedence over the built-in default. An
    unknown model raises ``ValueError`` so callers can stop execution.
    """
    provider = normalize_provider(provider)

    if model_name is not None:
        requested = model_name
        origin = "--model"
    else:
        env_var = MODEL_ENV_VARS[provider]
        env_value = os.getenv(env_var)
        if env_value is not None:
            requested = env_value
            origin = env_var
        else:
            requested = DEFAULT_MODELS[provider]
            origin = "the provider default"

    candidate = requested.strip().lower()
    allowed = SUPPORTED_MODELS[provider]
    if candidate not in allowed:
        raise ValueError(
            f"Unsupported model '{requested}' for provider '{provider}' "
            f"(from {origin}). Supported models: {', '.join(allowed)}. "
            "Run --list-models to see the complete list."
        )
    return candidate


def format_model_catalog() -> str:
    """Return a human-readable list of supported models for every provider."""
    lines = ["Supported models:"]
    for provider in SUPPORTED_PROVIDERS:
        lines.append(f"  {provider} (default: {DEFAULT_MODELS[provider]}):")
        lines.append(
            textwrap.fill(
                ", ".join(SUPPORTED_MODELS[provider]),
                width=78,
                initial_indent="    ",
                subsequent_indent="    ",
            )
        )
    return "\n".join(lines)


def build_model(provider: str, model_name: str | None = None) -> BaseChatModel:
    """Create the configured chat model from the provider allow-list."""
    provider = normalize_provider(provider)
    model = resolve_model_name(provider, model_name)
    if provider == "openai":
        return ChatOpenAI(model=model, temperature=0)
    if provider == "deepseek":
        return ChatDeepSeek(
            model=model,
            temperature=0,
            # The provider default truncates detailed plans. A larger cap
            # prevents malformed tool-call JSON without reserving tokens.
            max_tokens=DEEPSEEK_MAX_OUTPUT_TOKENS,
            # DeepSeek's thinking mode rejects the forced tool_choice that
            # LangChain uses for function-calling structured output. The
            # workflow depends on structured output, so use non-thinking mode.
            extra_body={"thinking": {"type": "disabled"}},
        )
    raise ValueError(
        f"Unsupported provider '{provider}'. Choose one of: {', '.join(SUPPORTED_PROVIDERS)}."
    )


def _raw_details(raw: Any) -> str:
    """Summarise a raw chat response so failures are diagnosable."""
    if raw is None:
        return "no raw response"
    metadata = getattr(raw, "response_metadata", None) or {}
    usage = getattr(raw, "usage_metadata", None) or {}
    invalid_calls = getattr(raw, "invalid_tool_calls", None) or []

    details: list[str] = []
    finish_reason = metadata.get("finish_reason") or metadata.get("stop_reason")
    if finish_reason:
        details.append(f"finish_reason={finish_reason}")
    output_tokens = usage.get("output_tokens")
    if output_tokens is not None:
        details.append(f"output_tokens={output_tokens}")
    if invalid_calls:
        details.append(f"invalid_tool_calls={len(invalid_calls)}")
    return ", ".join(details) or "no response metadata"


def _extract_parsed(schema: type[BaseModel], result: Any) -> Any:
    """Return the parsed model or raise a diagnostic error.

    ``with_structured_output(..., include_raw=True)`` returns ``None`` as the
    parsed value when the provider sends no tool call or sends malformed JSON
    (for example, a response truncated by the output-token limit). Passing that
    ``None`` to Pydantic would only produce a misleading ``model_type`` error,
    so report what actually happened and let the retry helper run again.
    """
    if isinstance(result, dict) and "parsed" in result:
        parsed = result.get("parsed")
        if parsed is not None:
            return parsed
        parsing_error = result.get("parsing_error")
        message = (
            f"the model returned no valid {schema.__name__} "
            f"({_raw_details(result.get('raw'))})"
        )
        if parsing_error is not None:
            message = f"{message}: {parsing_error}"
        raise StructuredOutputError(message)
    if result is None:
        raise StructuredOutputError(
            f"the model returned no valid {schema.__name__} (empty response)"
        )
    return as_model(schema, result)


def _structured_output(
    model: BaseChatModel,
    schema: type[BaseModel],
    provider: str,
    method: str | None = None,
) -> Any:
    """Return a runnable that produces the requested Pydantic schema."""
    if method is None:
        method = "function_calling" if provider == "deepseek" else "json_schema"
    try:
        runnable = model.with_structured_output(
            schema, method=method, include_raw=True
        )
    except TypeError:
        # Older LangChain versions may not accept the method argument.
        runnable = model.with_structured_output(schema)
    return runnable | RunnableLambda(lambda result: _extract_parsed(schema, result))


def invoke_structured(
    model: BaseChatModel,
    schema: type[BaseModel],
    provider: str,
    messages: Any,
    label: str,
    max_retries: int,
) -> Any:
    """Invoke structured output, falling back to function calling if needed."""
    methods = (
        ["function_calling"]
        if provider == "deepseek"
        else ["json_schema", "function_calling"]
    )
    last_error: Exception | None = None

    for index, method in enumerate(methods):
        try:
            runnable = _structured_output(model, schema, provider, method=method)
            return invoke_with_retry(
                runnable, messages, f"{label} [{method}]", max_retries
            )
        except Exception as exc:  # noqa: BLE001 - try the next structured mode
            last_error = exc
            if index < len(methods) - 1:
                print(
                    f"[{label}] structured output via {method} failed: {exc} "
                    "Trying function calling instead.",
                    file=sys.stderr,
                )

    raise RuntimeError(
        f"{label} failed with every structured-output method: {last_error}"
    ) from last_error


def as_model(schema: type[Any], value: Any) -> Any:
    """Coerce a structured-output result into the requested Pydantic model."""
    if value is None:
        raise StructuredOutputError(
            f"the model returned no valid {getattr(schema, '__name__', schema)} "
            "(empty response)"
        )
    if isinstance(value, schema):
        return value
    return schema.model_validate(value)


def invoke_with_retry(
    runnable: Any, payload: Any, label: str, max_retries: int = DEFAULT_MAX_RETRIES
) -> Any:
    """Invoke a runnable with a small exponential backoff."""
    attempts = max(1, max_retries)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            return runnable.invoke(payload)
        except Exception as exc:  # noqa: BLE001 - retry any provider/transport error
            last_error = exc
            if attempt >= attempts:
                break
            delay = min(2 ** (attempt - 1), 8)
            print(
                f"[{label}] attempt {attempt}/{attempts} failed: {exc}. "
                f"Retrying in {delay}s.",
                file=sys.stderr,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"{label} failed after {attempts} attempt(s): {last_error}"
    ) from last_error
