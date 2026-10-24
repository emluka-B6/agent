"""Tests for the provider model allow-lists, validation, and CLI behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.runnables import RunnableLambda

from agent import main
from issue_planner.config import (
    DEEPSEEK_MAX_OUTPUT_TOKENS,
    DEFAULT_MODELS,
    MODEL_ENV_VARS,
    SUPPORTED_MODELS,
    SUPPORTED_PROVIDERS,
)
from issue_planner.llm import (
    StructuredOutputError,
    as_model,
    build_model,
    format_model_catalog,
    invoke_structured,
    normalize_provider,
    resolve_model_name,
)
from issue_planner.models import ImplementationPlan, SubTask

EXPECTED_OPENAI_MODELS = {
    "gpt-5",
    "gpt-5-chat-latest",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-pro",
    "gpt-5-codex",
    "gpt-5.1",
    "gpt-5.1-chat-latest",
    "gpt-5.1-codex",
    "gpt-5.1-codex-mini",
    "gpt-5.1-codex-max",
    "gpt-5.2",
    "gpt-5.2-chat-latest",
    "gpt-5.2-pro",
    "gpt-5.2-codex",
    "gpt-5.3-chat-latest",
    "gpt-5.3-codex",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.4-pro",
    "gpt-5.5",
    "gpt-5.5-pro",
    "gpt-5.6-cyber",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "gpt-6-luna",
    "gpt-6-sol",
    "gpt-6.1-sol",
}


def test_registry_matches_the_documented_models() -> None:
    assert tuple(SUPPORTED_MODELS) == SUPPORTED_PROVIDERS
    assert set(SUPPORTED_MODELS["openai"]) == EXPECTED_OPENAI_MODELS
    assert SUPPORTED_MODELS["deepseek"] == ("deepseek-flash", "deepseek-v4-pro")


@pytest.mark.parametrize("provider", SUPPORTED_PROVIDERS)
def test_every_provider_has_a_supported_default(provider: str) -> None:
    assert DEFAULT_MODELS[provider] in SUPPORTED_MODELS[provider]


def test_openai_allow_list_starts_at_gpt_5() -> None:
    assert SUPPORTED_MODELS["openai"][0] == "gpt-5"
    assert not any(
        model.startswith(("gpt-3", "gpt-4")) for model in SUPPORTED_MODELS["openai"]
    )


@pytest.mark.parametrize("provider", SUPPORTED_PROVIDERS)
def test_resolve_model_name_uses_the_provider_default(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    monkeypatch.delenv(MODEL_ENV_VARS[provider], raising=False)
    assert resolve_model_name(provider) == DEFAULT_MODELS[provider]


def test_resolve_model_name_reads_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    assert resolve_model_name("openai") == "gpt-5.4-mini"


def test_resolve_model_name_prefers_the_explicit_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    assert resolve_model_name("openai", "gpt-5") == "gpt-5"


def test_resolve_model_name_is_case_insensitive() -> None:
    assert resolve_model_name("OpenAI", "GPT-5.1") == "gpt-5.1"


def test_resolve_model_name_rejects_an_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unsupported model 'gpt-4o-mini'"):
        resolve_model_name("openai", "gpt-4o-mini")


def test_resolve_model_name_rejects_a_model_from_another_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported model"):
        resolve_model_name("deepseek", "gpt-5")


def test_normalize_provider_rejects_an_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported provider"):
        normalize_provider("anthropic")


def test_normalize_provider_trims_and_lowercases() -> None:
    assert normalize_provider(" OpenAI ") == "openai"


def test_build_model_validates_before_creating_a_client() -> None:
    with pytest.raises(ValueError, match="Unsupported model"):
        build_model("openai", "gpt-4o-mini")


def test_build_model_normalizes_the_model_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    model = build_model("openai", "GPT-5.4-Mini")

    assert model.model_name == "gpt-5.4-mini"


def test_build_model_disables_thinking_for_deepseek(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    model = build_model("deepseek", "deepseek-v4-pro")

    assert model.model_name == "deepseek-v4-pro"
    assert model.max_tokens == DEEPSEEK_MAX_OUTPUT_TOKENS
    assert model.extra_body == {"thinking": {"type": "disabled"}}


def test_model_catalog_lists_every_supported_model() -> None:
    catalog = format_model_catalog()

    for provider, models in SUPPORTED_MODELS.items():
        assert provider in catalog
        assert f"default: {DEFAULT_MODELS[provider]}" in catalog
        for model in models:
            assert model in catalog


def test_cli_lists_models_without_an_issue(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--list-models"]) == 0

    captured = capsys.readouterr()
    assert "Supported models:" in captured.out
    assert "deepseek-v4-pro" in captured.out
    assert "gpt-6.1-sol" in captured.out
    assert captured.err == ""


def test_cli_terminates_on_an_unsupported_model(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["Add rate limiting", "--model", "gpt-4o-mini"]) == 2

    captured = capsys.readouterr()
    assert "Unsupported model 'gpt-4o-mini'" in captured.err
    assert "gpt-5" in captured.err


def test_cli_terminates_on_an_unsupported_env_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    assert main(["Add rate limiting"]) == 2

    captured = capsys.readouterr()
    assert "OPENAI_MODEL" in captured.err
    assert "Unsupported model 'gpt-4o-mini'" in captured.err


# ---------------------------------------------------------------------------
# Structured output handling
# ---------------------------------------------------------------------------
def make_plan() -> ImplementationPlan:
    return ImplementationPlan(
        summary="Add rate limiting",
        subtasks=[
            SubTask(
                id="T1",
                title="Add a limiter",
                description="Add a per-key rate limiter.",
                implementation_prompt="Add a per-key token bucket limiter with tests.",
                acceptance_criteria=["Requests above the limit return HTTP 429"],
                verification="pytest tests/test_ratelimit.py",
            )
        ],
        definition_of_done=["Rate limiting is enforced"],
    )


class _FakeStructuredModel:
    """Stand-in for the structured-output part of a chat model."""

    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []
        self.response_calls = 0

    def with_structured_output(
        self,
        schema: type[ImplementationPlan],
        *,
        method: str | None = None,
        include_raw: bool = False,
        **_kwargs: object,
    ) -> RunnableLambda:
        self.calls.append(
            {"schema": schema, "method": method, "include_raw": include_raw}
        )

        def respond(_messages: object) -> object:
            self.response_calls += 1
            return self.result

        return RunnableLambda(respond)


def test_invoke_structured_accepts_the_parsed_model() -> None:
    plan = make_plan()
    model = _FakeStructuredModel(
        {"raw": None, "parsed": plan, "parsing_error": None}
    )

    result = invoke_structured(
        model,  # type: ignore[arg-type]
        ImplementationPlan,
        "openai",
        [],
        "plan generation",
        max_retries=1,
    )

    assert result is plan
    assert model.calls == [
        {"schema": ImplementationPlan, "method": "json_schema", "include_raw": True}
    ]


def test_invoke_structured_explains_a_truncated_response() -> None:
    raw = SimpleNamespace(
        response_metadata={"finish_reason": "length"},
        usage_metadata={"output_tokens": 8192},
        invalid_tool_calls=[{"name": "ImplementationPlan", "args": "{"}],
    )
    model = _FakeStructuredModel(
        {"raw": raw, "parsed": None, "parsing_error": None}
    )

    with pytest.raises(RuntimeError, match="no valid ImplementationPlan") as excinfo:
        invoke_structured(
            model,  # type: ignore[arg-type]
            ImplementationPlan,
            "deepseek",
            [],
            "plan generation",
            max_retries=1,
        )

    message = str(excinfo.value)
    assert "finish_reason=length" in message
    assert "output_tokens=8192" in message
    assert "invalid_tool_calls=1" in message
    assert len(model.calls) == 1


def test_invoke_structured_retries_a_missing_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("issue_planner.llm.time.sleep", lambda _delay: None)
    raw = SimpleNamespace(
        response_metadata={"finish_reason": "tool_calls"},
        usage_metadata={"output_tokens": 100},
        invalid_tool_calls=[],
    )
    model = _FakeStructuredModel(
        {"raw": raw, "parsed": None, "parsing_error": None}
    )

    with pytest.raises(RuntimeError, match="no valid ImplementationPlan"):
        invoke_structured(
            model,  # type: ignore[arg-type]
            ImplementationPlan,
            "deepseek",
            [],
            "plan generation",
            max_retries=2,
        )

    assert len(model.calls) == 1
    assert model.response_calls == 2


def test_as_model_rejects_a_missing_structured_result() -> None:
    with pytest.raises(StructuredOutputError, match="no valid ImplementationPlan"):
        as_model(ImplementationPlan, None)
