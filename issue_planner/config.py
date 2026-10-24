"""Environment loading and shared defaults for the issue-planning agent."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# Load .env once when the package is imported so that the CLI and programmatic
# users observe the same configuration.
load_dotenv()

#: Directory shared by both entry points: ``agent.py`` writes its artifacts here
#: and ``viewer.py`` scans it, so a fresh plan is visible without extra flags.
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_MAX_ITERATIONS = 2

#: Total number of research sub-agent invocations per run. The first call is the
#: initial research pass; the remaining budget allows the planner to ask for
#: targeted follow-up research while addressing critic remarks.
DEFAULT_MAX_RESEARCH_CALLS = 2
#: Upper bound on focused search queries the planner may request at once.
MAX_FOLLOW_UP_QUERIES = 3

DEFAULT_MAX_SEARCH_RESULTS = 5
DEFAULT_MAX_RETRIES = 3
SUPPORTED_PROVIDERS = ("openai", "deepseek")

# DeepSeek's API defaults to an 8,192-token output ceiling. That is too small
# for detailed implementation plans: the response is truncated mid tool call,
# the JSON never parses, and LangChain silently returns no structured result.
# Raise the ceiling for DeepSeek; max_tokens only caps generation and does not
# reserve or bill the unused tokens.
DEEPSEEK_MAX_OUTPUT_TOKENS = 32768

# Environment variable that overrides the model for a provider.
MODEL_ENV_VARS = {
    "openai": "OPENAI_MODEL",
    "deepseek": "DEEPSEEK_MODEL",
}

# Strict allow-list of model IDs per provider. An unlisted model is treated as
# a configuration error and terminates the CLI with exit code 2.
#
# The OpenAI list starts at GPT-5 and includes every documented text/chat model
# released after it (GPT-4 and older are intentionally rejected). Access to a
# specific model still depends on the account's entitlements.
SUPPORTED_MODELS: dict[str, tuple[str, ...]] = {
    "openai": (
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
    ),
    "deepseek": (
        "deepseek-flash",
        "deepseek-v4-pro",
    ),
}

# Model used when neither --model nor the provider's environment variable is set.
DEFAULT_MODELS = {
    "openai": "gpt-5-mini",
    "deepseek": "deepseek-flash",
}
