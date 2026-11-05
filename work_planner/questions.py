"""Human answers for the open questions of a plan.

``open_questions`` are decisions that only a human can make, so the workflow
never waits for them: the planner records a recommended option and the plan
stays executable. This module turns those questions into something a human can
answer when a run is continued with ``agent.py --resume``:

- :func:`format_questions` renders the pending questions and their options for
  the terminal (Markdown rendering lives in :mod:`work_planner.output`);
- :func:`load_answer_file` and :func:`parse_answer_arguments` read answers from
  a JSON file or from repeated ``--answer QUESTION=ANSWER`` flags;
- :func:`prompt_answers` interviews the human on a terminal;
- :func:`apply_answers` records the answers on the plan, where the planner sees
  them (via :func:`format_answers`) and where later revisions keep them.

An answer is always recorded verbatim, so free text is as valid as one of the
suggested options. Answers never block the run: an unanswered question keeps
the recommended option.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TextIO

from work_planner.models import ImplementationPlan, OpenQuestion


def normalize_key(text: str) -> str:
    """Return a comparison key (case, punctuation, and spacing free)."""
    return re.sub(r"[\W_]+", " ", str(text).casefold()).strip()


def is_answered(question: OpenQuestion) -> bool:
    """True when a human already answered ``question``."""
    return bool((question.answer or "").strip())


def is_recommended(question: OpenQuestion, option: str) -> bool:
    """True when ``option`` is the option the plan already assumes."""
    if not question.recommended:
        return False
    return normalize_key(option) == normalize_key(question.recommended)


def pending_questions(
    questions: Iterable[OpenQuestion],
) -> list[OpenQuestion]:
    """Return the questions that still need a human answer."""
    return [question for question in questions if not is_answered(question)]


def numbered_questions(
    questions: Sequence[OpenQuestion],
) -> list[tuple[int, OpenQuestion]]:
    """Pair the unanswered questions with their one-based artifact number."""
    return [
        (index, question)
        for index, question in enumerate(questions, start=1)
        if not is_answered(question)
    ]


def format_questions(questions: Iterable[OpenQuestion]) -> str:
    """Render the pending questions and their options for the terminal.

    Each question keeps the number it has in the artifact, so the same number
    can be used with ``--answer NUMBER=ANSWER``. An empty string is returned
    when every question is answered.
    """
    entries = numbered_questions(list(questions))
    if not entries:
        return ""

    lines = [
        "Open questions (decisions only a human can make; the plan runs "
        "without an answer):"
    ]
    for index, question in entries:
        lines.append(f"  {index}. {question.question}")
        if question.options:
            for option_index, option in enumerate(question.options, start=1):
                marker = (
                    " (recommended)" if is_recommended(question, option) else ""
                )
                lines.append(f"     {option_index}) {option}{marker}")
        else:
            lines.append("     (no options suggested)")
    return "\n".join(lines)


def format_answers(plan: ImplementationPlan | None) -> str:
    """Render the human decisions of a plan for the planner prompt."""
    if plan is None:
        return "(none; no human answers were supplied)"
    answered = [
        question for question in plan.open_questions if is_answered(question)
    ]
    if not answered:
        return "(none; no human answers were supplied)"
    lines: list[str] = []
    for question in answered:
        lines.append(f"- {question.question}")
        lines.append(f"  Answer: {question.answer}")
    return "\n".join(lines)


def resolve_option(question: OpenQuestion, raw: str) -> str:
    """Return the recorded answer for ``raw``.

    ``raw`` may be the option number, the option text (case and spacing are
    ignored), or free text, which is kept verbatim.
    """
    text = str(raw).strip()
    if not text:
        return ""
    if text.isdigit():
        index = int(text)
        if 1 <= index <= len(question.options):
            return question.options[index - 1]
    for option in question.options:
        if normalize_key(option) == normalize_key(text):
            return option
    return text


def find_question(
    questions: Sequence[OpenQuestion], key: str
) -> tuple[int, OpenQuestion] | None:
    """Resolve a question key to its one-based number and the question.

    A key is either the number shown in the artifact or the question text
    (matched ignoring case, punctuation, and spacing).
    """
    text = str(key).strip()
    if not text:
        return None
    if text.isdigit():
        index = int(text)
        if 1 <= index <= len(questions):
            return index, questions[index - 1]
        return None
    wanted = normalize_key(text)
    for index, question in enumerate(questions, start=1):
        if normalize_key(question.question) == wanted:
            return index, question
    return None


def parse_answer_arguments(specs: Iterable[str]) -> list[tuple[str, str]]:
    """Parse ``--answer QUESTION=ANSWER`` values into key/answer pairs."""
    entries: list[tuple[str, str]] = []
    for spec in specs:
        text = str(spec)
        key, separator, answer = text.partition("=")
        if not separator or not key.strip() or not answer.strip():
            raise ValueError(
                f"invalid --answer {text!r}; expected QUESTION=ANSWER, for "
                "example --answer 1=2 or --answer 2=PostgreSQL."
            )
        entries.append((key.strip(), answer.strip()))
    return entries


def load_answer_file(path: Path | str) -> list[tuple[str, str]]:
    """Read answers from a JSON file.

    Two shapes are accepted::

        {"1": "2", "Which database?": "PostgreSQL"}
        [{"question": "Which database?", "answer": "PostgreSQL"}]

    A key is the question number shown in the artifact or the question text.
    """
    file_path = Path(path).expanduser()
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{file_path.name} is not a valid answers file: {exc}"
        ) from exc
    except OSError as exc:
        raise OSError(f"could not read {path}: {exc}") from exc

    entries: list[tuple[str, str]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            entries.append((_answer_key(key), _answer_value(value, key)))
    elif isinstance(payload, list):
        for position, item in enumerate(payload, start=1):
            if not isinstance(item, dict):
                raise ValueError(
                    f"{file_path.name}: answer {position} must be an object with "
                    "'question' (or 'index') and 'answer'."
                )
            key = item.get("question", item.get("index"))
            if key is None:
                raise ValueError(
                    f"{file_path.name}: answer {position} has no 'question' key."
                )
            if "answer" not in item:
                raise ValueError(
                    f"{file_path.name}: answer {position} has no 'answer' value."
                )
            entries.append(
                (_answer_key(key), _answer_value(item["answer"], key))
            )
    else:
        raise ValueError(
            f"{file_path.name}: expected a JSON object or a list of answers."
        )
    return entries


def _answer_key(key: Any) -> str:
    text = str(key).strip()
    if not text:
        raise ValueError("an answer key must not be empty")
    return text


def _answer_value(value: Any, key: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(
            f"the answer for {str(key)!r} must be a string or an integer."
        )
    text = str(value).strip()
    if not text:
        raise ValueError(f"the answer for {str(key)!r} must not be empty.")
    return text


def apply_answers(
    plan: ImplementationPlan,
    entries: Iterable[tuple[str, str]],
    *,
    answered_at: str | None = None,
) -> ImplementationPlan:
    """Record answers on a copy of ``plan`` and return it.

    Each entry maps a question key (number or text) to an answer. Unknown keys
    raise :class:`ValueError` so a typo is reported instead of silently
    dropping a decision.
    """
    pairs = list(entries)
    if not pairs:
        return plan

    timestamp = answered_at or datetime.now(timezone.utc).isoformat()
    questions = list(plan.open_questions)
    unknown: list[str] = []
    for key, raw in pairs:
        match = find_question(plan.open_questions, key)
        if match is None:
            unknown.append(key)
            continue
        index, question = match
        answer = resolve_option(question, raw)
        if not answer:
            continue
        questions[index - 1] = question.model_copy(
            update={"answer": answer, "answered_at": timestamp}
        )
    if unknown:
        raise ValueError(
            "no open question matches "
            + ", ".join(repr(key) for key in unknown)
            + "; pass the question number shown in the artifact or its exact text."
        )
    return plan.model_copy(update={"open_questions": questions})


def prompt_answers(
    plan: ImplementationPlan,
    *,
    stream: TextIO | None = None,
    input_fn: Callable[[], str] | None = None,
) -> list[tuple[str, str]]:
    """Interview the human for every pending question.

    Prompts are written to ``stream`` (stderr by default) so that stdout keeps
    carrying only the artifact path. An empty answer skips the question; EOF
    and Ctrl-C stop the interview and keep the answers given so far.
    """
    entries = numbered_questions(plan.open_questions)
    if not entries:
        return []

    output = stream or sys.stderr
    reader = input_fn or input
    answers: list[tuple[str, str]] = []
    for index, question in entries:
        print(f"\n[{index}] {question.question}", file=output)
        for option_index, option in enumerate(question.options, start=1):
            marker = " (recommended)" if is_recommended(question, option) else ""
            print(f"  {option_index}) {option}{marker}", file=output)
        hint = (
            f"1-{len(question.options)}, the option text, free text, "
            "or Enter to skip"
            if question.options
            else "free text, or Enter to skip"
        )
        print(f"  answer [{hint}]: ", file=output, end="", flush=True)
        try:
            raw = reader()
        except (EOFError, KeyboardInterrupt):
            print(file=output)
            break
        if raw.strip():
            answers.append((str(index), raw))
    return answers
