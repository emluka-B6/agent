"""Tests for answering a plan's open questions on a resumed run."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from work_planner.models import (
    ImplementationPlan,
    OpenQuestion,
    PlanRevision,
    SubTask,
)
from work_planner.plan_editor import apply_revision
from work_planner.plan_repair import normalize_open_questions, normalize_plan
from work_planner.questions import (
    apply_answers,
    find_question,
    format_answers,
    format_questions,
    is_recommended,
    load_answer_file,
    parse_answer_arguments,
    pending_questions,
    prompt_answers,
    resolve_option,
)

ANSWERED_AT = "2026-10-31T10:00:00+00:00"


def make_question(**overrides: Any) -> OpenQuestion:
    data: dict[str, Any] = {
        "question": "Which database should the service use in production?",
        "options": ["SQLite", "PostgreSQL"],
        "recommended": "SQLite",
    }
    data.update(overrides)
    return OpenQuestion(**data)


def make_task() -> SubTask:
    return SubTask(
        id="T1",
        title="Add a token bucket",
        description="Introduce a token bucket limiter.",
        implementation_prompt="Add a token bucket limiter with tests.",
        acceptance_criteria=["Requests above the limit are rejected"],
        verification="pytest tests/test_ratelimit.py",
    )


def make_plan(*questions: OpenQuestion) -> ImplementationPlan:
    return ImplementationPlan(
        summary="Harden the public API",
        subtasks=[make_task()],
        open_questions=list(questions),
        definition_of_done=["Rate limiting is enabled"],
    )


# ---------------------------------------------------------------------------
# Schema compatibility
# ---------------------------------------------------------------------------
def test_a_plain_question_string_is_upgraded_to_a_question_object() -> None:
    plan = ImplementationPlan(
        summary="A plan",
        subtasks=[make_task()],
        open_questions=["Which database?"],
        definition_of_done=["Done"],
    )

    assert plan.open_questions == [OpenQuestion(question="Which database?")]


def test_a_missing_option_list_and_a_single_option_are_tolerated() -> None:
    assert OpenQuestion.model_validate(
        {"question": "Which database?"}
    ).options == []
    assert OpenQuestion.model_validate(
        {"question": "Which database?", "options": "SQLite", "recommended": None}
    ) == OpenQuestion(
        question="Which database?", options=["SQLite"], recommended=""
    )


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def test_normalize_open_questions_caps_and_cleans_the_options(
    capsys: pytest.CaptureFixture[str],
) -> None:
    question = make_question(
        question="  Which   database? ",
        options=["SQLite", " sqlite ", "", "PostgreSQL", "Oracle", "MySQL", "MariaDB"],
        recommended="MySQL",
    )

    normalized = normalize_open_questions([question])

    assert len(normalized) == 1
    assert normalized[0].question == "Which database?"
    assert normalized[0].options == ["SQLite", "PostgreSQL", "Oracle", "MySQL"]
    assert "more than 4 options" in capsys.readouterr().err


def test_normalize_open_questions_snaps_an_unknown_recommendation_to_the_first(
    capsys: pytest.CaptureFixture[str],
) -> None:
    normalized = normalize_open_questions(
        [make_question(options=["SQLite", "PostgreSQL"], recommended="Oracle")]
    )

    assert normalized[0].recommended == "SQLite"
    assert "not one of its options" in capsys.readouterr().err


def test_normalize_plan_drops_blank_and_repeated_questions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan = make_plan(
        make_question(),
        make_question(
            question=(
                "  which   DATABASE should the service use in production? "
            )
        ),
        make_question(question="   "),
    )

    normalized = normalize_plan(plan)

    assert [question.question for question in normalized.open_questions] == [
        "Which database should the service use in production?"
    ]
    assert "dropped a blank or repeated open question" in capsys.readouterr().err


def test_is_recommended_ignores_case_and_spacing() -> None:
    question = make_question(recommended=" sqlite ")

    assert is_recommended(question, "SQLite")
    assert not is_recommended(question, "PostgreSQL")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def test_format_questions_numbers_the_options_and_marks_the_recommended() -> None:
    rendered = format_questions(
        [
            make_question(),
            make_question(question="Should the API be public?", answer="No"),
        ]
    )

    assert rendered.splitlines() == [
        "Open questions (decisions only a human can make; the plan runs "
        "without an answer):",
        "  1. Which database should the service use in production?",
        "     1) SQLite (recommended)",
        "     2) PostgreSQL",
    ]


def test_format_questions_reports_a_question_without_options() -> None:
    rendered = format_questions([make_question(options=[], recommended="")])

    assert "     (no options suggested)" in rendered


def test_format_questions_is_empty_when_everything_is_answered() -> None:
    assert format_questions([make_question(answer="SQLite")]) == ""


def test_pending_questions_skips_answered_ones() -> None:
    pending = pending_questions(
        [make_question(), make_question(question="Other?", answer="Yes")]
    )

    assert [question.question for question in pending] == [
        "Which database should the service use in production?"
    ]


def test_format_answers_feeds_the_decisions_to_the_planner() -> None:
    plan = make_plan(make_question(answer="PostgreSQL", answered_at=ANSWERED_AT))

    assert format_answers(plan).splitlines() == [
        "- Which database should the service use in production?",
        "  Answer: PostgreSQL",
    ]
    assert format_answers(None) == "(none; no human answers were supplied)"
    assert format_answers(make_plan()) == "(none; no human answers were supplied)"


# ---------------------------------------------------------------------------
# Resolving answers
# ---------------------------------------------------------------------------
def test_resolve_option_accepts_a_number_an_option_text_or_free_text() -> None:
    question = make_question()

    assert resolve_option(question, "2") == "PostgreSQL"
    assert resolve_option(question, " postgresql ") == "PostgreSQL"
    assert resolve_option(question, "MySQL 8") == "MySQL 8"
    assert resolve_option(question, "99") == "99"
    assert resolve_option(question, "  ") == ""


def test_find_question_accepts_a_number_or_the_question_text() -> None:
    questions = [make_question(), make_question(question="Should it be public?")]

    match = find_question(questions, "2")
    assert match is not None
    index, question = match
    assert (index, question.question) == (2, "Should it be public?")

    match = find_question(
        questions, "which database should the service use in production"
    )
    assert match is not None
    assert match[0] == 1
    assert find_question(questions, "3") is None
    assert find_question(questions, "unknown") is None


# ---------------------------------------------------------------------------
# Answer sources
# ---------------------------------------------------------------------------
def test_parse_answer_arguments_requires_question_and_answer() -> None:
    assert parse_answer_arguments(["1=2", "2=PostgreSQL"]) == [
        ("1", "2"),
        ("2", "PostgreSQL"),
    ]

    for spec in ("1", "=2", "1="):
        with pytest.raises(ValueError, match="expected QUESTION=ANSWER"):
            parse_answer_arguments([spec])


def test_load_answer_file_reads_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "answers.json"
    path.write_text(json.dumps({"1": "2", "Why?": "Because"}), encoding="utf-8")

    assert load_answer_file(path) == [("1", "2"), ("Why?", "Because")]


def test_load_answer_file_reads_a_list_that_mirrors_the_artifact(
    tmp_path: Path,
) -> None:
    path = tmp_path / "answers.json"
    path.write_text(
        json.dumps([{"question": "1", "answer": "2"}, {"index": 2, "answer": 3}]),
        encoding="utf-8",
    )

    assert load_answer_file(path) == [("1", "2"), ("2", "3")]


def test_load_answer_file_reports_invalid_files(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not a valid answers file"):
        load_answer_file(broken)

    missing_key = tmp_path / "missing.json"
    missing_key.write_text(json.dumps([{"answer": "2"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="has no 'question' key"):
        load_answer_file(missing_key)

    missing_answer = tmp_path / "no-answer.json"
    missing_answer.write_text(json.dumps([{"question": "1"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="has no 'answer' value"):
        load_answer_file(missing_answer)

    scalar = tmp_path / "scalar.json"
    scalar.write_text("42", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON object or a list"):
        load_answer_file(scalar)


# ---------------------------------------------------------------------------
# Recording answers
# ---------------------------------------------------------------------------
def test_apply_answers_records_the_option_text_and_the_timestamp() -> None:
    plan = make_plan(make_question())

    answered = apply_answers(plan, [("1", "2")], answered_at=ANSWERED_AT)

    assert answered.open_questions[0].answer == "PostgreSQL"
    assert answered.open_questions[0].answered_at == ANSWERED_AT
    # The source plan is left untouched.
    assert plan.open_questions[0].answer is None


def test_apply_answers_matches_a_question_by_its_text() -> None:
    plan = make_plan(make_question())

    answered = apply_answers(
        plan,
        [("which database should the service use in production?", "PostgreSQL")],
        answered_at=ANSWERED_AT,
    )

    assert answered.open_questions[0].answer == "PostgreSQL"


def test_apply_answers_overwrites_an_earlier_answer() -> None:
    plan = make_plan(make_question(answer="SQLite", answered_at=ANSWERED_AT))

    answered = apply_answers(plan, [("1", "PostgreSQL")], answered_at=ANSWERED_AT)

    assert answered.open_questions[0].answer == "PostgreSQL"


def test_apply_answers_reports_an_unknown_question() -> None:
    with pytest.raises(ValueError, match="no open question matches '3'"):
        apply_answers(make_plan(make_question()), [("3", "PostgreSQL")])


def test_apply_answers_without_entries_returns_the_same_plan() -> None:
    plan = make_plan(make_question())

    assert apply_answers(plan, []) is plan


# ---------------------------------------------------------------------------
# Interactive prompt
# ---------------------------------------------------------------------------
def test_prompt_answers_writes_to_the_stream_and_skips_empty_answers() -> None:
    stream = io.StringIO()
    answers = iter(["", "2"])
    plan = make_plan(make_question(), make_question(question="Should it be public?"))

    entries = prompt_answers(plan, stream=stream, input_fn=lambda: next(answers))

    assert entries == [("2", "2")]
    rendered = stream.getvalue()
    assert "[1] Which database should the service use in production?" in rendered
    assert "1) SQLite (recommended)" in rendered
    assert "[2] Should it be public?" in rendered


def test_prompt_answers_stops_at_end_of_input() -> None:
    stream = io.StringIO()

    def end_of_input() -> str:
        raise EOFError

    entries = prompt_answers(
        make_plan(make_question()), stream=stream, input_fn=end_of_input
    )

    assert entries == []
    assert "answer [1-2, the option text" in stream.getvalue()


def test_prompt_answers_does_nothing_without_pending_questions() -> None:
    stream = io.StringIO()

    assert (
        prompt_answers(
            make_plan(make_question(answer="SQLite")),
            stream=stream,
            input_fn=lambda: "1",
        )
        == []
    )
    assert stream.getvalue() == ""


# ---------------------------------------------------------------------------
# Revisions keep decisions
# ---------------------------------------------------------------------------
def test_a_revision_keeps_answered_questions_the_planner_dropped() -> None:
    answered = make_question(answer="PostgreSQL", answered_at=ANSWERED_AT)
    plan = make_plan(answered)

    revision = PlanRevision(
        open_questions=[make_question(question="Should the API be public?")]
    )

    updated = apply_revision(plan, revision)

    assert [question.question for question in updated.open_questions] == [
        "Should the API be public?",
        "Which database should the service use in production?",
    ]
    assert updated.open_questions[-1].answer == "PostgreSQL"
    assert updated.open_questions[-1].answered_at == ANSWERED_AT


def test_a_revision_keeps_the_answer_on_a_rewritten_question() -> None:
    plan = make_plan(make_question(answer="PostgreSQL", answered_at=ANSWERED_AT))

    revision = PlanRevision(
        open_questions=[
            make_question(
                question=(
                    " which database should the service use in production? "
                ),
                options=["SQLite", "PostgreSQL"],
                recommended="SQLite",
            )
        ]
    )

    updated = apply_revision(plan, revision)

    assert updated.open_questions[0].answer == "PostgreSQL"
    assert updated.open_questions[0].answered_at == ANSWERED_AT
