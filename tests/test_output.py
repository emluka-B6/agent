"""Tests for artifact naming and persistence."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any

import pytest

from issue_planner.config import DEFAULT_OUTPUT_DIR
from issue_planner.output import unique_output_path, write_output


def make_document(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "issue": "Add rate limiting to the public API",
        "generated_at": "2026-10-18T09:31:00+00:00",
        "status": "approved",
        "subtasks": [],
    }
    document.update(overrides)
    return document


def test_json_output_name_is_hashed(tmp_path: Path) -> None:
    path, output_format = unique_output_path(make_document(), tmp_path, "json")

    assert output_format == "json"
    assert path.parent == tmp_path
    assert re.fullmatch(r"output-[0-9a-f]{12}\.json", path.name)


def test_markdown_output_name_is_hashed(tmp_path: Path) -> None:
    path, output_format = unique_output_path(
        make_document(), tmp_path, "markdown"
    )

    assert output_format == "markdown"
    assert re.fullmatch(r"output-[0-9a-f]{12}\.md", path.name)


def test_auto_format_falls_back_to_json(tmp_path: Path) -> None:
    path, output_format = unique_output_path(make_document(), tmp_path, "auto")

    assert (output_format, path.suffix) == ("json", ".json")


def test_default_output_directory_is_shared_with_the_viewer() -> None:
    path, _ = unique_output_path(make_document())

    assert path.parent == DEFAULT_OUTPUT_DIR


def test_cli_defaults_to_the_shared_output_directory() -> None:
    from agent import build_parser

    args = build_parser().parse_args(["Add rate limiting"])

    assert args.output_dir == DEFAULT_OUTPUT_DIR


def test_repeated_runs_do_not_overwrite_previous_artifacts(
    tmp_path: Path,
) -> None:
    first, first_format = unique_output_path(make_document(), tmp_path, "json")
    write_output(make_document(), first, first_format)
    first_content = first.read_text(encoding="utf-8")

    second, second_format = unique_output_path(
        make_document(), tmp_path, "json"
    )
    write_output(
        make_document(status="needs_improvement"), second, second_format
    )

    assert first != second
    assert first.read_text(encoding="utf-8") == first_content
    assert json.loads(second.read_text(encoding="utf-8"))["status"] == (
        "needs_improvement"
    )


def test_resamples_when_the_hash_is_already_taken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = make_document()
    fixed = uuid.UUID(int=0)
    monkeypatch.setattr("issue_planner.output.uuid.uuid4", lambda: fixed)

    seed = "\x00".join(
        (document["issue"], document["generated_at"], fixed.hex, "0")
    )
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
    taken = tmp_path / f"output-{digest}.json"
    taken.write_text("{}", encoding="utf-8")

    path, _ = unique_output_path(document, tmp_path, "json")

    assert path != taken
    assert not path.exists()
