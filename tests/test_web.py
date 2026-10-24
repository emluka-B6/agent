"""Tests for the FastAPI viewer/editor API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from issue_planner.config import DEFAULT_OUTPUT_DIR
from issue_planner.models import ImplementationPlan, SubTask
from issue_planner.web.server import create_app
from viewer import build_parser as build_viewer_parser


def make_task(
    task_id: str, depends_on: tuple[str, ...] = ()
) -> SubTask:
    return SubTask(
        id=task_id,
        title=f"Task {task_id}",
        description=f"Description for {task_id}",
        implementation_prompt=f"Prompt for {task_id}",
        depends_on=list(depends_on),
        files=["agent.py"],
        acceptance_criteria=[f"{task_id} works"],
        verification="pytest",
    )


def sample_document() -> dict[str, Any]:
    plan = ImplementationPlan(
        summary="A plan",
        assumptions=["Python 3.10"],
        subtasks=[
            make_task("T1"),
            make_task("T2", ("T1",)),
            make_task("T3", ("T2",)),
        ],
        open_questions=[],
        definition_of_done=["Everything works"],
    )
    return {
        "issue": "Add rate limiting",
        "generated_at": "2026-10-17T00:00:00+00:00",
        "status": "approved",
        "approved": True,
        "review": {
            "approved": True,
            "summary": "Looks good",
            "blocking_issues": [],
            "suggestions": [],
        },
        "research": {"summary": None, "sources": [], "error": None},
        "metadata": {
            "provider": "openai",
            "model": "gpt-5-mini",
            "evaluations": 1,
            "max_iterations": 2,
        },
        **plan.model_dump(),
    }


@pytest.fixture()
def plans_dir(tmp_path: Path) -> Path:
    (tmp_path / "output.json").write_text(
        json.dumps(sample_document()), encoding="utf-8"
    )
    return tmp_path


@pytest.fixture()
def client(plans_dir: Path) -> TestClient:
    return TestClient(create_app(plans_dir))


def test_viewer_defaults_to_the_agent_output_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PLANS_DIR", raising=False)

    args = build_viewer_parser().parse_args([])
    app = create_app()

    assert args.plans_dir == DEFAULT_OUTPUT_DIR
    assert app.state.plans_dir == DEFAULT_OUTPUT_DIR.expanduser().resolve()


def test_viewer_still_honours_the_plans_dir_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PLANS_DIR", str(tmp_path))

    args = build_viewer_parser().parse_args([])

    assert args.plans_dir == tmp_path


def test_lists_plan_artifacts(client: TestClient) -> None:
    response = client.get("/api/plans")

    assert response.status_code == 200
    plans = response.json()["plans"]
    assert [plan["name"] for plan in plans] == ["output.json"]
    assert plans[0]["size"] > 0
    # The viewer identifies plans by their issue text, not the file name.
    assert plans[0]["title"] == "Add rate limiting"
    assert plans[0]["generated_at"] == "2026-10-17T00:00:00+00:00"
    assert plans[0]["status"] == "approved"


def test_list_falls_back_to_file_name_for_invalid_json(tmp_path: Path) -> None:
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    client = TestClient(create_app(tmp_path))

    response = client.get("/api/plans")

    assert response.status_code == 200
    plan = response.json()["plans"][0]
    assert plan["name"] == "broken.json"
    assert plan["title"] == "broken"
    assert plan["generated_at"] is None
    assert plan["status"] is None


def test_returns_an_empty_list_for_an_empty_directory(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    client = TestClient(create_app(empty))

    assert client.get("/api/plans").json() == {"plans": []}


def test_loads_a_plan(client: TestClient) -> None:
    response = client.get("/api/plans/output.json")

    assert response.status_code == 200
    document = response.json()
    assert document["issue"] == "Add rate limiting"
    assert [task["id"] for task in document["subtasks"]] == ["T1", "T2", "T3"]


def test_unknown_plan_returns_404(client: TestClient) -> None:
    response = client.get("/api/plans/missing.json")

    assert response.status_code == 404


def test_rejects_path_traversal_and_non_json_names(client: TestClient) -> None:
    assert client.get("/api/plans/notes.txt").status_code == 400
    # A slash cannot be part of the ``name`` path parameter at all.
    assert client.get("/api/plans/nested/output.json").status_code == 404
    # An encoded separator must not escape the plans directory.
    escaped = client.get("/api/plans/%2E%2E%2Foutput.json")
    assert escaped.status_code in (400, 404)


def test_returns_the_dependency_graph(client: TestClient) -> None:
    response = client.get("/api/plans/output.json/graph")

    assert response.status_code == 200
    graph = response.json()
    assert [node["id"] for node in graph["nodes"]] == ["T1", "T2", "T3"]
    assert graph["edges"] == [
        {"id": "T1->T2", "source": "T1", "target": "T2"},
        {"id": "T2->T3", "source": "T2", "target": "T3"},
    ]


def test_remove_rewires_dependencies(client: TestClient) -> None:
    response = client.post(
        "/api/plans/output.json/remove",
        json={"ids": ["T2"], "strategy": "rewire"},
    )

    assert response.status_code == 200
    document = response.json()
    assert [task["id"] for task in document["subtasks"]] == ["T1", "T3"]
    assert document["subtasks"][1]["depends_on"] == ["T1"]
    assert document["metadata"]["edits"][-1]["operation"] == "remove"
    assert document["metadata"]["edits"][-1]["strategy"] == "rewire"


def test_remove_accepts_an_unsaved_document(client: TestClient) -> None:
    document = sample_document()
    document["summary"] = "Edited in the browser"

    response = client.post(
        "/api/plans/output.json/remove",
        json={"ids": ["T1"], "strategy": "rewire", "document": document},
    )

    assert response.status_code == 200
    updated = response.json()
    assert updated["summary"] == "Edited in the browser"
    assert [task["id"] for task in updated["subtasks"]] == ["T2", "T3"]


def test_remove_block_returns_blockers(client: TestClient) -> None:
    response = client.post(
        "/api/plans/output.json/remove",
        json={"ids": ["T2"], "strategy": "block"},
    )

    assert response.status_code == 409
    body = response.json()
    assert body["blockers"] == ["T3"]
    assert "T3 depends on T2" in body["detail"]


def test_remove_cascade_deletes_dependents(client: TestClient) -> None:
    response = client.post(
        "/api/plans/output.json/remove",
        json={"ids": ["T2"], "strategy": "cascade"},
    )

    assert response.status_code == 200
    assert [task["id"] for task in response.json()["subtasks"]] == ["T1"]


def test_remove_unknown_ids_returns_409(client: TestClient) -> None:
    response = client.post(
        "/api/plans/output.json/remove",
        json={"ids": ["T9"], "strategy": "rewire"},
    )

    assert response.status_code == 409
    assert response.json()["unknown_ids"] == ["T9"]


def test_remove_requires_at_least_one_id(client: TestClient) -> None:
    response = client.post(
        "/api/plans/output.json/remove",
        json={"ids": [], "strategy": "rewire"},
    )

    assert response.status_code == 422


def test_save_persists_the_edited_document(
    client: TestClient, plans_dir: Path
) -> None:
    document = sample_document()
    document["subtasks"] = document["subtasks"][:2]

    response = client.post(
        "/api/plans/output.json/save", json={"document": document}
    )

    assert response.status_code == 200
    assert response.json()["format"] == "json"
    saved = json.loads((plans_dir / "output.json").read_text(encoding="utf-8"))
    assert [task["id"] for task in saved["subtasks"]] == ["T1", "T2"]
    assert saved["metadata"]["saved_at"]


def test_save_rejects_invalid_documents(client: TestClient) -> None:
    response = client.post(
        "/api/plans/output.json/save", json={"document": {"issue": "x"}}
    )

    assert response.status_code == 422


def test_export_markdown(client: TestClient) -> None:
    response = client.post(
        "/api/plans/output.json/export",
        json={"document": sample_document(), "format": "markdown"},
    )

    assert response.status_code == 200
    assert response.headers["content-disposition"] == (
        'attachment; filename="output.md"'
    )
    assert "# Implementation plan" in response.text
    assert "### 1. T1 - Task T1" in response.text


def test_export_json(client: TestClient) -> None:
    response = client.post(
        "/api/plans/output.json/export",
        json={"document": sample_document(), "format": "json"},
    )

    assert response.status_code == 200
    assert response.headers["content-disposition"] == (
        'attachment; filename="output.json"'
    )
    assert json.loads(response.text)["issue"] == "Add rate limiting"


def test_unknown_api_route_returns_404(client: TestClient) -> None:
    assert client.get("/api/does-not-exist").status_code == 404


def test_invalid_json_file_returns_422(tmp_path: Path) -> None:
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    client = TestClient(create_app(tmp_path))

    response = client.get("/api/plans/broken.json")

    assert response.status_code == 422
