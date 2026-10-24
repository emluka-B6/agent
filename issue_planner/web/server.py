"""FastAPI application that serves and edits generated plan artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from issue_planner.config import DEFAULT_OUTPUT_DIR
from issue_planner.output import render_markdown, write_output
from issue_planner.plan_editor import (
    RemovalError,
    apply_plan_to_document,
    plan_from_document,
    remove_subtasks,
)

#: Built React assets (``frontend`` runs ``vite build`` into this directory).
DIST_DIR = Path(__file__).resolve().parent / "dist"

DEV_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


class RemoveRequest(BaseModel):
    """Remove one or more subtasks from the supplied (or on-disk) document."""

    ids: list[str] = Field(min_length=1)
    strategy: Literal["rewire", "block", "cascade"] = "rewire"
    document: dict[str, Any] | None = None


class SaveRequest(BaseModel):
    """Persist the edited document back to the artifact on disk."""

    document: dict[str, Any]


class ExportRequest(BaseModel):
    """Render the edited document for download."""

    document: dict[str, Any]
    format: Literal["json", "markdown"] = "markdown"


def _safe_plan_path(plans_dir: Path, name: str) -> Path:
    """Resolve a plan file name without allowing path traversal."""
    candidate = Path(name)
    if candidate.name != name or candidate.suffix.lower() != ".json":
        raise HTTPException(
            status_code=400,
            detail="Plan name must be a plain .json file name.",
        )
    path = (plans_dir / candidate).resolve()
    if path.parent != plans_dir:
        raise HTTPException(
            status_code=400,
            detail="Plan path is outside the plans directory.",
        )
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Plan '{name}' was not found.")
    return path


def _read_document(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=422, detail=f"Could not read '{path.name}': {exc}"
        ) from exc
    _validate_document(document, path.name)
    return document


def _validate_document(document: dict[str, Any], name: str = "document") -> Any:
    try:
        return plan_from_document(document)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"'{name}' is not a valid plan artifact: {exc}",
        ) from exc


def _record_edit(
    document: dict[str, Any], operation: str, **details: Any
) -> dict[str, Any]:
    """Append an edit record to ``metadata`` without touching the review."""
    updated = dict(document)
    metadata = dict(updated.get("metadata") or {})
    now = datetime.now(timezone.utc).isoformat()
    edits = list(metadata.get("edits") or [])
    edits.append({"at": now, "operation": operation, **details})
    metadata["edits"] = edits
    metadata["edited_at"] = now
    updated["metadata"] = metadata
    return updated


def _touch_saved(document: dict[str, Any]) -> dict[str, Any]:
    updated = dict(document)
    metadata = dict(updated.get("metadata") or {})
    metadata["saved_at"] = datetime.now(timezone.utc).isoformat()
    updated["metadata"] = metadata
    return updated


def _one_line(text: str, limit: int = 80) -> str:
    """Collapse whitespace and shorten ``text`` for a picker label."""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _plan_title(document: dict[str, Any], fallback: str) -> str:
    """Return a human-readable identifier for a plan artifact."""
    for key in ("issue", "summary"):
        value = document.get(key)
        if isinstance(value, str) and value.strip():
            return _one_line(value)
    return Path(fallback).stem


def _plan_metadata(path: Path) -> dict[str, Any]:
    """Read the label fields of one artifact, tolerating broken JSON."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        document = {}
    if not isinstance(document, dict):
        document = {}
    return {
        "title": _plan_title(document, path.name),
        "generated_at": document.get("generated_at"),
        "status": document.get("status"),
        "domain": document.get("domain"),
    }


def create_app(plans_dir: Path | str = DEFAULT_OUTPUT_DIR) -> FastAPI:
    """Create the viewer application for artifacts in ``plans_dir``."""
    root = Path(plans_dir).expanduser().resolve()

    app = FastAPI(
        title="Issue plan viewer",
        version="1.0.0",
        description="Browse, analyze, and edit generated implementation plans.",
    )
    app.state.plans_dir = root
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(DEV_ORIGINS),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    router = APIRouter(prefix="/api")

    @router.get("/plans")
    def list_plans() -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        if root.is_dir():
            for path in root.glob("*.json"):
                stat = path.stat()
                items.append(
                    {
                        "name": path.name,
                        "size": stat.st_size,
                        "modified_at": datetime.fromtimestamp(
                            stat.st_mtime, timezone.utc
                        ).isoformat(),
                        **_plan_metadata(path),
                    }
                )
        items.sort(key=lambda item: item["modified_at"], reverse=True)
        return {"plans": items}

    @router.get("/plans/{name}")
    def get_plan(name: str) -> dict[str, Any]:
        return _read_document(_safe_plan_path(root, name))

    @router.get("/plans/{name}/graph")
    def get_graph(name: str) -> dict[str, Any]:
        document = _read_document(_safe_plan_path(root, name))
        tasks = document["subtasks"]
        known_ids = {task["id"] for task in tasks}
        return {
            "nodes": [
                {
                    "id": task["id"],
                    "title": task["title"],
                    "complexity": task.get("complexity", "medium"),
                }
                for task in tasks
            ],
            "edges": [
                {
                    "id": f"{dependency}->{task['id']}",
                    "source": dependency,
                    "target": task["id"],
                }
                for task in tasks
                for dependency in task.get("depends_on", [])
                if dependency in known_ids
            ],
        }

    @router.post("/plans/{name}/remove", response_model=None)
    def remove(name: str, payload: RemoveRequest) -> Any:
        path = _safe_plan_path(root, name)
        document = (
            payload.document
            if payload.document is not None
            else _read_document(path)
        )
        plan = _validate_document(document, path.name)
        try:
            updated_plan = remove_subtasks(plan, payload.ids, payload.strategy)
        except RemovalError as exc:
            return JSONResponse(
                status_code=409,
                content={
                    "detail": str(exc),
                    "blockers": exc.blockers,
                    "unknown_ids": exc.unknown_ids,
                },
            )

        updated = apply_plan_to_document(document, updated_plan)
        updated = _record_edit(
            updated,
            "remove",
            ids=sorted(set(payload.ids)),
            strategy=payload.strategy,
        )
        return updated

    @router.post("/plans/{name}/save")
    def save(name: str, payload: SaveRequest) -> dict[str, Any]:
        path = _safe_plan_path(root, name)
        _validate_document(payload.document, path.name)
        document = _touch_saved(payload.document)
        written_path, output_format = write_output(document, path, "json")
        return {"path": str(written_path), "format": output_format}

    @router.post("/plans/{name}/export", response_model=None)
    def export(name: str, payload: ExportRequest) -> Response:
        path = _safe_plan_path(root, name)
        _validate_document(payload.document, path.name)
        if payload.format == "markdown":
            content = render_markdown(payload.document)
            if not content.endswith("\n"):
                content += "\n"
            return Response(
                content=content,
                media_type="text/markdown; charset=utf-8",
                headers={
                    "Content-Disposition": (
                        f'attachment; filename="{path.with_suffix(".md").name}"'
                    )
                },
            )
        return Response(
            content=json.dumps(payload.document, indent=2, ensure_ascii=False) + "\n",
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{path.name}"'
            },
        )

    app.include_router(router)

    if (DIST_DIR / "assets").is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=DIST_DIR / "assets"),
            name="assets",
        )

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_frontend(full_path: str) -> Any:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found.")
        if DIST_DIR.is_dir():
            candidate = (DIST_DIR / full_path).resolve()
            if (
                full_path
                and candidate.is_relative_to(DIST_DIR)
                and candidate.is_file()
            ):
                return FileResponse(candidate)
            index = DIST_DIR / "index.html"
            if index.is_file():
                return FileResponse(index)
        raise HTTPException(
            status_code=404,
            detail=(
                "The frontend is not built. Run `npm install && npm run build` "
                "in frontend/ and restart the server."
            ),
        )

    return app

