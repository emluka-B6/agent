"""Local web viewer for generated implementation plans.

The FastAPI application lives in :mod:`work_planner.web.server`. The React
frontend is built into ``work_planner/web/dist`` and served by the same app.
"""

from work_planner.web.server import create_app

__all__ = ["create_app"]
