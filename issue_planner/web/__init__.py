"""Local web viewer for generated implementation plans.

The FastAPI application lives in :mod:`issue_planner.web.server`. The React
frontend is built into ``issue_planner/web/dist`` and served by the same app.
"""

from issue_planner.web.server import create_app

__all__ = ["create_app"]
