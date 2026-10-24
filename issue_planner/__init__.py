"""Building blocks of the issue-planning agent.

The package is split by responsibility so that the CLI entry point
(``agent.py``) stays small and readable:

- :mod:`issue_planner.config` - environment loading and shared defaults
- :mod:`issue_planner.domains` - domain profiles and their prompt values
- :mod:`issue_planner.models` - Pydantic schemas and the workflow state
- :mod:`issue_planner.prompts` - shared prompt templates
- :mod:`issue_planner.llm` - chat-model factory and structured-output helpers
- :mod:`issue_planner.research` - web-source extraction and prompt formatting
- :mod:`issue_planner.plan_repair` - subtask ID and dependency normalisation
- :mod:`issue_planner.nodes` - LangGraph node implementations
- :mod:`issue_planner.graph` - graph assembly and workflow runner
- :mod:`issue_planner.progress` - console progress and partial artifacts
- :mod:`issue_planner.output` - JSON/Markdown artifact rendering
"""

from issue_planner.domains import (
    DEFAULT_DOMAIN,
    DOMAIN_KEYS,
    DOMAINS,
    DomainProfile,
    get_domain,
)
from issue_planner.graph import build_graph, run_agent
from issue_planner.llm import build_model
from issue_planner.models import (
    Critique,
    ImplementationPlan,
    ResearchSummary,
    Source,
    SubTask,
    WorkflowState,
)
from issue_planner.output import (
    build_document,
    build_progress_document,
    document_status,
    render_markdown,
    unique_output_path,
    write_output,
)
from issue_planner.progress import ProgressReporter

__all__ = [
    "Critique",
    "DEFAULT_DOMAIN",
    "DOMAIN_KEYS",
    "DOMAINS",
    "DomainProfile",
    "ImplementationPlan",
    "ProgressReporter",
    "ResearchSummary",
    "Source",
    "SubTask",
    "WorkflowState",
    "build_document",
    "build_graph",
    "build_model",
    "build_progress_document",
    "document_status",
    "get_domain",
    "render_markdown",
    "run_agent",
    "unique_output_path",
    "write_output",
]
