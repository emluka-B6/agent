"""Building blocks of the work-planning agent.

The package is split by responsibility so that the CLI entry point
(``agent.py``) stays small and readable:

- :mod:`work_planner.config` - environment loading and shared defaults
- :mod:`work_planner.domains` - domain profiles and their prompt values
- :mod:`work_planner.models` - Pydantic schemas and the workflow state
- :mod:`work_planner.prompts` - shared prompt templates
- :mod:`work_planner.llm` - chat-model factory and structured-output helpers
- :mod:`work_planner.research` - web-source extraction and prompt formatting
- :mod:`work_planner.plan_repair` - subtask ID and dependency normalisation
- :mod:`work_planner.questions` - answering a plan's open questions on resume
- :mod:`work_planner.nodes` - LangGraph node implementations
- :mod:`work_planner.graph` - graph assembly and workflow runner
- :mod:`work_planner.progress` - console progress and partial artifacts
- :mod:`work_planner.output` - JSON/Markdown artifact rendering
- :mod:`work_planner.budget` - cumulative iteration/research budget
- :mod:`work_planner.resume` - continuing an unfinished run from an artifact
"""

from work_planner.budget import RunBudget
from work_planner.domains import (
    DEFAULT_DOMAIN,
    DOMAIN_KEYS,
    DOMAINS,
    DomainProfile,
    get_domain,
)
from work_planner.graph import build_graph, run_agent
from work_planner.llm import build_model
from work_planner.models import (
    Critique,
    ImplementationPlan,
    OpenQuestion,
    PlanRevision,
    ResearchSummary,
    Source,
    SubTask,
    WorkflowState,
)
from work_planner.output import (
    build_document,
    build_progress_document,
    document_status,
    render_markdown,
    unique_output_path,
    write_output,
)
from work_planner.progress import ProgressReporter
from work_planner.resume import (
    ResumeState,
    load_resume_state,
    resolve_resume_path,
)

__all__ = [
    "Critique",
    "DEFAULT_DOMAIN",
    "DOMAIN_KEYS",
    "DOMAINS",
    "DomainProfile",
    "ImplementationPlan",
    "OpenQuestion",
    "PlanRevision",
    "ProgressReporter",
    "ResearchSummary",
    "ResumeState",
    "RunBudget",
    "Source",
    "SubTask",
    "WorkflowState",
    "build_document",
    "build_graph",
    "build_model",
    "build_progress_document",
    "document_status",
    "get_domain",
    "load_resume_state",
    "render_markdown",
    "resolve_resume_path",
    "run_agent",
    "unique_output_path",
    "write_output",
]
