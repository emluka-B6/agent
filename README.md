# Issue planning agent

`agent.py` turns an issue from a selected domain (software engineering, AI,
finance, or cybersecurity) into an ordered set of self-contained subtasks. Each
subtask contains an `implementation_prompt` that can be sent directly to an
execution-oriented LLM.

## Project layout

```text
agent.py                  CLI entry point (arguments, main, exit codes)
viewer.py                 Web viewer entry point (uvicorn launcher)
issue_planner/
  __init__.py             Public re-exports
  config.py               .env loading and shared defaults
  domains.py              Domain profiles and their prompt values
  models.py               Pydantic schemas and the workflow state
  prompts.py              Shared prompt templates rendered per domain
  llm.py                  Model factory, structured output, retries
  research.py             Search-result normalisation and prompt formatting
  plan_repair.py          Subtask ID/dependency repair and size warnings
  plan_editor.py          Subtask insert/split/merge/remove and plan revisions
  nodes.py                research / planner / critic LangGraph nodes
  graph.py                Graph assembly and run_agent()
  budget.py               Cumulative iteration/research budget across runs
  progress.py             Console progress and partial-artifact persistence
  output.py               JSON and Markdown artifact rendering
  resume.py               Loading a previous artifact to continue a run
  web/
    server.py             FastAPI app (list/load/remove/save/export)
    dist/                 Built React assets (generated, gitignored)
frontend/                 React + Vite viewer with a React Flow DAG
tests/
  test_plan_editor.py     Unit tests for plan editing and revisions
  test_plan_repair.py     Dependency repair and plan size warnings
  test_prompts.py         Domain profiles and shared prompt templates
  test_progress.py        Console progress and partial artifacts
  test_resume.py          Continuing an unfinished run from its artifact
  test_web.py             API tests for the viewer
  test_workflow.py        Follow-up research routing, budgets, and revisions
```

`agent.py` re-exports the public helpers (`build_model`, `build_graph`,
`run_agent`, `build_document`, `render_markdown`, `write_output`, and the
schemas), so existing code such as `from agent import run_agent` keeps working.

## Workflow

```
research -> planner -> critic
    ^          ^          |
    |          |          |
    |          +----------+   needs improvement
    |
    +----- planner requested more research (bounded)
```

1. **research** optionally calls Tavily and condenses the results into facts,
   constraints, and assumptions. The first call searches the issue. If the
   planner later asks for follow-up research, the research sub-agent runs the
   requested queries, merges the new sources with the earlier ones, and updates
   the summary using both. At most `--max-research-calls` research calls
   (default `2`) run per plan, so the initial pass is followed by at most one
   additional research call.
2. **planner** produces a typed plan with stable task IDs, dependencies,
   artifacts, acceptance criteria, verification steps, and prompts. When a
   critic remark needs external or current facts that are missing from the
   research summary, the planner puts focused search queries in
   `research_requests`. If any are present and the research budget remains, the
   workflow returns to research before the next critic review. Research requests
   are workflow control data and are not written to the output artifact. If web
   research is skipped or unavailable, requests are ignored and the workflow
   continues to the critic. The planner sees the research limitation, how much
   follow-up research budget remains, a numbered title/URL list of the sources,
   and (on revisions) the previous review.

   Plans follow a size policy. A subtask is `small` (one file or artifact, one
   behaviour), `medium` (two to five files or artifacts), or `large` (more than
   five files, a schema or contract change, or several independent behaviours);
   `large` work must be split, a plan holds at most `DEFAULT_MAX_SUBTASKS` (8)
   subtasks, and each `implementation_prompt` targets about 1,500 characters.
   The planner node warns on stderr when a generated plan exceeds those limits,
   and the critic blocks plans that are not sized accordingly.

   The first pass generates the complete plan. Every later pass — after
   follow-up research or a critic review — returns a `PlanRevision` patch
   instead: `upsert_subtasks` replaces subtasks by id (or adds new ones),
   `remove_subtasks` deletes them with dependency rewiring, and the optional
   top-level fields replace the summary, assumptions, open questions, or
   definition of done. Untouched subtasks keep their ids and content, so a
   revision costs a fraction of a full regeneration and ids stay stable in the
   viewer.
3. **critic** reviews the plan. It receives the research summary, the numbered
   source list, the research limitation and budget, its position in the revision
   budget (`review 1 of 2`, `final review`), and its own previous review, so it
   can confirm that earlier blocking issues are resolved and report regressions.
   If it finds blocking issues, the planner revises the plan with a patch. The
   loop is bounded by `--max-iterations`.

A run that ends with `status=max_iterations_reached` is not lost: `--resume`
loads the artifact and continues the loop from the planner (see
[Continue an unfinished run](#continue-an-unfinished-run)).

The final plan is written atomically to a uniquely named output file in
`outputs/` as JSON by default. The name is derived from a hash (for example
`output-4f1c9a7d2b3e.json`), so running the agent again never overwrites an
earlier plan. Pass `--format markdown` to write `output-<hash>.md` instead. The
generated path is printed as the last line of output.

While the workflow runs, progress is printed to stderr (stdout stays reserved
for the final path):

```text
[agent] planning started | domain=software | workflow=research sub-agent -> planner sub-agent -> critic sub-agent | max iterations=2 | max research calls=2
[agent] issue: Add rate limiting to the public API
[agent] partial result file: output-4f1c9a7d2b3e.json (updated after every step)
[1/3] research sub-agent: running...
[1/3] research sub-agent: done in 4.2s | sources=5
[2/3] planner sub-agent: running... | review 1/2
[2/3] planner sub-agent: done in 28.7s | subtasks=6
    - T1: Add a token bucket limiter
    - T2: Wire the limiter into the API middleware
    - T3: Add unit and integration tests
    ...
[3/3] critic sub-agent: running... | reviewing 6 subtasks
[3/3] critic sub-agent: done in 6.1s | approved=no | blocking issues=2
[agent] next sub-agent: planner revision (review 2/2)
[2/3] planner sub-agent: running... | review 2/2 | revising 6 subtasks
[2/3] planner sub-agent: done in 31.2s | subtasks=6
[agent] next sub-agent: research follow-up (research 2/2) | questions=1
[1/3] research sub-agent: running... | follow-up research (2/2) | questions=1
[1/3] research sub-agent: done in 3.8s | sources=9 | research calls=2/2
[2/3] planner sub-agent: running... | review 2/2 | revising 6 subtasks
[2/3] planner sub-agent: done in 19.8s | subtasks=6
[3/3] critic sub-agent: running... | reviewing 6 subtasks
[3/3] critic sub-agent: done in 5.4s | approved=yes | blocking issues=0
[agent] planning finished in 58.7s | status=approved | approved=yes | subtasks=6 | iterations=2/2 | research calls=2/2 | runs=1
```

The final line reports the cumulative `iterations` and `research calls` next to
their caps, plus how many chained runs produced the artifact (`runs`), so a
continuation starts from a known number instead of guessing.

The output file is also updated after every step as a partial artifact, so a
long run never leaves you without a result. Partial artifacts have
`"status": "running"` and record their progress in `metadata.progress` (current
stage, completed stages, subtask count, and subtask titles). When the workflow
finishes, the same path is atomically rewritten with the final plan and status.
If the run fails, the partial artifact is kept and its path is printed.

## Continue an unfinished run

A run can stop without an approval because its revision budget ran out:

```text
[agent] planning finished in 2m 12s | status=max_iterations_reached | approved=no | subtasks=10 | iterations=2/2 | research calls=1/2 | runs=1
```

The plan is complete enough to keep working on, so `--resume` continues from it
instead of regenerating everything:

```bash
# Continue the newest artifact in outputs/ with a fresh budget
python agent.py --resume latest --max-iterations 2

# Continue a specific artifact (or a directory: its newest artifact is used)
python agent.py --resume outputs/output-90a34bf68ba1.json --max-iterations 2
python agent.py --resume outputs --max-iterations 2
```

What a resumed run keeps and what it adds:

- **Kept from the artifact:** the issue, the domain, the plan (same subtask ids),
  the research summary and sources, and the last review. The planner receives
  that review, so the revision targets the blocking issues of the previous run.
- **Skipped:** the initial research pass. The workflow starts at the planner, and
  follow-up research only runs when the planner asks for it and budget remains.
- **Added:** `--max-iterations` and `--max-research-calls` are *extra* budget on
  top of what the artifact already spent. Resuming an artifact that used two
  reviews with `--max-iterations 2` therefore allows two more reviews (a total of
  four), instead of ending immediately. The effective cumulative budget is
  printed before the run starts:

  ```text
  info: continuing output-90a34bf68ba1.json (status=max_iterations_reached, runs=1, iterations=2/2, research calls=1/2); this run adds 2 iteration(s) and 2 research call(s), for a total budget of 4 iteration(s) and 3 research call(s).
  [agent] planning started | domain=software | workflow=research sub-agent -> planner sub-agent -> critic sub-agent | max iterations=4 | max research calls=3 | run 2
  [agent] continuing outputs/output-90a34bf68ba1.json (previous status=max_iterations_reached) | already used: iterations=2, research calls=1 | this run may add 2 iteration(s) and 2 research call(s)
  [2/3] planner sub-agent: running... | review 3/4 | revising 10 subtasks
  ```

- **Recorded in the new artifact:** the cumulative counters
  (`metadata.evaluations`, `metadata.research_calls`), their cumulative caps, the
  per-run values (`metadata.iterations_this_run`, ...), the run count
  (`metadata.runs`), and the artifact it continues (`metadata.resumed_from`,
  `metadata.previous_status`). The new run writes its own file, so the history of
  a long analysis is preserved and every artifact can be resumed again.

Only JSON artifacts can be resumed; a Markdown report is not machine-readable.
An artifact that is still marked `"status": "running"` is rejected, because the
interrupted run may not have written a consistent state.

## Domains

The workflow and output contract are shared by every domain. The selected
domain only changes the values substituted into the shared prompt templates in
`issue_planner/prompts.py`; those values live in `issue_planner/domains.py`.

| `--domain` | Focus | Domain-specific guards |
| --- | --- | --- |
| `software` (default) | software implementation and maintenance | file-level plans and automated verification |
| `ai` | building, evaluating, and using AI responsibly | model/version assumptions, evaluations, safety, cost, and privacy |
| `finance` | investing, portfolio analysis, and financial planning | risk/cost/tax assumptions, no personalized advice or guaranteed returns |
| `cybersecurity` | defensive security engineering | authorized scope only, non-destructive validation, no offensive payloads |

The shared templates keep the same prompt variables and output schema for all
domains. To add another area, register one more `DomainProfile` in
`issue_planner/domains.py`; the research, planner, and critic prompts are then
rendered from the same templates automatically.

The domain can also be set with the `ISSUE_DOMAIN` environment variable. A
programmatic caller can pass `domain="finance"` to `run_agent()`.

## Setup

```bash
cd /home/mlukasik/vscode/ai
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add OPENAI_API_KEY and TAVILY_API_KEY
```

`TAVILY_API_KEY` is optional. Without it, the agent continues without web
research and records that limitation in the output.

### VS Code

The dependencies live in `.venv`. If imports such as `langchain_deepseek` or
`langchain_tavily` are underlined, VS Code is using a different interpreter:

1. Press `Ctrl+Shift+P` and run **Python: Select Interpreter**.
2. Choose `./.venv/bin/python` from the workspace.

The repository ships `.vscode/settings.json` and `pyrightconfig.json`, which
point Pylance at `.venv` and add the project root to the analysis path.

The environment lives in `.venv` on purpose: that is the default name used by
the VS Code Python extension and the one recommended by the Python Packaging
Authority, so editors and tools discover it without extra configuration. Do not
create a second environment in `venv/`; if one appears, remove it and use
`.venv` instead.

## Usage

```bash
# JSON output; the file is named output-<hash>.json (default)
python agent.py "Add rate limiting to the public API"

# Markdown output; the file is named output-<hash>.md
python agent.py "Fix login timeout" --format markdown

# Read a long issue from a file
python agent.py --issue-file issue.md

# Write the artifact into a different directory (the viewer can scan it too)
python agent.py "Add structured logging" --output-dir plans

# Use DeepSeek
python agent.py "Refactor the billing module" --provider deepseek

# List every accepted model ID for both providers
python agent.py --list-models

# Pick a specific model (validated against the provider allow-list)
python agent.py "Refactor the billing module" --provider deepseek --model deepseek-v4-pro
python agent.py "Add rate limiting" --model gpt-5.1

# Plan work in another domain
python agent.py "Audit a RAG assistant for prompt injection" --domain ai
python agent.py "Build a rebalancing checklist for a three-fund portfolio" --domain finance
python agent.py "Harden the public API against the OWASP API Top 10" --domain cybersecurity

# Control planner-triggered follow-up research (2 = initial pass + one follow-up)
python agent.py "Add rate limiting" --max-research-calls 2
python agent.py "Add rate limiting" --max-research-calls 1

# Continue an unfinished run (max_iterations_reached) instead of starting over;
# the budgets are added to what the artifact already spent
python agent.py --resume latest --max-iterations 2
python agent.py --resume outputs/output-4f1c9a7d2b3e.json --max-iterations 3

# Skip web research
python agent.py "Add structured logging" --skip-research

# Read the issue from stdin
cat issue.md | python agent.py
```

Every run creates a new file, so previous results are never overwritten. The
chosen path is printed as the final stdout line (progress messages go to
stderr), which makes it easy to capture:

```bash
PLAN=$(python agent.py "Add structured logging" | tail -n 1)
```

### Providers and models

The provider and model are validated against a strict allow-list before any
network call. An unknown ID stops the agent with exit code `2` and prints the
accepted values.

| Provider | Accepted models | Default |
| --- | --- | --- |
| `openai` | GPT-5 and newer, from `gpt-5` through `gpt-6.1-sol` | `gpt-5-mini` |
| `deepseek` | `deepseek-flash`, `deepseek-v4-pro` | `deepseek-flash` |

`--model` takes precedence over `OPENAI_MODEL` / `DEEPSEEK_MODEL`, which takes
precedence over the provider default. A value from any of those sources must be
in the allow-list. Run `python agent.py --list-models` for the complete list.
Access to a specific model still depends on your provider account entitlements.
DeepSeek models run in non-thinking mode because the workflow relies on
function-calling structured output, which DeepSeek's thinking mode rejects.
They are also given a 32,768-token output budget instead of the API's
8,192-token default: detailed plans would otherwise be truncated mid tool call,
which produces an unparseable response and fails schema validation.

Exit codes:

- `0`: the output file was written (the plan may still be unapproved; check
  `status` in the artifact).
- `1`: the agent or the output writer failed.
- `2`: invalid arguments, an unsupported provider or model, or missing
  configuration.

## Web viewer

The viewer is a small FastAPI server plus a React (Vite) frontend. It lists the
generated JSON artifacts in a directory, renders the subtask dependency graph,
shows every subtask and the planning context, and lets you remove subtasks while
keeping `depends_on` consistent. Plans are identified in the picker by their
issue text and generation time; the hashed file name is shown as a tooltip and
remains the stable API identifier.

### Install

```bash
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

### Run

Production (one process serves the API and the built frontend):

```bash
cd frontend && npm run build && cd ..
python viewer.py --port 8000
# open http://127.0.0.1:8000
```

Development (Vite hot reload on `:5173`, API proxied to `:8000`):

```bash
python viewer.py --port 8000
# in another terminal
cd frontend && npm run dev
```

`--plans-dir` defaults to `$PLANS_DIR` or `outputs`, the same directory
`agent.py` writes to. Use `--plans-dir <dir>` to view plans generated elsewhere.

### Viewer controls

- **Text size** in the toolbar scales each font family separately: interface
  text, monospaced code/ids, and dependency-graph labels. The choice is stored
  in the browser and restored on the next visit.
- **Maximize** on the dependency graph or on the details panel expands it to the
  whole page; **Restore** (or `Esc`) returns to the three-column layout. Only one
  panel can be maximized at a time.
- The dependency graph always shows the **transitive reduction**: if `T9`
  depends on `T8`, and `T8` already depends on `T2` and `T3`, the direct
  `T2 -> T9` and `T3 -> T9` links are hidden. The artifact and every
  `depends_on` list stay untouched.
- Clicking a dependency line **close to a task port** highlights every line
  attached to that port (the click point decides whether the output of the
  source task or the input of the target task is used). That makes a one-to-many
  relation, such as a task with several dependencies or several dependents,
  visible at once. Clicking a line further away from both ports selects just
  that line. Clicking an already highlighted line removes only that line from
  the selection; clicking the empty canvas clears it.

### Editing behavior

A removal can follow three strategies:

- **Rewire** (default): dependents are re-pointed to the removed task's own
  dependencies, so the graph stays valid.
- **Cascade**: every task that transitively depends on the selection is removed
  as well.
- **Block**: the removal is refused while other tasks still depend on the
  selection; the API returns the blocking task ids.

Edits are recorded under `metadata.edits`. `Save` rewrites the original `.json`
artifact atomically, and `Export` downloads the edited plan as JSON or Markdown.
At least one subtask must remain in the plan.

The viewer UI currently exposes removal, but the same UI-agnostic module
(`issue_planner/plan_editor.py`) also provides `insert_subtask`,
`split_subtask`, and `merge_subtasks` for tooling that needs to change the
number of subtasks. They keep ids and `depends_on` consistent: a split chains
the replacement parts and re-points dependents to the last part, and a merge
unions files, acceptance criteria, and risks under the earliest id. The planner
reuses the same module through `apply_revision` to apply `PlanRevision` patches.

### API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/plans` | list `.json` artifacts (with `title`, `generated_at`, `status`) |
| `GET` | `/api/plans/{name}` | load a document |
| `GET` | `/api/plans/{name}/graph` | nodes and dependency edges |
| `POST` | `/api/plans/{name}/remove` | `{ids, strategy, document?}` → updated document |
| `POST` | `/api/plans/{name}/save` | `{document}` → persist to disk |
| `POST` | `/api/plans/{name}/export` | `{document, format}` → file download |

### Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
cd frontend && npm run build   # type-checks and builds the UI
```

## Output

The default JSON artifact contains:

```json
{
  "issue": "...",
  "domain": "software",
  "generated_at": "2026-10-17T00:00:00+00:00",
  "status": "approved",
  "approved": true,
  "summary": "...",
  "subtasks": [
    {
      "id": "T1",
      "title": "...",
      "description": "...",
      "implementation_prompt": "...",
      "depends_on": [],
      "files": ["..."],
      "acceptance_criteria": ["..."],
      "verification": "...",
      "complexity": "small",
      "risks": []
    }
  ],
  "assumptions": [],
  "open_questions": [],
  "definition_of_done": [],
  "review": {
    "approved": true,
    "summary": "...",
    "blocking_issues": [],
    "suggestions": []
  },
  "research": {
    "summary": {},
    "sources": [],
    "error": null
  },
  "metadata": {
    "provider": "openai",
    "model": "gpt-5-mini",
    "evaluations": 1,
    "max_iterations": 2,
    "iterations_this_run": 1,
    "max_iterations_this_run": 2,
    "research_calls": 2,
    "max_research_calls": 2,
    "research_calls_this_run": 2,
    "max_research_calls_this_run": 2,
    "runs": 1,
    "resumed_from": null,
    "previous_status": null
  }
}
```

The plan-level sections have separate jobs, so an item should appear in exactly
one of them:

- `summary` — two to four sentences of strategy; no subtask-by-subtask recap.
- `assumptions` — unverified premises the plan relies on, one bullet each.
- `open_questions` — decisions only a human can make; they never block execution.
- `definition_of_done` — a bullet list of observable, testable end-state
  conditions that prove the whole issue is solved, one condition per item.

`definition_of_done` is the only completion checklist. It does not repeat
assumptions, open questions, or the per-subtask `acceptance_criteria`. If the
planner still emits the same item twice, the artifact builder drops the later
duplicate (ignoring case, punctuation, and spacing) and keeps it in the first
section it belongs to: the definition of done wins over assumptions, and
assumptions win over open questions.

`metadata.evaluations` records how many critic reviews the whole analysis spent,
and `metadata.max_iterations` the cumulative bound. The same applies to
`metadata.research_calls`/`metadata.max_research_calls`. The `*_this_run` values
describe the current invocation, `metadata.runs` counts how many chained runs
produced the artifact, and `metadata.resumed_from`/`metadata.previous_status`
point at the artifact a continued run picked up. Planner `research_requests` are
workflow control data and never appear in the artifact.

`status` is one of `approved`, `needs_improvement`, or
`max_iterations_reached`. While a run is still in progress the same file
contains `status: "running"` and an additional `metadata.progress` object:

```json
{
  "metadata": {
    "progress": {
      "partial": true,
      "stage": "planner_done",
      "running_stage": null,
      "completed_stages": ["research", "planner"],
      "subtask_count": 6,
      "subtask_titles": ["Add a token bucket limiter", "..."]
    }
  }
}
```

`domain` records the profile used to generate the plan. A scheduler can read
`subtasks`, respect `depends_on`, and send each `implementation_prompt` to an
execution LLM.
