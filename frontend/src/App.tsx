import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import {
  ApiError,
  exportPlan,
  listPlans,
  loadPlan,
  removeSubtasks,
  savePlan,
  type RemoveStrategy,
} from "./api/client";
import type { PlanDocument } from "./api/types";
import DependencyGraph from "./components/DependencyGraph";
import RemoveDialog from "./components/RemoveDialog";
import SectionsPanel from "./components/SectionsPanel";
import TaskDetail from "./components/TaskDetail";
import TaskList from "./components/TaskList";
import Toolbar from "./components/Toolbar";
import { useFontSizes } from "./useFontSizes";

interface Banner {
  kind: "error" | "info";
  message: string;
  blockers?: string[];
}

type PanelId = "graph" | "detail";

function toBanner(error: unknown): Banner {
  if (error instanceof ApiError) {
    return { kind: "error", message: error.message, blockers: error.blockers };
  }
  return {
    kind: "error",
    message: error instanceof Error ? error.message : String(error),
  };
}

export default function App() {
  const [planName, setPlanName] = useState<string | null>(null);
  const [plan, setPlan] = useState<PlanDocument | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [dirty, setDirty] = useState(false);
  const [busy, setBusy] = useState(false);
  const [banner, setBanner] = useState<Banner | null>(null);
  const [pendingRemoval, setPendingRemoval] = useState<string[] | null>(null);
  const [expandedPanel, setExpandedPanel] = useState<PanelId | null>(null);
  const { sizes: fontSizes, setSize: setFontSize, reset: resetFontSizes } =
    useFontSizes();

  const plansQuery = useQuery({ queryKey: ["plans"], queryFn: listPlans });
  const planQuery = useQuery({
    queryKey: ["plan", planName],
    queryFn: () => loadPlan(planName as string),
    enabled: planName !== null,
  });

  useEffect(() => {
    if (
      planName === null &&
      plansQuery.data &&
      plansQuery.data.plans.length > 0
    ) {
      setPlanName(plansQuery.data.plans[0].name);
    }
  }, [planName, plansQuery.data]);

  useEffect(() => {
    const loaded = planQuery.data;
    if (!loaded) {
      return;
    }
    setPlan(loaded);
    setSelectedId(loaded.subtasks[0]?.id ?? null);
    setSelectedIds(new Set());
    setDirty(false);
    setBanner(null);
    setExpandedPanel(null);
  }, [planQuery.data]);

  // A maximized panel covers the page, so the page behind it must not scroll.
  useEffect(() => {
    if (!expandedPanel) {
      return;
    }
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previous;
    };
  }, [expandedPanel]);

  // Esc restores a maximized panel, unless the removal dialog is in charge.
  useEffect(() => {
    if (!expandedPanel || pendingRemoval) {
      return;
    }
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setExpandedPanel(null);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [expandedPanel, pendingRemoval]);

  const selectedTask = useMemo(
    () => plan?.subtasks.find((task) => task.id === selectedId),
    [plan, selectedId],
  );

  function selectPlan(name: string) {
    if (name === planName) {
      return;
    }
    if (dirty && !window.confirm("Discard unsaved edits and load another plan?")) {
      return;
    }
    setPlanName(name);
    setPlan(null);
    setSelectedId(null);
    setSelectedIds(new Set());
    setDirty(false);
    setBanner(null);
    setExpandedPanel(null);
  }

  function toggleExpanded(panel: PanelId) {
    setExpandedPanel((current) => (current === panel ? null : panel));
  }

  function toggleSelection(id: string) {
    setSelectedIds((current) => {
      const next = new Set(current);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }

  function selectMany(ids: string[]) {
    setSelectedIds((current) => new Set([...current, ...ids]));
  }

  function clearSelection() {
    setSelectedIds(new Set());
  }

  function requestRemoval(ids: string[]) {
    if (ids.length === 0) {
      return;
    }
    setBanner(null);
    setPendingRemoval(ids);
  }

  async function handleReload() {
    if (!planName) {
      return;
    }
    if (dirty && !window.confirm("Discard unsaved edits and reload the plan?")) {
      return;
    }
    setBanner(null);
    await planQuery.refetch();
  }

  async function handleSave() {
    if (!planName || !plan) {
      return;
    }
    setBusy(true);
    setBanner(null);
    try {
      const result = await savePlan(planName, plan);
      setDirty(false);
      setBanner({ kind: "info", message: `Saved to ${result.path}` });
      await plansQuery.refetch();
    } catch (error) {
      setBanner(toBanner(error));
    } finally {
      setBusy(false);
    }
  }

  async function handleExport(format: "json" | "markdown") {
    if (!planName || !plan) {
      return;
    }
    setBusy(true);
    setBanner(null);
    try {
      const { blob, filename } = await exportPlan(planName, plan, format);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = filename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
      setBanner({ kind: "info", message: `Exported ${filename}` });
    } catch (error) {
      setBanner(toBanner(error));
    } finally {
      setBusy(false);
    }
  }

  async function handleConfirmRemoval(strategy: RemoveStrategy) {
    if (!planName || !plan || !pendingRemoval) {
      return;
    }
    const removed = pendingRemoval;
    setBusy(true);
    setBanner(null);
    try {
      const updated = await removeSubtasks(planName, plan, removed, strategy);
      setPlan(updated);
      setDirty(true);
      setSelectedIds(new Set());
      setSelectedId((current) =>
        current && removed.includes(current)
          ? (updated.subtasks[0]?.id ?? null)
          : current,
      );
      setPendingRemoval(null);
      setBanner({
        kind: "info",
        message: `Removed ${removed.join(", ")} (${strategy}). Save to persist.`,
      });
    } catch (error) {
      setBanner(toBanner(error));
    } finally {
      setBusy(false);
    }
  }

  const queryError = plansQuery.error ?? planQuery.error;
  const plans = plansQuery.data?.plans ?? [];

  return (
    <div className="app">
      <Toolbar
        plans={plans}
        planName={planName}
        plan={plan}
        dirty={dirty}
        busy={busy}
        fontSizes={fontSizes}
        onSelectPlan={selectPlan}
        onReload={() => void handleReload()}
        onSave={() => void handleSave()}
        onExport={(format) => void handleExport(format)}
        onFontSizeChange={setFontSize}
        onFontSizeReset={resetFontSizes}
      />

      {banner && !pendingRemoval && (
        <div className={`banner ${banner.kind}`}>{banner.message}</div>
      )}
      {queryError && (
        <div className="banner error">
          {queryError instanceof Error
            ? queryError.message
            : String(queryError)}
        </div>
      )}

      {plan ? (
        <>
          <main className="workspace">
            <TaskList
              document={plan}
              selectedId={selectedId}
              selectedIds={selectedIds}
              onSelect={setSelectedId}
              onToggle={toggleSelection}
              onSelectMany={selectMany}
              onClearSelection={clearSelection}
              onRequestRemove={requestRemoval}
            />
            <DependencyGraph
              document={plan}
              selectedId={selectedId}
              fontSize={fontSizes.graph}
              maximized={expandedPanel === "graph"}
              onSelect={setSelectedId}
              onRequestRemove={requestRemoval}
              onToggleMaximize={() => toggleExpanded("graph")}
            />
            <TaskDetail
              task={selectedTask}
              maximized={expandedPanel === "detail"}
              onToggleMaximize={() => toggleExpanded("detail")}
              onRequestRemove={requestRemoval}
            />
          </main>
          <SectionsPanel document={plan} />
        </>
      ) : (
        <main className="empty-state">
          <h2>{plansQuery.isLoading ? "Loading plans…" : "No plan loaded"}</h2>
          <p>
            Generate a plan with <code>python agent.py "…"</code>; it is written
            to <code>outputs/</code>, which the viewer scans by default.
          </p>
          {plansQuery.isSuccess && plans.length === 0 && (
            <p className="muted">
              No <code>.json</code> files were found. Start the viewer with{" "}
              <code>python viewer.py</code> or point it at another directory
              with <code>--plans-dir &lt;dir&gt;</code>.
            </p>
          )}
        </main>
      )}

      {pendingRemoval && plan && (
        <RemoveDialog
          plan={plan}
          ids={pendingRemoval}
          busy={busy}
          error={banner?.kind === "error" ? banner : null}
          onCancel={() => {
            setPendingRemoval(null);
            setBanner(null);
          }}
          onConfirm={(strategy) => void handleConfirmRemoval(strategy)}
        />
      )}
    </div>
  );
}
