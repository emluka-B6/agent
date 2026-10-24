import { useMemo, useState } from "react";

import type { PlanDocument } from "../api/types";

interface Props {
  document: PlanDocument;
  selectedId: string | null;
  selectedIds: Set<string>;
  onSelect: (id: string) => void;
  onToggle: (id: string) => void;
  onSelectMany: (ids: string[]) => void;
  onClearSelection: () => void;
  onRequestRemove: (ids: string[]) => void;
}

export default function TaskList({
  document,
  selectedId,
  selectedIds,
  onSelect,
  onToggle,
  onSelectMany,
  onClearSelection,
  onRequestRemove,
}: Props) {
  const [query, setQuery] = useState("");
  const [complexity, setComplexity] = useState("all");

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return document.subtasks.filter((task) => {
      if (complexity !== "all" && task.complexity !== complexity) {
        return false;
      }
      if (!needle) {
        return true;
      }
      return [
        task.id,
        task.title,
        task.description,
        task.verification,
        ...task.files,
      ]
        .join(" ")
        .toLowerCase()
        .includes(needle);
    });
  }, [document, query, complexity]);

  const allVisibleSelected =
    visible.length > 0 && visible.every((task) => selectedIds.has(task.id));

  return (
    <section className="panel list-panel" aria-label="Subtasks">
      <header className="panel-header">
        <h2>
          Subtasks <span className="count">{document.subtasks.length}</span>
        </h2>
      </header>

      <div className="filters">
        <input
          type="search"
          placeholder="Filter by id, title, file…"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
        <select
          value={complexity}
          onChange={(event) => setComplexity(event.target.value)}
          aria-label="Filter by complexity"
        >
          <option value="all">All sizes</option>
          <option value="small">Small</option>
          <option value="medium">Medium</option>
          <option value="large">Large</option>
        </select>
      </div>

      <div className="list-actions">
        <label className="checkbox">
          <input
            type="checkbox"
            checked={allVisibleSelected}
            onChange={() =>
              allVisibleSelected
                ? onClearSelection()
                : onSelectMany(visible.map((task) => task.id))
            }
          />
          Select visible
        </label>
        <button
          type="button"
          className="danger"
          disabled={selectedIds.size === 0}
          onClick={() => onRequestRemove([...selectedIds])}
        >
          Remove selected ({selectedIds.size})
        </button>
      </div>

      <ol className="task-list">
        {visible.map((task) => (
          <li
            key={task.id}
            className={selectedId === task.id ? "selected" : undefined}
          >
            <input
              type="checkbox"
              checked={selectedIds.has(task.id)}
              onChange={() => onToggle(task.id)}
              aria-label={`Select ${task.id}`}
            />
            <button
              type="button"
              className="task-row"
              onClick={() => onSelect(task.id)}
            >
              <span className="task-id">{task.id}</span>
              <span className="task-title">{task.title}</span>
              <span className={`badge badge-${task.complexity}`}>
                {task.complexity}
              </span>
            </button>
          </li>
        ))}
        {visible.length === 0 && (
          <li className="empty">No subtasks match the filter.</li>
        )}
      </ol>
    </section>
  );
}
