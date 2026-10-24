import { useState } from "react";

import type { SubTask } from "../api/types";
import MaximizeButton from "./MaximizeButton";

interface Props {
  task: SubTask | undefined;
  maximized: boolean;
  onToggleMaximize: () => void;
  onRequestRemove: (ids: string[]) => void;
}

function Bullets({
  items,
  empty = "none",
}: {
  items: string[];
  empty?: string;
}) {
  if (items.length === 0) {
    return <p className="muted">{empty}</p>;
  }
  return (
    <ul className="bullets">
      {items.map((item, index) => (
        <li key={`${index}-${item}`}>{item}</li>
      ))}
    </ul>
  );
}

export default function TaskDetail({
  task,
  maximized,
  onToggleMaximize,
  onRequestRemove,
}: Props) {
  const [copied, setCopied] = useState(false);
  const prompt = task?.implementation_prompt ?? "";
  const panelClass = `panel detail-panel${maximized ? " maximized" : ""}`;
  const maximizeButton = (
    <MaximizeButton
      expanded={maximized}
      label="details panel"
      onToggle={onToggleMaximize}
    />
  );

  async function copyPrompt() {
    if (!prompt) {
      return;
    }
    await navigator.clipboard.writeText(prompt);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  }

  if (!task) {
    return (
      <section className={panelClass} aria-label="Subtask details">
        <header className="panel-header">
          <h2>Details</h2>
          <div className="panel-actions">{maximizeButton}</div>
        </header>
        <p className="empty">Select a subtask to inspect it.</p>
      </section>
    );
  }

  return (
    <section className={panelClass} aria-label="Subtask details">
      <header className="panel-header">
        <h2>
          {task.id} · {task.title}
        </h2>
        <div className="panel-actions">
          <span className={`badge badge-${task.complexity}`}>
            {task.complexity}
          </span>
          {maximizeButton}
          <button
            type="button"
            className="danger"
            onClick={() => onRequestRemove([task.id])}
          >
            Remove
          </button>
        </div>
      </header>

      <div className="detail-scroll">
        <h3>Description</h3>
        <p>{task.description || "—"}</p>

        <h3>Depends on</h3>
        {task.depends_on.length === 0 ? (
          <p className="muted">none</p>
        ) : (
          <ul className="chips">
            {task.depends_on.map((dependency) => (
              <li key={dependency}>{dependency}</li>
            ))}
          </ul>
        )}

        <h3>Files</h3>
        <Bullets items={task.files} empty="not specified" />

        <h3>Acceptance criteria</h3>
        <Bullets items={task.acceptance_criteria} />

        <h3>Verification</h3>
        <pre className="code">{task.verification || "—"}</pre>

        <h3>Risks</h3>
        <Bullets items={task.risks} empty="none recorded" />

        <div className="prompt-header">
          <h3>Implementation prompt</h3>
          <button type="button" onClick={() => void copyPrompt()}>
            {copied ? "Copied" : "Copy"}
          </button>
        </div>
        <pre className="prompt">{task.implementation_prompt || "—"}</pre>
      </div>
    </section>
  );
}
