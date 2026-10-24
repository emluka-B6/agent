import { useMemo, useState } from "react";

import type { RemoveStrategy } from "../api/client";
import { cascadeIds, findDependents } from "../api/graph";
import type { PlanDocument } from "../api/types";

interface Props {
  plan: PlanDocument;
  ids: string[];
  busy: boolean;
  error: { message: string; blockers?: string[] } | null;
  onCancel: () => void;
  onConfirm: (strategy: RemoveStrategy) => void;
}

export default function RemoveDialog({
  plan,
  ids,
  busy,
  error,
  onCancel,
  onConfirm,
}: Props) {
  const [strategy, setStrategy] = useState<RemoveStrategy>("rewire");

  const dependents = useMemo(
    () => findDependents(plan, ids),
    [plan, ids],
  );
  const cascade = useMemo(() => cascadeIds(plan, ids), [plan, ids]);
  const cascadeExtras = cascade.filter((id) => !ids.includes(id));
  const dependentIds = Object.keys(dependents);
  const removedCount = strategy === "cascade" ? cascade.length : ids.length;
  const remaining = plan.subtasks.length - removedCount;
  const blocked = strategy === "block" && dependentIds.length > 0;
  const removesEverything = remaining < 1;

  return (
    <div className="modal-backdrop" role="presentation">
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="remove-title"
      >
        <h2 id="remove-title">
          {ids.length === 1
            ? `Remove subtask ${ids[0]}?`
            : `Remove ${ids.length} subtasks?`}
        </h2>
        <p className="muted">Selected: {ids.join(", ")}</p>

        <fieldset className="strategy">
          <legend>Dependency strategy</legend>
          <label>
            <input
              type="radio"
              name="strategy"
              checked={strategy === "rewire"}
              onChange={() => setStrategy("rewire")}
            />
            <span>
              <strong>Rewire</strong> — re-point dependents to the removed
              task&apos;s own dependencies.
            </span>
          </label>
          <label>
            <input
              type="radio"
              name="strategy"
              checked={strategy === "cascade"}
              onChange={() => setStrategy("cascade")}
            />
            <span>
              <strong>Cascade</strong> — also remove {cascadeExtras.length}{" "}
              dependent task(s).
            </span>
          </label>
          <label>
            <input
              type="radio"
              name="strategy"
              checked={strategy === "block"}
              onChange={() => setStrategy("block")}
            />
            <span>
              <strong>Block</strong> — refuse while {dependentIds.length} task(s)
              still depend on the selection.
            </span>
          </label>
        </fieldset>

        {dependentIds.length > 0 && (
          <div className="preview">
            <p>Dependents:</p>
            <ul className="bullets">
              {Object.entries(dependents).map(([taskId, dependencies]) => (
                <li key={taskId}>
                  <code>{taskId}</code> depends on {dependencies.join(", ")}
                </li>
              ))}
            </ul>
          </div>
        )}

        {strategy === "cascade" && cascadeExtras.length > 0 && (
          <p className="preview">
            Also removed: <code>{cascadeExtras.join(", ")}</code>
          </p>
        )}

        <p className={removesEverything ? "error-text" : "muted"}>
          {removesEverything
            ? "At least one subtask must remain in the plan."
            : `${remaining} subtask(s) will remain.`}
        </p>

        {error && (
          <div className="banner error">
            {error.message}
            {error.blockers && error.blockers.length > 0
              ? ` (${error.blockers.join(", ")})`
              : ""}
          </div>
        )}

        <div className="modal-actions">
          <button type="button" onClick={onCancel} disabled={busy}>
            Cancel
          </button>
          <button
            type="button"
            className="danger"
            onClick={() => onConfirm(strategy)}
            disabled={busy || blocked || removesEverything}
          >
            {busy ? "Removing…" : "Remove"}
          </button>
        </div>
      </div>
    </div>
  );
}
