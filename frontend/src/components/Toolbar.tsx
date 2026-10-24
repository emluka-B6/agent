import type { PlanDocument, PlanSummary } from "../api/types";
import type { FontSizes } from "../fontSizes";
import FontSizeMenu from "./FontSizeMenu";
import PlanPicker from "./PlanPicker";

interface Props {
  plans: PlanSummary[];
  planName: string | null;
  plan: PlanDocument | null;
  dirty: boolean;
  busy: boolean;
  fontSizes: FontSizes;
  onSelectPlan: (name: string) => void;
  onReload: () => void;
  onSave: () => void;
  onExport: (format: "json" | "markdown") => void;
  onFontSizeChange: (kind: keyof FontSizes, value: number) => void;
  onFontSizeReset: () => void;
}

export default function Toolbar({
  plans,
  planName,
  plan,
  dirty,
  busy,
  fontSizes,
  onSelectPlan,
  onReload,
  onSave,
  onExport,
  onFontSizeChange,
  onFontSizeReset,
}: Props) {
  const status = plan?.status ?? "unknown";

  return (
    <header className="toolbar">
      <div className="brand">
        <h1>Issue plan viewer</h1>
        <span className="subtitle">
          Browse, analyze, and trim generated plans
        </span>
      </div>

      <div className="toolbar-controls">
        <PlanPicker
          plans={plans}
          value={planName}
          disabled={busy}
          onChange={onSelectPlan}
        />
        <span className={`status status-${status}`}>{status}</span>
        {dirty && <span className="dirty">unsaved edits</span>}
        <button type="button" onClick={onReload} disabled={busy || !planName}>
          Reload
        </button>
        <button
          type="button"
          onClick={onSave}
          disabled={busy || !plan || !dirty}
        >
          Save
        </button>
        <button
          type="button"
          onClick={() => onExport("markdown")}
          disabled={busy || !plan}
        >
          Export .md
        </button>
        <button
          type="button"
          onClick={() => onExport("json")}
          disabled={busy || !plan}
        >
          Export .json
        </button>
        <FontSizeMenu
          sizes={fontSizes}
          onChange={onFontSizeChange}
          onReset={onFontSizeReset}
        />
      </div>
    </header>
  );
}
