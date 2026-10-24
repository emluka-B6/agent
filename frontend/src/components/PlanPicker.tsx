import type { PlanSummary } from "../api/types";

interface Props {
  plans: PlanSummary[];
  value: string | null;
  disabled?: boolean;
  onChange: (name: string) => void;
}

/** `2026-10-18T09:31:00+00:00` -> `2026-10-18 09:31:00`. */
function shortTimestamp(value: string | null | undefined): string {
  if (!value) {
    return "";
  }
  const match = /^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}(?::\d{2})?)/.exec(value);
  return match ? `${match[1]} ${match[2]}` : "";
}

/** Prefer the issue text over the hashed file name in the picker. */
function planLabel(plan: PlanSummary): string {
  const title = plan.title?.trim() || plan.name;
  const timestamp = shortTimestamp(plan.generated_at ?? plan.modified_at);
  return timestamp ? `${title} — ${timestamp}` : title;
}

export default function PlanPicker({
  plans,
  value,
  disabled = false,
  onChange,
}: Props) {
  if (plans.length === 0) {
    return (
      <span className="plan-picker empty">
        No .json plans in this directory
      </span>
    );
  }

  return (
    <label className="plan-picker">
      <span>Plan</span>
      <select
        value={value ?? ""}
        onChange={(event) => onChange(event.target.value)}
        disabled={disabled}
      >
        {plans.map((plan) => (
          <option key={plan.name} value={plan.name} title={plan.name}>
            {planLabel(plan)}
          </option>
        ))}
      </select>
    </label>
  );
}
