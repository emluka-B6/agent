/**
 * Font sizing for the viewer. The UI uses three font families, so each one can
 * be scaled independently: interface text (sans), ids/code (monospace), and
 * dependency graph node labels.
 */
export interface FontSizes {
  ui: number;
  code: number;
  graph: number;
}

export const FONT_SIZE_LIMITS: Record<
  keyof FontSizes,
  { min: number; max: number; label: string; description: string }
> = {
  ui: {
    min: 11,
    max: 20,
    label: "Interface",
    description: "Toolbars, lists, details and section text.",
  },
  code: {
    min: 9,
    max: 18,
    label: "Code & IDs",
    description: "Monospaced task ids, code and prompt blocks.",
  },
  graph: {
    min: 10,
    max: 20,
    label: "Graph nodes",
    description: "Task labels inside the dependency graph.",
  },
};

export const FONT_KINDS: Array<keyof FontSizes> = ["ui", "code", "graph"];

export const DEFAULT_FONT_SIZES: FontSizes = { ui: 14, code: 12, graph: 12 };

const STORAGE_KEY = "issue-plan-viewer:font-sizes";

export function clampFontSize(kind: keyof FontSizes, value: number): number {
  const { min, max } = FONT_SIZE_LIMITS[kind];
  return Math.min(max, Math.max(min, Math.round(value)));
}

/** Read the persisted sizes, falling back to the defaults for bad data. */
export function loadFontSizes(): FontSizes {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return DEFAULT_FONT_SIZES;
    }
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null) {
      return DEFAULT_FONT_SIZES;
    }
    const record = parsed as Record<string, unknown>;
    const sizes: FontSizes = { ...DEFAULT_FONT_SIZES };
    for (const kind of FONT_KINDS) {
      const value = record[kind];
      if (typeof value === "number" && Number.isFinite(value)) {
        sizes[kind] = clampFontSize(kind, value);
      }
    }
    return sizes;
  } catch {
    return DEFAULT_FONT_SIZES;
  }
}

export function saveFontSizes(sizes: FontSizes): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(sizes));
  } catch {
    // Storage can be unavailable (private mode); sizes still apply per session.
  }
}

/** Publish the sizes as CSS variables consumed by ``styles.css``. */
export function applyFontSizes(sizes: FontSizes): void {
  const root = document.documentElement;
  root.style.setProperty("--font-ui", `${sizes.ui}px`);
  root.style.setProperty("--font-code", `${sizes.code}px`);
  root.style.setProperty("--font-graph", `${sizes.graph}px`);
}
