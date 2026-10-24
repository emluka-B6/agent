import {
  DEFAULT_FONT_SIZES,
  FONT_KINDS,
  FONT_SIZE_LIMITS,
  type FontSizes,
} from "../fontSizes";

interface Props {
  sizes: FontSizes;
  onChange: (kind: keyof FontSizes, value: number) => void;
  onReset: () => void;
}

function isDefault(sizes: FontSizes): boolean {
  return FONT_KINDS.every((kind) => sizes[kind] === DEFAULT_FONT_SIZES[kind]);
}

/** Toolbar popover with one font-size slider per font family. */
export default function FontSizeMenu({ sizes, onChange, onReset }: Props) {
  return (
    <details className="settings-menu">
      <summary title="Adjust text sizes">Text size</summary>
      <div className="settings-popover">
        <p className="settings-title">Text size</p>
        <p className="settings-note">
          Each font family scales independently and the choice is remembered.
        </p>

        {FONT_KINDS.map((kind) => {
          const limit = FONT_SIZE_LIMITS[kind];
          const inputId = `font-size-${kind}`;
          return (
            <div className="settings-row" key={kind}>
              <label htmlFor={inputId} title={limit.description}>
                {limit.label}
              </label>
              <output htmlFor={inputId}>{sizes[kind]}px</output>
              <input
                id={inputId}
                type="range"
                min={limit.min}
                max={limit.max}
                step={1}
                value={sizes[kind]}
                onChange={(event) => onChange(kind, Number(event.target.value))}
              />
            </div>
          );
        })}

        <button
          type="button"
          className="settings-reset"
          onClick={onReset}
          disabled={isDefault(sizes)}
        >
          Reset to defaults
        </button>
      </div>
    </details>
  );
}
