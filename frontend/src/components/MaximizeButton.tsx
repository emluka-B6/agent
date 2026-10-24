interface Props {
  expanded: boolean;
  /** Human readable panel name used in the accessible label. */
  label: string;
  onToggle: () => void;
}

/**
 * Expands a panel to the full page and restores it again. The icon reflects
 * the current state; the text stays readable when the glyph does not render.
 */
export default function MaximizeButton({ expanded, label, onToggle }: Props) {
  const action = expanded ? "Restore" : "Maximize";
  return (
    <button
      type="button"
      className="icon-button"
      onClick={onToggle}
      aria-pressed={expanded}
      aria-label={`${action} ${label}`}
      title={expanded ? `${action} ${label} (Esc)` : `${action} ${label}`}
    >
      <svg
        viewBox="0 0 24 24"
        width="14"
        height="14"
        aria-hidden="true"
        focusable="false"
      >
        {expanded ? (
          <path d="M4 9h5V4M20 9h-5V4M4 15h5v5M20 15h-5v5" />
        ) : (
          <path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" />
        )}
      </svg>
      <span>{action}</span>
    </button>
  );
}
