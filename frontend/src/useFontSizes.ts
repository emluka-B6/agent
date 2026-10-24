import { useCallback, useEffect, useState } from "react";

import {
  applyFontSizes,
  clampFontSize,
  DEFAULT_FONT_SIZES,
  loadFontSizes,
  saveFontSizes,
  type FontSizes,
} from "./fontSizes";

export interface FontSizeControls {
  sizes: FontSizes;
  setSize: (kind: keyof FontSizes, value: number) => void;
  reset: () => void;
}

/**
 * Owns the viewer font sizes: applies them to the document, persists changes,
 * and exposes a setter that already clamps to the supported range.
 */
export function useFontSizes(): FontSizeControls {
  const [sizes, setSizes] = useState<FontSizes>(loadFontSizes);

  useEffect(() => {
    applyFontSizes(sizes);
    saveFontSizes(sizes);
  }, [sizes]);

  const setSize = useCallback((kind: keyof FontSizes, value: number) => {
    if (!Number.isFinite(value)) {
      return;
    }
    setSizes((current) => {
      const next = clampFontSize(kind, value);
      return next === current[kind] ? current : { ...current, [kind]: next };
    });
  }, []);

  const reset = useCallback(() => setSizes(DEFAULT_FONT_SIZES), []);

  return { sizes, setSize, reset };
}
