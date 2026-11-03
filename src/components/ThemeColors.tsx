import { useTranslation } from "../i18n/Language.tsx";
import { createContext, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";

const DEFAULT_COLORS = { primary: "#54cbbb", highlight: "#76ddcd" };
const COOKIE = "ml-playground-colors";
type Colors = typeof DEFAULT_COLORS;

function readColors(): Colors {
  try {
    const cookie = document.cookie
      .split("; ")
      .find((entry) => entry.startsWith(`${COOKIE}=`));
    const value: unknown = cookie
      ? JSON.parse(decodeURIComponent(cookie.slice(COOKIE.length + 1)))
      : null;
    if (
      value &&
      typeof value === "object" &&
      "primary" in value &&
      "highlight" in value &&
      typeof value.primary === "string" &&
      typeof value.highlight === "string" &&
      /^#[\da-f]{6}$/i.test(value.primary) &&
      /^#[\da-f]{6}$/i.test(value.highlight)
    )
      return { primary: value.primary, highlight: value.highlight };
  } catch {
    /* Ignore malformed or old cookie values. */
  }
  return DEFAULT_COLORS;
}

/** Choose readable text for any user-selected button color. */
function ink(color: string): string {
  const rgb = [1, 3, 5].map((start) => {
    const value = parseInt(color.slice(start, start + 2), 16) / 255;
    return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  });
  return rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722 > 0.179
    ? "#101415"
    : "#ffffff";
}

const ThemeContext = createContext({
  colors: DEFAULT_COLORS,
  setColors: (_colors: Colors) => {},
});
export const useThemeColors = () => useContext(ThemeContext).colors;

export function ThemeProvider({ children }: { children: ReactNode }) {
  const t = useTranslation();

  const [colors, setColors] = useState(readColors);
  useEffect(() => {
    const style = document.documentElement.style;
    style.setProperty("--primary", colors.primary);
    style.setProperty("--accent", colors.highlight);
    style.setProperty("--primary-ink", ink(colors.primary));
    style.setProperty("--accent-ink", ink(colors.highlight));
    style.setProperty(
      "--primary-text",
      ink(colors.primary) === "#ffffff"
        ? `color-mix(in srgb, ${colors.primary}, white 60%)`
        : colors.primary,
    );
    document.cookie = `${COOKIE}=${encodeURIComponent(JSON.stringify(colors))}; Max-Age=31536000; Path=/; SameSite=Lax${location.protocol === "https:" ? "; Secure" : ""}`;
  }, [colors]);
  return (
    <ThemeContext.Provider value={{ colors, setColors }}>
      {t(children)}
    </ThemeContext.Provider>
  );
}

export default function ThemeColors() {
  const t = useTranslation();

  const { colors, setColors } = useContext(ThemeContext);
  return (
    <details className="theme-picker">
      <summary aria-label={t("Personalizza colori")}>
        <svg
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          aria-hidden="true"
        >
          <path d="M12 3a9 9 0 1 0 0 18h1a2 2 0 0 0 1.5-3.3 1.5 1.5 0 0 1 1.1-2.5H17a4 4 0 0 0 4-4C21 6.7 17 3 12 3Z" />
          <circle cx="7.5" cy="11" r="1" />
          <circle cx="10" cy="7.5" r="1" />
          <circle cx="14.5" cy="7.5" r="1" />
          <circle cx="17" cy="11" r="1" />
        </svg>
        <span>{t("Colori")}</span>
      </summary>
      <div className="theme-picker-menu">
        <label>
          {t("Primario")}
          <input
            type="color"
            aria-label={t("Colore primario")}
            value={colors.primary}
            onChange={(e) => setColors({ ...colors, primary: e.target.value })}
          />
        </label>
        <label>
          {t("Evidenziazione")}
          <input
            type="color"
            aria-label={t("Colore evidenziazione")}
            value={colors.highlight}
            onChange={(e) =>
              setColors({ ...colors, highlight: e.target.value })
            }
          />
        </label>
        <button
          type="button"
          className="secondary"
          onClick={() => setColors(DEFAULT_COLORS)}
        >
          {t("Ripristina colori")}
        </button>
      </div>
    </details>
  );
}
