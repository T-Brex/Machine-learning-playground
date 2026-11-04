import { LANGUAGES, useLanguage, useTranslation } from "../i18n/Language.tsx";
import type { Language } from "../i18n/Language.tsx";

export default function LanguageSelector() {
  const { language, setLanguage } = useLanguage();
  const t = useTranslation();
  return (
    <label className="language-picker">
      <svg
        width="18"
        height="18"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        aria-hidden="true"
      >
        <circle cx="12" cy="12" r="9" />
        <path d="M3 12h18M12 3c-5 5-5 13 0 18M12 3c5 5 5 13 0 18" />
      </svg>
      <select
        aria-label={t("Lingua")}
        value={language}
        onChange={(event) => setLanguage(event.target.value as Language)}
      >
        {Object.entries(LANGUAGES).map(([code, label]) => (
          <option key={code} value={code} lang={code}>
            {label}
          </option>
        ))}
      </select>
    </label>
  );
}
