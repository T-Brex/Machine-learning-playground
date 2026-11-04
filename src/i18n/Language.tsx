import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react";
import type { ReactNode } from "react";
import { translate } from "./translations.ts";

export const LANGUAGES = {
  it: "Italiano",
  en: "English",
  ru: "Русский",
  es: "Español",
  zh: "中文",
};
export type Language = keyof typeof LANGUAGES;
const COOKIE = "ml-playground-language";
const LanguageContext = createContext<{
  language: Language;
  setLanguage: (value: Language) => void;
}>({ language: "it", setLanguage: () => {} });
function savedLanguage(): Language {
  const value = document.cookie
    .split("; ")
    .find((cookie) => cookie.startsWith(`${COOKIE}=`))
    ?.slice(COOKIE.length + 1);
  return value && Object.hasOwn(LANGUAGES, value) ? (value as Language) : "it";
}
export function LanguageProvider({ children }: { children: ReactNode }) {
  const [language, setLanguage] = useState(savedLanguage);
  useEffect(() => {
    document.documentElement.lang = language === "zh" ? "zh-CN" : language;
    document.cookie = `${COOKIE}=${language}; Max-Age=31536000; Path=/; SameSite=Lax${location.protocol === "https:" ? "; Secure" : ""}`;
  }, [language]);
  return (
    <LanguageContext.Provider value={{ language, setLanguage }}>
      {children}
    </LanguageContext.Provider>
  );
}
export const useLanguage = () => useContext(LanguageContext);
/** Translate text at render boundaries; preserve numbers, elements, and model values. */
export function useTranslation() {
  const { language } = useLanguage();
  return useCallback(
    <T,>(value: T): T =>
      typeof value === "string" ? (translate(value, language) as T) : value,
    [language],
  );
}
