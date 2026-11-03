import { useTranslation } from "../i18n/Language.tsx";
import type { HTMLAttributes, ReactNode } from "react";

/** Small presentational primitives keep panels, labels, and metric cards consistent. */
export function Panel({
  title,
  children,
  headerAside,
  className = "",
  ...props
}: HTMLAttributes<HTMLElement> & {
  title: string;
  children: ReactNode;
  headerAside?: ReactNode;
}) {
  const t = useTranslation();

  return (
    <section className={`panel ${className}`} {...props}>
      <div className="ph">
        <span>{t(title)}</span>
        {t(headerAside)}
      </div>
      <div className="pb">{t(children)}</div>
    </section>
  );
}
export function Field({
  label,
  id,
  children,
}: {
  label: string;
  id: string;
  children: ReactNode;
}) {
  const t = useTranslation();

  return (
    <div className="field">
      <label htmlFor={id}>{t(label)}</label>
      {t(children)}
    </div>
  );
}
export function Stat({ label, value }: { label: string; value: ReactNode }) {
  const t = useTranslation();

  return (
    <div className="stat">
      <span>{t(label)}</span>
      <b>{t(value)}</b>
    </div>
  );
}
export const format = (value: unknown): string =>
  typeof value === "number"
    ? Number.isFinite(value)
      ? Number(value.toFixed(4)).toString()
      : "—"
    : String(value ?? "—");
