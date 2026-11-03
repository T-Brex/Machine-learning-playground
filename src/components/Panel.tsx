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
  return (
    <section className={`panel ${className}`} {...props}>
      <div className="ph">
        <span>{title}</span>
        {headerAside}
      </div>
      <div className="pb">{children}</div>
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
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      {children}
    </div>
  );
}
export function Stat({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <b>{value}</b>
    </div>
  );
}
export const format = (value: unknown): string =>
  typeof value === "number"
    ? Number.isFinite(value)
      ? Number(value.toFixed(4)).toString()
      : "—"
    : String(value ?? "—");
