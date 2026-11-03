/** Small presentational primitives keep panels, labels, and metric cards consistent. */
export function Panel({ title, children, className = "", ...props }) {
  return (
    <section className={`panel ${className}`} {...props}>
      <div className="ph">{title}</div>
      <div className="pb">{children}</div>
    </section>
  );
}
export function Field({ label, id, children }) {
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      {children}
    </div>
  );
}
export function Stat({ label, value }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <b>{value}</b>
    </div>
  );
}
export const format = (value) =>
  typeof value === "number"
    ? Number.isFinite(value)
      ? Number(value.toFixed(4)).toString()
      : "—"
    : (value ?? "—");
