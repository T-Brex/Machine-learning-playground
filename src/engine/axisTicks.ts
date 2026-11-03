/** Ticks stay on an origin-anchored 1/2/5 lattice; panning only changes which are visible. */
export function axisTicks(min: number, max: number): number[] {
  const span = max - min;
  if (!Number.isFinite(span) || span <= 0) return [];
  const magnitude = 10 ** Math.floor(Math.log10(span / 6));
  const fraction = span / 6 / magnitude;
  const unit =
    [1, 2, 5, 10].find((value) => fraction <= value * (1 + 1e-9)) ?? 10;
  const step = unit * magnitude;
  const first = Math.ceil(min / step - 1e-9);
  const last = Math.floor(max / step + 1e-9);
  return Array.from(
    { length: Math.max(0, Math.min(64, last - first + 1)) },
    (_, index) => Number(((first + index) * step).toPrecision(12)),
  );
}
