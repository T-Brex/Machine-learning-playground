import type { Model } from "../types.ts";

/**
 * Walk a model as unknown data and collect a bounded parameter table.
 *
 * A discriminated model is strongly typed at the engine boundary, but this generic
 * presentation intentionally discovers nested numbers without coupling the table
 * to every family. Narrow unknown values before descending into arrays/objects.
 * Skip training rows, kernel caches, and optimizer bookkeeping because displaying
 * them would overwhelm the learned parameters. Depth and row limits bound work.
 */
export function parameterRows(model: Model | null): [string, number][] {
  const rows: [string, number][] = [];
  function visit(value: unknown, path: string, depth = 0): void {
    if (rows.length >= 200 || depth > 5) return;
    if (typeof value === "number") rows.push([path, value]);
    else if (Array.isArray(value))
      value.forEach((v, i) => visit(v, `${path}[${i}]`, depth + 1));
    else if (value && typeof value === "object")
      Object.entries(value)
        .filter(
          ([key]) =>
            ![
              "rows",
              "K",
              "state",
              "resp",
              "oobVotes",
              "oobCounts",
              "labels",
              "previousModel",
            ].includes(key),
        )
        .forEach(([key, v]) =>
          visit(v, path ? `${path}.${key}` : key, depth + 1),
        );
  }
  visit(model, "");
  return rows;
}
