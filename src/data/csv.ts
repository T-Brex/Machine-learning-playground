import type { Dataset } from "../types.ts";

/**
 * Validate a simple numeric, delimited dataset and normalize its target column.
 *
 * Accept comma, semicolon, or tab separators and an optional nonnumeric header.
 * Every subsequent row must have the same width, at least two numerical features,
 * and a final numerical target. Reject missing/non-finite values rather than letting
 * Number('') silently introduce a zero. Keep at most eight input dimensions.
 *
 * Regression returns target values unchanged. Classification sorts unique numeric
 * labels, maps them to contiguous class indices, and retains the original vocabulary.
 * This parser targets plain numeric files, not quoted fields or general RFC CSV.
 *
 * @throws Error when the dataset is too small or contains malformed numeric rows.
 */
export function parseCsv(text: string, regression = false): Dataset {
  const rows = text
    .replace(/^\uFEFF/, "")
    .trim()
    .split(/\r?\n/)
    .filter((line) => line.trim())
    .map((line) => line.split(/[;,\t]/).map((value) => value.trim()));
  if (rows[0]?.some((value) => value === "" || !Number.isFinite(Number(value))))
    rows.shift();
  if (rows.length < 4 || rows[0].length < 3)
    throw new Error(
      "Servono almeno 4 campioni, due feature numeriche e un target finale.",
    );
  const width = rows[0].length;
  if (
    rows.some(
      (row) =>
        row.length !== width ||
        row.some((value) => value === "" || !Number.isFinite(Number(value))),
    )
  )
    throw new Error(
      "Il CSV contiene valori mancanti, non numerici o righe di lunghezza diversa.",
    );
  const numeric = rows.map((row) => row.map(Number));
  const labels = [...new Set(numeric.map((row) => row[row.length - 1]))].sort(
    (a, b) => a - b,
  );
  return {
    imported: true,
    classes: regression ? 1 : labels.length,
    labels,
    data: numeric.map((row) => ({
      x: row.slice(0, -1).slice(0, 8),
      y: regression ? row[row.length - 1] : labels.indexOf(row[row.length - 1]),
    })),
  };
}
