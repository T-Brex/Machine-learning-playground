/** Numeric CSV import: final column is the target; headers are optional. */
export function parseCsv(text, regression = false) {
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
  const labels = [...new Set(numeric.map((row) => row.at(-1)))].sort(
    (a, b) => a - b,
  );
  return {
    imported: true,
    classes: regression ? 1 : labels.length,
    labels,
    data: numeric.map((row) => ({
      x: row.slice(0, -1).slice(0, 8),
      y: regression ? row.at(-1) : labels.indexOf(row.at(-1)),
    })),
  };
}
