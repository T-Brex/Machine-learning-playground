import { useState } from "react";
import { METHOD_LABELS } from "../data/catalog.ts";
import type { Model, PlaygroundProps } from "../types.ts";
import { Field, format, Panel } from "./Panel.tsx";

/**
 * Walk a model as unknown data and collect a bounded parameter table.
 *
 * A discriminated model is strongly typed at the engine boundary, but this generic
 * presentation intentionally discovers nested numbers without coupling the table
 * to every family. Narrow unknown values before descending into arrays/objects.
 * Skip training rows, kernel caches, and optimizer bookkeeping because displaying
 * them would overwhelm the learned parameters. Depth and row limits bound work.
 */
function parameterRows(model: Model | null): [string, number][] {
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
/** Show learned numerical state for any model family using a bounded recursive table. */
export function WeightsPanel({ playground: p }: PlaygroundProps) {
  const rows = parameterRows(p.snapshot.model);
  return (
    <Panel title="Parametri del modello">
      <div className="tablewrap">
        <table className="weighttable">
          <thead>
            <tr>
              <th>Parametro</th>
              <th>Valore</th>
            </tr>
          </thead>
          <tbody>
            {rows.length ? (
              rows.map(([key, value]) => (
                <tr key={key}>
                  <td>{key}</td>
                  <td>{format(value)}</td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={2}>
                  Avvia il training per apprendere i parametri.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <p className="small muted">
        Valori correnti · fino a 200 parametri mostrati.
      </p>
    </Panel>
  );
}
/** Render source rows and live model predictions; optional selection links a table to a chart. */
export function SamplesPanel({
  playground: p,
  selected,
  onSelect,
}: PlaygroundProps & {
  selected?: number;
  onSelect?: (index: number) => void;
}) {
  const dimensions = p.dataset.data[0]?.x.length || 2;
  return (
    <Panel title="Campioni del dataset">
      <div className="tablewrap">
        <table className="data">
          <thead>
            <tr>
              <th>#</th>
              {Array.from({ length: dimensions }, (_, i) => (
                <th key={i}>x{i + 1}</th>
              ))}
              <th>Target y</th>
              <th>Predizione</th>
            </tr>
          </thead>
          <tbody>
            {p.dataset.data.slice(0, 12).map((row, i) => (
              <tr key={i} className={selected === i ? "selected-sample" : ""}>
                <td>
                  {onSelect ? (
                    <button
                      className="sample-button"
                      onClick={() => onSelect(i)}
                    >
                      {i + 1}
                    </button>
                  ) : (
                    i + 1
                  )}
                </td>
                {row.x.map((value, j) => (
                  <td key={j}>{format(value)}</td>
                ))}
                <td>{format(row.y)}</td>
                <td>
                  {p.snapshot.fitted ? format(p.engine.predict(row.x)) : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted">
        Primi 12 campioni. La predizione usa tutte le feature.
      </p>
    </Panel>
  );
}
/**
 * Keep exploratory inputs separate from the trained experiment configuration.
 *
 * Changing an input recomputes a prediction using the existing learned model; it
 * must not reset training, reshuffle validation rows, or alter hyperparameters.
 * Classification displays both its internal class index and, for imported data,
 * the original numeric label retained by the CSV parser.
 */
export function PredictionPanel({ playground: p }: PlaygroundProps) {
  const dimensions = p.dataset.data[0]?.x.length || 2;
  const [values, setValues] = useState<Record<number, string | number>>({
    0: 0.8,
    1: 0.5,
  });
  const x = Array.from(
    { length: dimensions },
    (_, i) => Number(values[i]) || 0,
  );
  const prediction = p.snapshot.fitted ? p.engine.predict(x) : null;
  const output =
    prediction === null
      ? "Modello non addestrato"
      : p.snapshot.regression
        ? `ŷ = ${format(prediction)}`
        : p.snapshot.clustering
          ? `Cluster ${prediction < 0 ? "rumore" : prediction + 1}`
          : `Classe ${prediction}`;
  return (
    <Panel title="Prova un input">
      <div className="twocol">
        {x.map((_, i) => (
          <Field key={i} id={`input-${i}`} label={`Input x${i + 1}`}>
            <input
              id={`input-${i}`}
              type="number"
              step=".1"
              value={values[i] ?? 0}
              onChange={(e) =>
                setValues((v) => ({ ...v, [i]: e.target.value }))
              }
            />
          </Field>
        ))}
      </div>
      <div className="result" aria-live="polite">
        <span className="bigbadge">{output}</span>
      </div>
      <p className="note">
        {METHOD_LABELS[p.config.method]} · previsione aggiornata in tempo reale
        usando tutte le {dimensions} feature.
      </p>
      {p.dataset.labels &&
        !p.snapshot.regression &&
        prediction !== null &&
        prediction >= 0 && (
          <p>Etichetta originale: {p.dataset.labels[prediction]}</p>
        )}
    </Panel>
  );
}
