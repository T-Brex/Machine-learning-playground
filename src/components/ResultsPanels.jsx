import { useState } from "react";
import { Panel, Field, format } from "./Panel.jsx";
import { METHOD_LABELS } from "../data/catalog.js";

/** Recursive rows preserve matrices/trees without special-casing every model family. */
/** Flatten inspectable model values into table rows, with limits for large matrices and trees. */
function parameterRows(model) {
  const rows = [];
  function visit(value, path, depth = 0) {
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
export function WeightsPanel({ playground: p }) {
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
export function SamplesPanel({ playground: p, selected, onSelect }) {
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
/** Keep exploratory inputs local so changing a prediction never resets the trained model. */
export function PredictionPanel({ playground: p }) {
  const dimensions = p.dataset.data[0]?.x.length || 2;
  const [values, setValues] = useState({ 0: 0.8, 1: 0.5 });
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
