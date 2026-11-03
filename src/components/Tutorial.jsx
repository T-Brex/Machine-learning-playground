import { useMemo, useState } from "react";
import { METHOD_INFO } from "../data/methods.js";
import { METHOD_LABELS } from "../data/catalog.js";
import { Panel, Field, format } from "./Panel.jsx";
import { DataChart, NetworkPanel } from "./LearningCharts.jsx";

/** Trace content comes from the same engine that performs the real training steps. */
export default function Tutorial({ playground: p }) {
  const [position, setPosition] = useState(0),
    [query, setQuery] = useState("1"),
    [selected, setSelected] = useState(0);
  const trace = useMemo(
    () => p.engine.trace(),
    [p.engine, p.snapshot.iteration],
  );
  const steps = trace.steps,
    active = Math.min(position, Math.max(0, steps.length - 1)),
    current = steps[active];
  // Tutorial row numbers refer to the fixed training split, matching the engine's trace.
  const rows = p.engine.snapshot().train;
  const sample = rows[selected];
  const visualization = useMemo(
    () => (p.snapshot.fitted ? p.engine.visualization(selected) : null),
    [p.engine, p.snapshot.iteration, selected],
  );
  const entries = visualization?.entries || visualization?.points || [];
  const matches = rows
    .map((row, index) => ({ row, index }))
    .filter(({ index }) => String(index + 1).startsWith(query))
    .slice(0, 12);
  function choose(index) {
    setSelected(index);
    setQuery(String(index + 1));
  }
  return (
    <Panel
      title="Guida didattica · percorso e calcoli passo passo"
      className="algorithm-tutorial"
    >
      <h2>{METHOD_LABELS[p.config.method]}</h2>
      <p>{METHOD_INFO[p.config.method].explain}</p>
      <p className="small muted">{trace.status}</p>
      <div className="tutorial-progress">
        {steps.map((step, i) => (
          <button
            key={i}
            className={i === active ? "active" : "secondary"}
            onClick={() => setPosition(i)}
            aria-label={step.title}
          >
            {i + 1}
          </button>
        ))}
      </div>
      {current && (
        <article className="tutorial-step">
          <div className="tutorial-step-count">
            Passo {active + 1} / {steps.length}
          </div>
          <h3>{current.title}</h3>
          <pre>{current.detail}</pre>
          {current.fullDetail && (
            <details>
              <summary>Calcolo completo</summary>
              <pre>{current.fullDetail}</pre>
            </details>
          )}
        </article>
      )}
      <div className="tutorial-actions">
        <button
          className="secondary"
          disabled={active === 0}
          onClick={() => setPosition(active - 1)}
        >
          ← Indietro
        </button>
        <button onClick={() => setPosition((active + 1) % steps.length)}>
          Avanti →
        </button>
        <button
          className="secondary"
          disabled={p.running || Boolean(p.snapshot.issue)}
          onClick={p.step}
        >
          ▶ Esegui un passo reale nel playground
        </button>
      </div>
      <Field id="tutorialSampleIndex" label="Cerca un campione per numero">
        <input
          id="tutorialSampleIndex"
          type="text"
          inputMode="numeric"
          list="sample-suggestions"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            const n = Number(e.target.value) - 1;
            if (Number.isInteger(n) && n >= 0 && n < rows.length)
              setSelected(n);
          }}
          aria-invalid={
            !/^\d+$/.test(query) ||
            Number(query) < 1 ||
            Number(query) > rows.length
          }
        />
      </Field>
      <datalist id="sample-suggestions">
        {matches.map(({ index }) => (
          <option key={index} value={index + 1} />
        ))}
      </datalist>
      <p className="small muted">
        {sample
          ? `Campione ${selected + 1} · x = [${sample.x.map(format).join(", ")}] · target ${format(sample.y)}${p.snapshot.fitted ? ` · previsione ${format(p.engine.predict(sample.x))}` : ""}`
          : "Scegli un campione valido."}
      </p>
      <DataChart
        playground={p}
        selected={p.dataset.data.indexOf(sample)}
        onSelect={(index) => {
          const trainingIndex = rows.indexOf(p.dataset.data[index]);
          if (trainingIndex >= 0) choose(trainingIndex);
        }}
      />
      {visualization?.message && (
        <p className="note">{visualization.message}</p>
      )}
      {entries.length > 0 && (
        <div className="tablewrap">
          <table className="data">
            <thead>
              <tr>
                <th>Campione</th>
                <th>Target</th>
                <th>Predizione</th>
                <th>Distanza / residuo prima</th>
                <th>Distanza / residuo dopo</th>
                <th>Variazione</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((row, i) => (
                <tr key={i} className={i === selected ? "selected-sample" : ""}>
                  <td>
                    <button className="sample-button" onClick={() => choose(i)}>
                      {i + 1}
                    </button>
                  </td>
                  <td>{format(row.y)}</td>
                  <td>
                    {format(row.prediction ?? row.predicted ?? row.after)}
                  </td>
                  <td>{format(row.beforeDistance)}</td>
                  <td>{format(row.afterDistance)}</td>
                  <td>{format(row.change)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {visualization?.details?.map((detail, i) => (
        <details key={i} open>
          <summary>{detail.title}</summary>
          <table className="weighttable">
            <tbody>
              {detail.rows.map((row, j) => (
                <tr key={j}>
                  <td>{row.label}</td>
                  <td>{format(row.value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      ))}
      <NetworkPanel playground={p} />
      <h3>Calcolo numerico completo</h3>
      {steps.map((step, i) => (
        <details key={i}>
          <summary>{step.title}</summary>
          <pre>{step.detail}</pre>
          {step.fullDetail && <pre>{step.fullDetail}</pre>}
        </details>
      ))}
    </Panel>
  );
}
