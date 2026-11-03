/** Render the parameter schema for the selected algorithm, with accessible labels for every input. */
import { METHOD_LABELS, TASKS, NEURAL_METHODS } from "../data/catalog.js";
import { PARAMS } from "../data/methods.js";
import { Field, Panel } from "./Panel.jsx";

export default function ModelPanel({ playground: p }) {
  const { config: c, snapshot: s } = p;
  return (
    <Panel title="Configurazione modello">
      <Field id="algorithm" label="Algoritmo">
        <select
          id="algorithm"
          value={c.method}
          onChange={(e) => p.selectMethod(e.target.value)}
        >
          {TASKS[c.task].methods.map((id) => (
            <option
              key={id}
              value={id}
              disabled={id === "logistic" && p.dataset.classes !== 2}
            >
              {METHOD_LABELS[id]}
            </option>
          ))}
        </select>
      </Field>
      {s.issue && (
        <div className="note" role="status">
          {s.issue}
        </div>
      )}
      <div className="twocol">
        {(PARAMS[c.method] || []).map(([key, label, , type]) => (
          <Field key={key} id={`methodParam_${key}`} label={label}>
            {type.startsWith("select:") ? (
              <select
                id={`methodParam_${key}`}
                value={c.parameters[key]}
                onChange={(e) => p.parameter(key, e.target.value)}
              >
                {type
                  .slice(7)
                  .split("|")
                  .map((value) => (
                    <option key={value}>{value}</option>
                  ))}
              </select>
            ) : (
              <input
                id={`methodParam_${key}`}
                type="number"
                step="any"
                value={c.parameters[key]}
                onChange={(e) => p.parameter(key, e.target.value)}
              />
            )}
          </Field>
        ))}
      </div>
      {NEURAL_METHODS.includes(c.method) && c.task !== "regression" && (
        <Field id="nnOutput" label="Output layer">
          <select
            id="nnOutput"
            value={c.nnOutput}
            onChange={(e) => p.update({ nnOutput: e.target.value })}
          >
            <option value="auto">Automatico dal dataset</option>
            <option value="binary">Sigmoid · binario</option>
            <option value="softmax">Softmax · multiclasse</option>
          </select>
        </Field>
      )}
      <Field id="split" label="Split training / validazione">
        <select
          id="split"
          value={c.split}
          onChange={(e) => p.update({ split: Number(e.target.value) })}
        >
          {[0.8, 0.7, 0.9].map((n) => (
            <option key={n} value={n}>
              {n * 100}% / {Math.round((1 - n) * 100)}%
            </option>
          ))}
        </select>
      </Field>
      <button id="reset" className="secondary full" onClick={p.reset}>
        Reset parametri
      </button>
    </Panel>
  );
}
