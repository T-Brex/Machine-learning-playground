import { useTranslation } from "../i18n/Language.tsx";
/** Render the parameter schema for the selected algorithm, with accessible labels for every input. */
import { METHOD_LABELS, NEURAL_METHODS, TASKS } from "../data/catalog.ts";
import { PARAMS } from "../data/methods.ts";
import type { MethodId, PlaygroundProps } from "../types.ts";
import { Field, Panel } from "./Panel.tsx";

export default function ModelPanel({ playground: p }: PlaygroundProps) {
  const t = useTranslation();

  const { config: c, snapshot: s } = p;
  return (
    <Panel title={t("Configurazione modello")} className="model-panel">
      <div className="model-selection">
        <Field id="algorithm" label={t("Algoritmo")}>
          <select
            id="algorithm"
            value={c.method}
            onChange={(e) => p.selectMethod(e.target.value as MethodId)}
          >
            {TASKS[c.task].methods.map((id) => (
              <option
                key={id}
                value={id}
                disabled={id === "logistic" && p.dataset.classes !== 2}
              >
                {t(METHOD_LABELS[id])}
              </option>
            ))}
          </select>
        </Field>
      </div>
      {s.issue && (
        <div className="note" role="status">
          {t(s.issue)}
        </div>
      )}
      <div className="model-parameters">
        {(PARAMS[c.method] || []).map(([key, label, , type]) => (
          <Field key={key} id={`methodParam_${key}`} label={t(label)}>
            {type.startsWith("select:") ? (
              <select
                id={`methodParam_${key}`}
                value={String(c.parameters[key])}
                onChange={(e) => p.parameter(key, e.target.value)}
              >
                {type
                  .slice(7)
                  .split("|")
                  .map((value) => (
                    <option key={value}>{t(value)}</option>
                  ))}
              </select>
            ) : (
              <input
                id={`methodParam_${key}`}
                type="number"
                step="any"
                value={String(c.parameters[key])}
                onChange={(e) => p.parameter(key, e.target.value)}
              />
            )}
          </Field>
        ))}
      </div>
      <div className="model-actions">
        {NEURAL_METHODS.includes(c.method) && c.task !== "regression" && (
          <Field id="nnOutput" label={t("Output layer")}>
            <select
              id="nnOutput"
              value={c.nnOutput}
              onChange={(e) =>
                p.update({
                  nnOutput: e.target.value as "auto" | "binary" | "softmax",
                })
              }
            >
              <option value="auto">{t("Automatico dal dataset")}</option>
              <option value="binary">{t("Sigmoid · binario")}</option>
              <option value="softmax">{t("Softmax · multiclasse")}</option>
            </select>
          </Field>
        )}
        <Field id="split" label={t("Split training / validazione")}>
          <select
            id="split"
            value={c.split}
            onChange={(e) => p.update({ split: Number(e.target.value) })}
          >
            {[0.8, 0.7, 0.9].map((n) => (
              <option key={n} value={n}>
                {t(n * 100)}
                {t("% / ")}
                {t(Math.round((1 - n) * 100))}
                {t("%")}
              </option>
            ))}
          </select>
        </Field>
        <button id="reset" className="secondary full" onClick={p.reset}>
          {t("Reset parametri")}
        </button>
      </div>
    </Panel>
  );
}
