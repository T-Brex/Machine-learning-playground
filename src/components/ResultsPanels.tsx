import { useTranslation } from "../i18n/Language.tsx";
import { useEffect, useMemo, useState } from "react";
import { METHOD_LABELS } from "../data/catalog.ts";
import type { PlaygroundProps } from "../types.ts";
import { Field, format, Panel } from "./Panel.tsx";

/** Show learned numerical state for any model family using a bounded recursive table. */
export function WeightsPanel({ playground: p }: PlaygroundProps) {
  const t = useTranslation();

  const rows = p.parameters;
  return (
    <Panel title={t("Parametri del modello")}>
      <div className="tablewrap">
        <table className="weighttable">
          <thead>
            <tr>
              <th>{t("Parametro")}</th>
              <th>{t("Valore")}</th>
            </tr>
          </thead>
          <tbody>
            {rows.length ? (
              rows.map(([key, value]) => (
                <tr key={key}>
                  <td>{t(key)}</td>
                  <td>{t(format(value))}</td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={2}>
                  {t("Avvia il training per apprendere i parametri.")}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <p className="small muted">
        {t("Valori correnti · fino a 200 parametri mostrati.")}
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
  const t = useTranslation();

  const dimensions = p.dataset.data[0]?.x.length || 2;
  return (
    <Panel title={t("Campioni del dataset")}>
      <div className="tablewrap">
        <table className="data">
          <thead>
            <tr>
              <th>{t("#")}</th>
              {Array.from({ length: dimensions }, (_, i) => (
                <th key={i}>
                  {t("x")}
                  {t(i + 1)}
                </th>
              ))}
              <th>{t("Target y")}</th>
              <th>{t("Predizione")}</th>
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
                      {t(i + 1)}
                    </button>
                  ) : (
                    i + 1
                  )}
                </td>
                {row.x.map((value, j) => (
                  <td key={j}>{t(format(value))}</td>
                ))}
                <td>{t(format(row.y))}</td>
                <td>
                  {t(p.snapshot.fitted ? format(p.engine.predict(row.x)) : "—")}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted">
        {t("Primi 12 campioni. La predizione usa tutte le feature.")}
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
  const t = useTranslation();

  const dimensions = p.dataset.data[0]?.x.length || 2;
  const [values, setValues] = useState<Record<number, string | number>>({
    0: 0.8,
    1: 0.5,
  });
  const x = useMemo(
    () => Array.from({ length: dimensions }, (_, i) => Number(values[i]) || 0),
    [dimensions, values],
  );
  const [prediction, setPrediction] = useState<number | null>(null);
  useEffect(() => {
    let cancelled = false;
    if (!p.snapshot.fitted) {
      setPrediction(null);
      return;
    }
    // Ignore a previous input's response after typing again or replacing a run.
    // React receives a number; inference itself remains inside the worker.
    p.predict(x)
      .then((value) => {
        if (!cancelled) setPrediction(value);
      })
      .catch(() => {
        if (!cancelled) setPrediction(null);
      });
    return () => {
      cancelled = true;
    };
  }, [p.predict, p.snapshot.fitted, p.snapshot.iteration, p.dataset, x]);
  const output =
    prediction === null
      ? "Modello non addestrato"
      : p.snapshot.regression
        ? `ŷ = ${format(prediction)}`
        : p.snapshot.clustering
          ? `Cluster ${prediction < 0 ? "rumore" : prediction + 1}`
          : `Classe ${prediction}`;
  return (
    <Panel title={t("Prova un input")}>
      <div className="twocol">
        {x.map((_, i) => (
          <Field key={i} id={`input-${i}`} label={t(`Input x${i + 1}`)}>
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
        <span className="bigbadge">{t(output)}</span>
      </div>
      <p className="note">
        {t(METHOD_LABELS[p.config.method])}{" "}
        {t("· previsione aggiornata in tempo reale usando tutte le")}{" "}
        {t(dimensions)} {t("feature.")}
      </p>
      {p.dataset.labels &&
        !p.snapshot.regression &&
        prediction !== null &&
        prediction >= 0 && (
          <p>
            {t("Etichetta originale:")} {p.dataset.labels[prediction]}
          </p>
        )}
    </Panel>
  );
}
