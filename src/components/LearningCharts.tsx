import { useTranslation } from "../i18n/Language.tsx";
import { useMemo } from "react";
import { NEURAL_METHODS } from "../data/catalog.ts";
import {
  drawHistogram,
  drawLines,
  drawLossSurface,
  drawNetwork,
} from "../engine/charts.ts";
import type { PlaygroundProps } from "../types.ts";
import CanvasChart from "./CanvasChart.tsx";
import { useThemeColors } from "./ThemeColors.tsx";
import { Panel, Stat, format } from "./Panel.tsx";

export { default as DataChart } from "./DataChart.tsx";

/** Choose the displayed validation metric from the actual task rather than the algorithm name. */
export function LossCharts({ playground: p }: PlaygroundProps) {
  const t = useTranslation();

  const colors = useThemeColors();
  const s = p.snapshot,
    metric = s.regression ? "R²" : s.clustering ? "Purezza" : "Accuracy";
  return (
    <Panel title={t("Loss e prestazione per epoca")}>
      <CanvasChart
        label={t("Loss per epoca")}
        interactive
        draw={(ctx, w, h, view) =>
          drawLines(
            ctx,
            w,
            h,
            [
              {
                values: s.losses,
                color: colors.primary,
                label: "Training loss",
              },
            ],
            { log: true, view },
          )
        }
      />
      <CanvasChart
        label={t(`${metric} train / validazione`)}
        interactive
        draw={(ctx, w, h, view) =>
          drawLines(
            ctx,
            w,
            h,
            [
              {
                values: s.trainHistory,
                color: colors.primary,
                label: "Training",
              },
              {
                values: s.testHistory,
                color: colors.highlight,
                label: "Validazione",
              },
            ],
            { yLabel: `${metric} (%)`, view },
          )
        }
      />
      <p className="small muted">
        {t(
          s.regression
            ? "R² può essere negativo quando il modello predice peggio della media."
            : s.clustering
              ? "Le etichette sono usate solo per valutare la purezza. Con una sola classe etichettata, questa metrica non è informativa."
              : "Percentuale di previsioni corrette sui set separati di training e validazione.",
        )}
      </p>
      <div className="statrow">
        <Stat label={t("Loss finale")} value={format(s.loss)} />
        <Stat label={t("Passi")} value={s.iteration} />
      </div>
    </Panel>
  );
}
/** Display the same loss history used by the epoch chart as a frequency distribution. */
export function Histogram({ playground: p }: PlaygroundProps) {
  const t = useTranslation();

  const colors = useThemeColors();
  return (
    <Panel title={t("Distribuzione della loss")}>
      <CanvasChart
        label={t("Distribuzione della loss")}
        interactive
        draw={(ctx, w, h, view) =>
          drawHistogram(ctx, w, h, p.snapshot.losses, view, colors.primary)
        }
      />
    </Panel>
  );
}
/** Reuse the live engine for activations, gradients, and a loss slice; memoize the expensive slice. */
export function NetworkPanel({ playground: p }: PlaygroundProps) {
  const t = useTranslation();

  const diagnostics = useMemo(
    () => p.engine.neuralDiagnostics(),
    [p.engine, p.snapshot.iteration],
  );
  if (!NEURAL_METHODS.includes(p.config.method)) return null;
  const network = p.engine.neuralVisualization();
  const trace = p.engine.trace();
  return (
    <Panel title={t("Rete neurale · attivazioni e backpropagation")}>
      <CanvasChart
        label={t("Struttura della rete neurale e pesi")}
        height={Math.max(
          560,
          Math.min(650, (network?.hiddenValues.length || 5) * 35),
        )}
        draw={(ctx, w, h) => drawNetwork(ctx, w, h, network)}
      />
      <p className="small muted">
        {t(
          "Blu: pesi positivi · arancione: pesi negativi · spessore proporzionale al valore assoluto. Ogni nodo mostra l’attivazione.",
        )}
      </p>
      {p.running && (
        <p className="small muted">
          {t(
            "La superficie loss e i gradienti dettagliati si aggiornano in pausa o al termine del training",
          )}
          {t(
            p.diagnosticsIteration === null
              ? ""
              : ` ? ultimo passo ${p.diagnosticsIteration}`,
          )}
          {t(".")}
        </p>
      )}
      {diagnostics && (
        <>
          <h3>{t("Superficie loss · due pesi della rete")}</h3>
          <CanvasChart
            label={t("Superficie loss attorno ai due primi pesi")}
            interactive
            draw={(ctx, w, h, view) =>
              drawLossSurface(ctx, w, h, diagnostics, view)
            }
          />
          <p className="small muted">
            {t(
              "Sezione della loss sul training, con gli altri parametri fissi. Bianco = pesi correnti. Blu = loss minore, arancione = loss maggiore.",
            )}
          </p>
          <h3>{t("Pesi e bias · gradienti")}</h3>
          <div className="tablewrap">
            <table className="data">
              <thead>
                <tr>
                  <th>{t("Parametro")}</th>
                  <th>{t("Valore")}</th>
                  <th>{t("∂L / ∂parametro")}</th>
                </tr>
              </thead>
              <tbody>
                {diagnostics.parameters.map((row) => (
                  <tr key={row.name}>
                    <td>{t(row.name)}</td>
                    <td>{t(format(row.value))}</td>
                    <td>{t(format(row.gradient))}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
      {trace.steps
        .filter((step) =>
          /gradient|backprop|ottimizz|aggiorn/i.test(step.title),
        )
        .map((step, i) => (
          <details key={i}>
            <summary>{t(step.title)}</summary>
            <pre>{t(step.detail)}</pre>
            {step.fullDetail && <pre>{t(step.fullDetail)}</pre>}
          </details>
        ))}
    </Panel>
  );
}
