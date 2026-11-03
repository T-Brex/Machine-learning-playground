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
import { Panel, Stat, format } from "./Panel.tsx";

export { default as DataChart } from "./DataChart.tsx";

/** Choose the displayed validation metric from the actual task rather than the algorithm name. */
export function LossCharts({ playground: p }: PlaygroundProps) {
  const s = p.snapshot,
    metric = s.regression ? "R²" : s.clustering ? "Purezza" : "Accuracy";
  return (
    <Panel title="Loss e prestazione per epoca">
      <CanvasChart
        label="Loss per epoca"
        interactive
        draw={(ctx, w, h, view) =>
          drawLines(
            ctx,
            w,
            h,
            [{ values: s.losses, color: "#54cbbb", label: "Training loss" }],
            { log: true, view },
          )
        }
      />
      <CanvasChart
        label={`${metric} train / validazione`}
        interactive
        draw={(ctx, w, h, view) =>
          drawLines(
            ctx,
            w,
            h,
            [
              { values: s.trainHistory, color: "#54cbbb", label: "Training" },
              { values: s.testHistory, color: "#b6c0c3", label: "Validazione" },
            ],
            { yLabel: `${metric} (%)`, view },
          )
        }
      />
      <p className="small muted">
        {s.regression
          ? "R² può essere negativo quando il modello predice peggio della media."
          : s.clustering
            ? "Le etichette sono usate solo per valutare la purezza. Con una sola classe etichettata, questa metrica non è informativa."
            : "Percentuale di previsioni corrette sui set separati di training e validazione."}
      </p>
      <div className="statrow">
        <Stat label="Loss finale" value={format(s.loss)} />
        <Stat label="Passi" value={s.iteration} />
      </div>
    </Panel>
  );
}
/** Display the same loss history used by the epoch chart as a frequency distribution. */
export function Histogram({ playground: p }: PlaygroundProps) {
  return (
    <Panel title="Distribuzione della loss">
      <CanvasChart
        label="Distribuzione della loss"
        interactive
        draw={(ctx, w, h, view) =>
          drawHistogram(ctx, w, h, p.snapshot.losses, view)
        }
      />
    </Panel>
  );
}
/** Reuse the live engine for activations, gradients, and a loss slice; memoize the expensive slice. */
export function NetworkPanel({ playground: p }: PlaygroundProps) {
  const diagnostics = useMemo(
    () => p.engine.neuralDiagnostics(),
    [p.engine, p.snapshot.iteration],
  );
  if (!NEURAL_METHODS.includes(p.config.method)) return null;
  const network = p.engine.neuralVisualization();
  const trace = p.engine.trace();
  return (
    <Panel title="Rete neurale · attivazioni e backpropagation">
      <CanvasChart
        label="Struttura della rete neurale e pesi"
        height={Math.max(
          560,
          Math.min(650, (network?.hiddenValues.length || 5) * 35),
        )}
        draw={(ctx, w, h) => drawNetwork(ctx, w, h, network)}
      />
      <p className="small muted">
        Blu: pesi positivi · arancione: pesi negativi · spessore proporzionale
        al valore assoluto. Ogni nodo mostra l’attivazione.
      </p>
      {p.running && (
        <p className="small muted">
          La superficie loss e i gradienti dettagliati si aggiornano in pausa o
          al termine del training
          {p.diagnosticsIteration === null
            ? ""
            : ` ? ultimo passo ${p.diagnosticsIteration}`}
          .
        </p>
      )}
      {diagnostics && (
        <>
          <h3>Superficie loss · due pesi della rete</h3>
          <CanvasChart
            label="Superficie loss attorno ai due primi pesi"
            interactive
            draw={(ctx, w, h, view) =>
              drawLossSurface(ctx, w, h, diagnostics, view)
            }
          />
          <p className="small muted">
            Sezione della loss sul training, con gli altri parametri fissi.
            Bianco = pesi correnti. Blu = loss minore, arancione = loss
            maggiore.
          </p>
          <h3>Pesi e bias · gradienti</h3>
          <div className="tablewrap">
            <table className="data">
              <thead>
                <tr>
                  <th>Parametro</th>
                  <th>Valore</th>
                  <th>∂L / ∂parametro</th>
                </tr>
              </thead>
              <tbody>
                {diagnostics.parameters.map((row) => (
                  <tr key={row.name}>
                    <td>{row.name}</td>
                    <td>{format(row.value)}</td>
                    <td>{format(row.gradient)}</td>
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
            <summary>{step.title}</summary>
            <pre>{step.detail}</pre>
            {step.fullDetail && <pre>{step.fullDetail}</pre>}
          </details>
        ))}
    </Panel>
  );
}
