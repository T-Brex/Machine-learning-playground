import { useMemo, useState } from "react";
import { NEURAL_METHODS } from "../data/catalog.ts";
import {
  COLORS,
  classColor,
  drawHistogram,
  drawLines,
  drawLossSurface,
  drawNetwork,
  drawScatter,
} from "../engine/charts.ts";
import type { PlaygroundProps } from "../types.ts";
import CanvasChart from "./CanvasChart.tsx";
import { Panel, Stat, format } from "./Panel.tsx";

/** Keep per-class surface visibility local to this chart; prediction still uses the full model. */

export function DataChart({
  playground: p,
  selected,
  onSelect,
}: PlaygroundProps & {
  selected?: number;
  onSelect?: (index: number) => void;
}) {
  const [hiddenClasses, setHiddenClasses] = useState<number[]>([]);
  return (
    <Panel title="Dati e previsione">
      {p.dataset.data[0]?.x.length >= 3 &&
        p.snapshot.fitted &&
        !p.snapshot.regression && (
          <div className="surface-controls">
            {Array.from(
              {
                length: p.snapshot.clustering
                  ? Number(p.snapshot.metrics["Numero cluster"]) || 1
                  : p.dataset.classes,
              },
              (_, i) => (
                <label className="surface-toggle" key={i}>
                  <input
                    type="checkbox"
                    checked={!hiddenClasses.includes(i)}
                    onChange={() =>
                      setHiddenClasses((values) =>
                        values.includes(i)
                          ? values.filter((v) => v !== i)
                          : [...values, i],
                      )
                    }
                  />
                  <span style={{ background: classColor(i) }} />
                  Superficie {i + 1}
                </label>
              ),
            )}
          </div>
        )}
      <CanvasChart
        label="Dati e previsione del modello"
        interactive
        rotate={p.dataset.data[0]?.x.length >= 3}
        resetKey={p.dataset}
        height={340}
        onSelect={onSelect}
        draw={(ctx, w, h, view) =>
          drawScatter(ctx, w, h, {
            dataset: p.dataset,
            engine: p.engine,
            snapshot: p.snapshot,
            view,
            selected,
            hiddenClasses,
          })
        }
      />
      <p className="small muted">
        Punti = esempi · sfondo = previsioni. Con almeno 3 feature, proiezione
        3D delle prime tre. Le altre feature sono fissate a zero nelle sezioni
        dello sfondo.
      </p>
    </Panel>
  );
}
/** Choose the displayed validation metric from the actual task rather than the algorithm name. */
export function LossCharts({ playground: p }: PlaygroundProps) {
  const s = p.snapshot,
    metric = s.regression ? "R²" : s.clustering ? "Purezza" : "Accuracy";
  return (
    <Panel title="Loss e prestazione per epoca">
      <CanvasChart
        label="Loss per epoca"
        draw={(ctx, w, h) =>
          drawLines(
            ctx,
            w,
            h,
            [{ values: s.losses, color: COLORS[0], label: "Training loss" }],
            { log: true },
          )
        }
      />
      <CanvasChart
        label={`${metric} train / validazione`}
        draw={(ctx, w, h) =>
          drawLines(
            ctx,
            w,
            h,
            [
              { values: s.trainHistory, color: COLORS[0], label: "Training" },
              { values: s.testHistory, color: COLORS[1], label: "Validazione" },
            ],
            { yLabel: `${metric} (%)` },
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
        draw={(ctx, w, h) => drawHistogram(ctx, w, h, p.snapshot.losses)}
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
          300,
          Math.min(650, (network?.hiddenValues.length || 5) * 35),
        )}
        draw={(ctx, w, h) => drawNetwork(ctx, w, h, network)}
      />
      <p className="small muted">
        Blu: pesi positivi · arancione: pesi negativi · spessore proporzionale
        al valore assoluto. Ogni nodo mostra l’attivazione.
      </p>
      {diagnostics && (
        <>
          <h3>Superficie loss · due pesi della rete</h3>
          <CanvasChart
            label="Superficie loss attorno ai due primi pesi"
            draw={(ctx, w, h) => drawLossSurface(ctx, w, h, diagnostics)}
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
