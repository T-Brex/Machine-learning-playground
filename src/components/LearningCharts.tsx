import { useEffect, useMemo, useState } from "react";
import { NEURAL_METHODS } from "../data/catalog.ts";
import {
  COLORS,
  decisionSurfaceColor,
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
  const [hiddenSurfaces, setHiddenSurfaces] = useState<string[]>([]);
  const [surfaceOpacity, setSurfaceOpacity] = useState(0.26);
  const [hiddenPointGroups, setHiddenPointGroups] = useState<number[]>([]);
  useEffect(() => {
    setHiddenSurfaces([]);
    setSurfaceOpacity(0.26);
    setHiddenPointGroups([]);
  }, [p.dataset, p.config.method]);
  const boundaries = useMemo(
    () => [
      ...new Map(
        (p.plot?.surfaces || []).map((face) => [
          face.classes.join(":"),
          face.classes,
        ]),
      ).entries(),
    ],
    [p.plot],
  );
  const pointGroups = useMemo(
    () => Array.from({ length: Math.max(0, p.dataset.classes) }, (_, i) => i),
    [p.dataset.classes],
  );
  return (
    <Panel title="Dati e previsione">
      {p.dataset.data[0]?.x.length >= 3 &&
        p.snapshot.fitted &&
        boundaries.length > 0 && (
          <>
            <div
              className="surface-controls"
              aria-label="Visibilità dei confini"
            >
              {boundaries.map(([key, classes]) => (
                <label className="surface-toggle" key={key}>
                  <input
                    type="checkbox"
                    checked={!hiddenSurfaces.includes(key)}
                    onChange={() =>
                      setHiddenSurfaces((values) =>
                        values.includes(key)
                          ? values.filter((value) => value !== key)
                          : [...values, key],
                      )
                    }
                  />
                  <span
                    style={{
                      background: p.snapshot.regression
                        ? "#35d58a"
                        : decisionSurfaceColor(classes, p.dataset.classes),
                    }}
                  />
                  {p.snapshot.regression
                    ? "Superficie di previsione"
                    : `Confine ${classes.map((c) => c + 1).join(" ↔ ")}`}
                </label>
              ))}
            </div>
            <div className="surface-controls" aria-label="Opacità dei confini">
              <label className="surface-toggle surface-opacity-control">
                <span className="surface-opacity-label">Opacità</span>
                <input
                  type="range"
                  min={0.05}
                  max={1}
                  step={0.05}
                  value={surfaceOpacity}
                  onChange={(event) =>
                    setSurfaceOpacity(Number(event.target.value))
                  }
                />
                <strong>{surfaceOpacity.toFixed(2)}</strong>
              </label>
            </div>
          </>
        )}
      {p.dataset.classes > 1 &&
        !p.snapshot.regression &&
        p.snapshot.fitted &&
        (p.plot?.surfaces.length ?? 0) > 0 && (
          <div
            className="surface-controls"
            aria-label="Visibilità dei gruppi di punti"
          >
            {pointGroups.map((group) => (
              <label className="surface-toggle" key={group}>
                <input
                  type="checkbox"
                  checked={!hiddenPointGroups.includes(group)}
                  onChange={() =>
                    setHiddenPointGroups((values) =>
                      values.includes(group)
                        ? values.filter((value) => value !== group)
                        : [...values, group],
                    )
                  }
                />
                <span
                  style={{
                    background: decisionSurfaceColor(
                      [group],
                      p.dataset.classes,
                    ),
                  }}
                />
                {p.snapshot.clustering && p.snapshot.fitted
                  ? `Cluster ${group + 1}`
                  : `Classe ${group + 1}`}
              </label>
            ))}
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
            plot: p.plot,
            predictions: p.predictions,
            centers: p.centers,
            snapshot: p.snapshot,
            view,
            selected,
            hiddenSurfaces,
            hiddenClasses: hiddenPointGroups,
            surfaceOpacity,
          })
        }
      />
      <p className="small muted">
        {p.dataset.data[0]?.x.length >= 3
          ? p.snapshot.regression
            ? "Punti = target osservati · superficie = previsione al target mediano. Assi = prime tre feature; le altre sono fissate alle medie del dataset."
            : p.snapshot.clustering
              ? "Colori = cluster assegnati · croci = centroidi. Assi = prime tre feature."
              : "Superfici = confini di decisione tra le classi, non sezioni orizzontali. Assi = prime tre feature; le altre sono fissate alle medie del dataset."
          : "Punti = esempi · sfondo = regioni previste dal modello."}
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
