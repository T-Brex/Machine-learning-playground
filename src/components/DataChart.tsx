import { useEffect, useMemo, useState } from "react";
import {
  classColor,
  decisionSurfaceColor,
  drawScatter,
} from "../engine/charts.ts";
import type { PlaygroundProps } from "../types.ts";
import CanvasChart from "./CanvasChart.tsx";
import { Panel } from "./Panel.tsx";

/**
 * Presentation controls are local to this view: they never filter the dataset
 * supplied to training, change class scores, or reset the learned model.
 * Point visibility, boundary opacity, and occupied-space visibility are separate
 * layers so a user can isolate a region without keeping its samples on screen.
 */
export default function DataChart({
  playground: p,
  selected,
  onSelect,
}: PlaygroundProps & {
  selected?: number;
  onSelect?: (index: number) => void;
}) {
  const [hiddenSurfaces, setHiddenSurfaces] = useState<string[]>([]);
  const [surfaceOpacity, setSurfaceOpacity] = useState<Record<string, number>>(
    {},
  );
  const [showPoints, setShowPoints] = useState(true);
  const [hiddenPointSets, setHiddenPointSets] = useState<number[]>([]);
  const [showRegions, setShowRegions] = useState(false);
  const [hiddenRegions, setHiddenRegions] = useState<number[]>([]);
  const [regionOpacity, setRegionOpacity] = useState(0.12);
  const three = (p.dataset.data[0]?.x.length || 2) >= 3;
  const regression = p.config.task === "regression" || p.snapshot.regression;
  const regionsAvailable =
    three && p.snapshot.fitted && !regression && !p.snapshot.clustering;

  useEffect(() => {
    setHiddenSurfaces([]);
    setSurfaceOpacity({});
    setShowPoints(true);
    setHiddenPointSets([]);
    setShowRegions(false);
    setHiddenRegions([]);
    setRegionOpacity(0.12);
  }, [p.dataset, p.config.method]);
  // Ask the worker for volumes only while they are displayed. Sliders and
  // per-class visibility use cached geometry and never send a solver command.
  useEffect(() => {
    if (p.ready) p.requestRegions(showRegions && regionsAvailable);
    return () => p.requestRegions(false);
  }, [
    p.requestRegions,
    p.ready,
    showRegions,
    regionsAvailable,
    p.dataset,
    p.config.method,
  ]);

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
  const pointSets = useMemo(
    () =>
      regression
        ? []
        : p.snapshot.clustering && p.snapshot.fitted
          ? [...new Set(p.predictions)].sort((a, b) => a - b)
          : Array.from({ length: p.dataset.classes }, (_, i) => i),
    [
      regression,
      p.snapshot.clustering,
      p.snapshot.fitted,
      p.predictions,
      p.dataset.classes,
    ],
  );
  const pointName = (label: number) =>
    label < 0
      ? "Rumore"
      : `${p.snapshot.clustering && p.snapshot.fitted ? "Cluster" : "Classe"} ${label + 1}`;
  const togglePointSet = (label: number) =>
    setHiddenPointSets((values) =>
      values.includes(label)
        ? values.filter((value) => value !== label)
        : [...values, label],
    );

  return (
    <Panel title="Dati e previsione" className="data-plot">
      <div className="plot-settings">
        {three && p.snapshot.fitted && boundaries.length > 0 && (
          <fieldset className="graph-control-group">
            <legend>Confini e opacità</legend>
            <div
              className="surface-controls"
              aria-label="Visibilità dei confini"
            >
              {boundaries.map(([key, classes]) => {
                const name = regression
                  ? "Superficie di previsione"
                  : `Confine ${classes.map((c) => c + 1).join(" ↔ ")}`;
                const opacity = surfaceOpacity[key] ?? 0.26;
                return (
                  <div className="boundary-control" key={key}>
                    <label className="surface-toggle">
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
                          background: regression
                            ? "#35d58a"
                            : decisionSurfaceColor(classes, p.dataset.classes),
                        }}
                      />
                      {name}
                    </label>
                    <label className="graph-opacity-control">
                      <span>Opacità {Math.round(opacity * 100)}%</span>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        step="1"
                        value={Math.round(opacity * 100)}
                        aria-label={`Opacità ${name.toLowerCase()}`}
                        disabled={hiddenSurfaces.includes(key)}
                        onChange={(event) =>
                          setSurfaceOpacity((values) => ({
                            ...values,
                            [key]: Number(event.target.value) / 100,
                          }))
                        }
                      />
                    </label>
                  </div>
                );
              })}
            </div>
          </fieldset>
        )}

        <fieldset className="graph-control-group">
          <legend>Punti del dataset</legend>
          <label className="graph-layer-toggle">
            <input
              type="checkbox"
              checked={showPoints}
              onChange={(event) => setShowPoints(event.target.checked)}
              aria-label="Mostra tutti i punti"
            />
            Mostra punti
          </label>
          <div
            className="point-controls"
            aria-label="Visibilità dei gruppi di punti"
          >
            {pointSets.map((label) => (
              <label className="surface-toggle" key={label}>
                <input
                  type="checkbox"
                  checked={!hiddenPointSets.includes(label)}
                  disabled={!showPoints}
                  onChange={() => togglePointSet(label)}
                  aria-label={`Mostra punti ${pointName(label).toLowerCase()}`}
                />
                <span style={{ background: classColor(label) }} />
                {pointName(label)}
              </label>
            ))}
          </div>
        </fieldset>

        {regionsAvailable && (
          <fieldset
            className="graph-control-group region-controls"
            aria-busy={showRegions && !p.plot?.regionsRequested}
          >
            <legend>Spazio classificato</legend>
            <label className="graph-layer-toggle">
              <input
                type="checkbox"
                checked={showRegions}
                onChange={(event) => setShowRegions(event.target.checked)}
                aria-label="Evidenzia spazio classificato"
              />
              Evidenzia lo spazio previsto per classe
            </label>
            {showRegions && (
              <>
                <label className="graph-opacity-control region-opacity">
                  <span>
                    Opacità dello spazio {Math.round(regionOpacity * 100)}%
                  </span>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    step="1"
                    value={Math.round(regionOpacity * 100)}
                    onChange={(event) =>
                      setRegionOpacity(Number(event.target.value) / 100)
                    }
                    aria-label="Opacità spazio classificato"
                  />
                </label>
                <div
                  className="point-controls"
                  aria-label="Visibilità delle regioni classificate"
                >
                  {Array.from({ length: p.dataset.classes }, (_, label) => (
                    <label className="surface-toggle" key={label}>
                      <input
                        type="checkbox"
                        checked={!hiddenRegions.includes(label)}
                        onChange={() =>
                          setHiddenRegions((values) =>
                            values.includes(label)
                              ? values.filter((value) => value !== label)
                              : [...values, label],
                          )
                        }
                        aria-label={`Mostra spazio classe ${label + 1}`}
                      />
                      <span style={{ background: classColor(label) }} />
                      Classe {label + 1}
                    </label>
                  ))}
                </div>
                {!p.plot?.regionsRequested && (
                  <p className="small muted" role="status">
                    Calcolo delle regioni…
                  </p>
                )}
                {p.plot?.regions.some((region) => region.approximate) && (
                  <p className="small muted">
                    Lo spazio dei modelli non lineari è mostrato con
                    un’approssimazione campionata.
                  </p>
                )}
              </>
            )}
          </fieldset>
        )}
      </div>
      <CanvasChart
        label="Dati e previsione del modello"
        interactive
        rotate={three}
        resetKey={p.dataset}
        height={720}
        aspectRatio={1.6}
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
            surfaceOpacity,
            showPoints,
            hiddenPointSets,
            showRegions,
            hiddenRegions,
            regionOpacity,
          })
        }
      />
      <p className="small muted">
        {three
          ? regression
            ? "Punti = target osservati · superficie = previsione al target mediano. Assi = prime tre feature; le altre sono fissate alle medie del dataset."
            : p.snapshot.clustering
              ? "Colori = cluster assegnati · croci = centroidi. Assi = prime tre feature."
              : "Superfici = confini di decisione. Lo spazio evidenziato mostra la classe prevista. Assi = prime tre feature; le altre sono fissate alle medie del dataset."
          : "Punti = esempi · sfondo = regioni previste dal modello."}
      </p>
    </Panel>
  );
}
