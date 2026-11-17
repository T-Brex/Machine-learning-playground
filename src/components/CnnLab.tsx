import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "../i18n/Language.tsx";
import {
  createCnnModel,
  forwardCnn,
  parseMnistImages,
  parseMnistLabels,
  trainCnnBatch,
  traceCnnPixel,
  traceDenseHidden,
  traceDenseOutput,
} from "../engine/cnn.ts";
import type {
  CnnLayer,
  CnnModel,
  CnnPixelStage,
  CnnPooling,
  DenseTrace,
  MnistDigit,
} from "../engine/cnn.ts";
import { Panel } from "./Panel.tsx";

const MNIST_URL = "https://raw.githubusercontent.com/fgnt/mnist/master";
const BATCH_SIZE = 8;

async function downloadIdx(
  url: string,
  signal: AbortSignal,
): Promise<ArrayBuffer> {
  const response = await fetch(url, { signal });
  if (!response.ok)
    throw new Error(`Download MNIST fallito (${response.status}): ${url}`);
  if (!response.body)
    throw new Error("Il browser non può leggere lo stream MNIST.");
  if (!("DecompressionStream" in window))
    throw new Error(
      "La decompressione gzip non è supportata da questo browser.",
    );
  return new Response(
    response.body.pipeThrough(new DecompressionStream("gzip")),
  ).arrayBuffer();
}

function Heatmap({
  values,
  label,
  signed = false,
  selectedCells,
  onSelect,
}: {
  values: number[][];
  label: string;
  signed?: boolean;
  selectedCells?: { row: number; column: number }[] | null;
  onSelect?: (row: number, column: number) => void;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !values.length || !values[0]?.length) return;
    const rows = values.length,
      columns = values[0].length,
      scale = Math.max(1e-8, ...values.flat().map((value) => Math.abs(value)));
    canvas.width = columns * 6;
    canvas.height = rows * 6;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Impossibile disegnare la mappa di attivazione.");
    for (let y = 0; y < rows; y++)
      for (let x = 0; x < columns; x++) {
        const value = values[y][x],
          intensity = Math.round(Math.min(1, Math.abs(value) / scale) * 235);
        ctx.fillStyle = signed
          ? value >= 0
            ? `rgb(40, ${Math.max(20, intensity)}, ${Math.max(20, intensity)})`
            : `rgb(${Math.max(20, intensity)}, 75, 35)`
          : `rgb(${intensity}, ${intensity}, ${intensity})`;
        ctx.fillRect(x * 6, y * 6, 6, 6);
      }
    for (const selectedCell of selectedCells || []) {
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.strokeRect(
        selectedCell.column * 6 + 1,
        selectedCell.row * 6 + 1,
        4,
        4,
      );
    }
  }, [values, signed, selectedCells]);
  return (
    <canvas
      ref={ref}
      className={`cnn-heatmap${onSelect ? " cnn-heatmap-selectable" : ""}`}
      role="img"
      aria-label={label}
      title={label}
      onClick={(event) => {
        if (!onSelect) return;
        const bounds = event.currentTarget.getBoundingClientRect(),
          row = Math.min(
            values.length - 1,
            Math.floor(
              ((event.clientY - bounds.top) / bounds.height) * values.length,
            ),
          ),
          column = Math.min(
            values[0].length - 1,
            Math.floor(
              ((event.clientX - bounds.left) / bounds.width) * values[0].length,
            ),
          );
        onSelect(row, column);
      }}
    />
  );
}

function ValueMatrix({ values, label }: { values: number[][]; label: string }) {
  return (
    <div className="cnn-matrix-wrap">
      <span className="small">{label}</span>
      <table className="cnn-matrix" aria-label={label}>
        <tbody>
          {values.map((row, y) => (
            <tr key={y}>
              {row.map((value, x) => (
                <td key={x}>{value.toFixed(3)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DenseCalculation({
  trace,
  label,
}: {
  trace: DenseTrace;
  label: string;
}) {
  const t = useTranslation();
  return (
    <details className="cnn-dense-details">
      <summary>{label}</summary>
      <div className="tablewrap">
        <table className="data">
          <thead>
            <tr>
              <th>{t("Ingresso")}</th>
              <th>{t("Attivazione precedente")}</th>
              <th>{t("Peso")}</th>
              <th>{t("Prodotto")}</th>
            </tr>
          </thead>
          <tbody>
            {trace.inputs.map((value, index) => (
              <tr key={index}>
                <td>{index}</td>
                <td>{value.toFixed(5)}</td>
                <td>{trace.weights[index].toFixed(5)}</td>
                <td>{trace.products[index].toFixed(5)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted">
        Σ {trace.productSum.toFixed(5)} + {t("bias")} {trace.bias.toFixed(5)} ={" "}
        {trace.value.toFixed(5)}
        {trace.activation !== undefined &&
          ` · ReLU = ${trace.activation.toFixed(5)}`}
      </p>
    </details>
  );
}

export default function CnnLab() {
  const t = useTranslation();
  const [images, setImages] = useState<MnistDigit[]>([]);
  const [selected, setSelected] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [running, setRunning] = useState(false);
  const [learningRate, setLearningRate] = useState(0.05);
  const [targetEpochs, setTargetEpochs] = useState(2);
  const [pooling, setPooling] = useState<CnnPooling>("max");
  const [denseNode, setDenseNode] = useState(0);
  const [outputClass, setOutputClass] = useState(0);
  const [revision, setRevision] = useState(0);
  const [selection, setSelection] = useState<{
    layer: CnnLayer;
    stage: CnnPixelStage;
    filterIndex: number;
    row: number;
    column: number;
  }>({ layer: 1, stage: "convolution", filterIndex: 0, row: 12, column: 12 });
  const [progress, setProgress] = useState({
    steps: 0,
    epoch: 0,
    loss: NaN,
    accuracy: NaN,
  });
  const modelRef = useRef<CnnModel>(createCnnModel());
  const controllerRef = useRef<AbortController | null>(null);
  const cursorRef = useRef(0);
  const stepsRef = useRef(0);
  const totalsRef = useRef({ loss: 0, accuracy: 0, batches: 0 });

  useEffect(
    () => () => {
      controllerRef.current?.abort();
      controllerRef.current = null;
    },
    [],
  );

  async function loadMnist() {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setLoading(true);
    setRunning(false);
    setError("");
    try {
      const [imageBuffer, labelBuffer] = await Promise.all([
        downloadIdx(
          `${MNIST_URL}/train-images-idx3-ubyte.gz`,
          controller.signal,
        ),
        downloadIdx(
          `${MNIST_URL}/train-labels-idx1-ubyte.gz`,
          controller.signal,
        ),
      ]);
      const subset = parseMnistImages(
        imageBuffer,
        parseMnistLabels(labelBuffer),
        80,
      );
      setImages(subset);
      setSelected(0);
      modelRef.current = createCnnModel(pooling);
      cursorRef.current = 0;
      stepsRef.current = 0;
      totalsRef.current = { loss: 0, accuracy: 0, batches: 0 };
      setProgress({ steps: 0, epoch: 0, loss: NaN, accuracy: NaN });
      setRevision((value) => value + 1);
    } catch (cause) {
      if (cause instanceof DOMException && cause.name === "AbortError") return;
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      if (controllerRef.current === controller) {
        controllerRef.current = null;
        setLoading(false);
      }
    }
  }

  const sample = images[selected];
  const forward = useMemo(
    () => (sample ? forwardCnn(modelRef.current, sample.image) : null),
    [sample, revision],
  );
  const pixelTrace = useMemo(
    () =>
      sample && forward
        ? traceCnnPixel(
            modelRef.current,
            sample.image,
            forward,
            selection.layer,
            selection.filterIndex,
            selection.row,
            selection.column,
            selection.stage,
          )
        : null,
    [sample, forward, selection, revision],
  );
  const stepsPerEpoch = Math.ceil(images.length / BATCH_SIZE),
    totalSteps = stepsPerEpoch * targetEpochs;

  const denseHiddenTrace =
      forward && traceDenseHidden(modelRef.current, forward, denseNode),
    denseOutputTrace =
      forward && traceDenseOutput(modelRef.current, forward, outputClass);

  function selectedMapCells(layer: CnnLayer, stage: CnnPixelStage) {
    if (!pixelTrace) return null;
    if (selection.layer !== layer) return null;
    if (stage === "pool")
      return pixelTrace.poolRow === null || pixelTrace.poolColumn === null
        ? null
        : [{ row: pixelTrace.poolRow, column: pixelTrace.poolColumn }];
    if (selection.stage === "pool" && modelRef.current.pooling === "max")
      return pixelTrace.poolWinner ? [pixelTrace.poolWinner] : [];
    if (selection.stage === "pool")
      return pixelTrace.poolCandidates.flat().map(({ row, column }) => ({
        row,
        column,
      }));
    return [
      {
        row: pixelTrace.convolutionRow,
        column: pixelTrace.convolutionColumn,
      },
    ];
  }

  function selectMapPixel(
    layer: CnnLayer,
    stage: CnnPixelStage,
    filterIndex: number,
    row: number,
    column: number,
  ) {
    setSelection({ layer, stage, filterIndex, row, column });
  }

  function changeLayer(layer: CnnLayer) {
    setSelection((current) => ({
      ...current,
      layer,
      stage: "convolution",
      row: layer === 1 ? 12 : 5,
      column: layer === 1 ? 12 : 5,
    }));
  }

  function changeStage(stage: CnnPixelStage) {
    const featureSize = selection.layer === 1 ? 26 : 11,
      poolSize = selection.layer === 1 ? 13 : 5;
    setSelection((current) => ({
      ...current,
      stage,
      row:
        stage === "pool" && current.stage !== "pool"
          ? Math.min(poolSize - 1, Math.floor(current.row / 2))
          : stage !== "pool" && current.stage === "pool"
            ? Math.min(featureSize - 1, current.row * 2)
            : current.row,
      column:
        stage === "pool" && current.stage !== "pool"
          ? Math.min(poolSize - 1, Math.floor(current.column / 2))
          : stage !== "pool" && current.stage === "pool"
            ? Math.min(featureSize - 1, current.column * 2)
            : current.column,
    }));
  }

  function changeCoordinate(axis: "row" | "column", value: string) {
    const max =
        selection.stage === "pool"
          ? selection.layer === 1
            ? 12
            : 4
          : selection.layer === 1
            ? 25
            : 10,
      coordinate = Math.max(0, Math.min(max, Math.floor(Number(value) || 0)));
    setSelection((current) => ({ ...current, [axis]: coordinate }));
  }

  function trainBatch(): boolean {
    if (!images.length || stepsRef.current >= totalSteps) return true;
    const start = cursorRef.current,
      examples = images.slice(
        start,
        Math.min(start + BATCH_SIZE, images.length),
      ),
      result = trainCnnBatch(modelRef.current, examples, learningRate),
      nextCursor = start + examples.length;
    cursorRef.current = nextCursor >= images.length ? 0 : nextCursor;
    stepsRef.current++;
    totalsRef.current.loss += result.loss;
    totalsRef.current.accuracy += result.accuracy;
    totalsRef.current.batches++;
    const steps = stepsRef.current,
      epoch = Math.floor(steps / stepsPerEpoch);
    setProgress({
      steps,
      epoch: Math.min(targetEpochs, epoch),
      loss: totalsRef.current.loss / totalsRef.current.batches,
      accuracy: totalsRef.current.accuracy / totalsRef.current.batches,
    });
    return steps >= totalSteps;
  }

  useEffect(() => {
    if (!running) return;
    let active = true,
      timer = 0;
    const tick = () => {
      if (!active) return;
      let completed = false;
      for (let i = 0; i < 4 && !completed; i++) completed = trainBatch();
      setRevision((value) => value + 1);
      if (completed) setRunning(false);
      else timer = window.setTimeout(tick, 0);
    };
    timer = window.setTimeout(tick, 0);
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [running, images, learningRate, targetEpochs]);

  function resetTraining(poolingMode: CnnPooling = pooling) {
    setRunning(false);
    modelRef.current = createCnnModel(poolingMode);
    cursorRef.current = 0;
    stepsRef.current = 0;
    totalsRef.current = { loss: 0, accuracy: 0, batches: 0 };
    setProgress({ steps: 0, epoch: 0, loss: NaN, accuracy: NaN });
    setRevision((value) => value + 1);
  }

  function updatePooling(poolingMode: CnnPooling) {
    setPooling(poolingMode);
    resetTraining(poolingMode);
  }

  function startTraining() {
    if (progress.steps >= totalSteps) resetTraining();
    setRunning(true);
  }

  function renderFeatureMaps(
    maps: number[][][],
    layer: CnnLayer,
    stage: CnnPixelStage,
  ) {
    return (
      <div className="cnn-map-grid">
        {maps.map((map, index) => {
          const label =
            stage === "convolution"
              ? layer === 1
                ? `${t("Mappa di convoluzione")} ${index + 1}`
                : `${t("Mappa di convoluzione layer 2")} ${index + 1}`
              : stage === "relu"
                ? layer === 1
                  ? `${t("Attivazioni ReLU")} ${index + 1}`
                  : `${t("Attivazioni ReLU layer 2")} ${index + 1}`
                : `${t(pooling === "max" ? "Mappa max-pooling" : "Mappa average pooling")} ${layer} · ${index + 1}`;
          return (
            <div className="cnn-map" key={index}>
              <Heatmap
                values={map}
                label={label}
                signed={stage === "convolution"}
                selectedCells={selectedMapCells(layer, stage)}
                onSelect={(row, column) =>
                  selectMapPixel(layer, stage, index, row, column)
                }
              />
              <span className="small">
                {t("Filtro")} {index + 1} · {map.length}×{map[0].length}
              </span>
            </div>
          );
        })}
      </div>
    );
  }

  function renderFilterKernels(layer: CnnLayer) {
    const kernels =
      layer === 1
        ? modelRef.current.filters1.map((weights, output) => ({
            output,
            input: 0,
            weights,
          }))
        : modelRef.current.filters2.flatMap((banks, output) =>
            banks.map((weights, input) => ({ output, input, weights })),
          );
    return (
      <div className="cnn-map-grid cnn-kernel-grid">
        {kernels.map(({ output, input, weights }) => (
          <div className="cnn-map" key={`${output}-${input}`}>
            <Heatmap
              values={Array.from({ length: 3 }, (_, y) =>
                weights.slice(y * 3, y * 3 + 3),
              )}
              label={`${t("Filtro, kernel 3 per 3")} ${output + 1}`}
              signed
            />
            <span className="small">
              {t("Filtro")} {output + 1}
              {layer === 2 && ` · ${t("Canale ingresso")} ${input + 1}`}
            </span>
            <span className="small">
              [{weights.map((value) => value.toFixed(2)).join(", ")}]
            </span>
          </div>
        ))}
      </div>
    );
  }

  return (
    <Panel title={t("CNN · laboratorio MNIST")} className="cnn-lab">
      <p className="small muted">
        {t(
          "Esplora tutti gli stadi di una CNN su MNIST: due blocchi convoluzionali, ReLU, pooling, flatten, neuroni dense e softmax. I filtri e i pesi si aggiornano con la backpropagation.",
        )}
      </p>
      <div className="cnn-data-controls">
        <button onClick={loadMnist} disabled={loading}>
          {t(
            loading
              ? "Download MNIST in corso…"
              : images.length
                ? "Ricarica MNIST"
                : "Carica MNIST",
          )}
        </button>
        <span className="small muted">
          {images.length
            ? t(`${images.length} immagini bilanciate dal training set MNIST`)
            : t(
                "Scarica il training set MNIST ufficiale (circa 10 MB, solo download).",
              )}
        </span>
      </div>
      {error && (
        <p className="error" role="alert">
          {t(error)}
        </p>
      )}
      {loading && (
        <p className="small muted" role="status">
          {t("Download e decompressione dei file MNIST…")}
        </p>
      )}
      {images.length > 0 && (
        <>
          <div className="cnn-toolbar" aria-label={t("Controlli CNN")}>
            <button onClick={startTraining} disabled={running}>
              {t(
                progress.steps >= totalSteps
                  ? "Ricomincia training"
                  : "Avvia training",
              )}
            </button>
            <button
              className="secondary"
              onClick={() => {
                if (progress.steps < totalSteps) {
                  trainBatch();
                  setRevision((value) => value + 1);
                }
              }}
              disabled={running || progress.steps >= totalSteps}
            >
              {t("Un mini-batch")}
            </button>
            <button
              className="secondary"
              onClick={() => setRunning(false)}
              disabled={!running}
            >
              {t("Pausa")}
            </button>
            <button className="secondary" onClick={() => resetTraining()}>
              {t("Reinizializza filtri")}
            </button>
            <label htmlFor="cnn-pooling">
              {t("Tipo di pooling")}
              <select
                id="cnn-pooling"
                value={pooling}
                onChange={(event) =>
                  updatePooling(event.target.value as CnnPooling)
                }
              >
                <option value="max">{t("Max pooling")}</option>
                <option value="average">{t("Average pooling")}</option>
              </select>
            </label>
            <span className="small muted">
              {t("Il cambio del tipo di pooling reinizializza i pesi.")}
            </span>
            <label>
              {t("Learning rate")}
              <input
                type="number"
                min="0.001"
                max="0.5"
                step="0.01"
                value={learningRate}
                onChange={(event) =>
                  setLearningRate(
                    Math.max(
                      0.001,
                      Math.min(0.5, Number(event.target.value) || 0.05),
                    ),
                  )
                }
              />
            </label>
            <label>
              {t("Epoche")}
              <select
                value={targetEpochs}
                onChange={(event) => {
                  resetTraining();
                  setTargetEpochs(Number(event.target.value));
                }}
              >
                {[1, 2, 3].map((value) => (
                  <option key={value}>{value}</option>
                ))}
              </select>
            </label>
          </div>
          <div className="cnn-progress">
            <div
              className="progress"
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={totalSteps}
              aria-valuenow={progress.steps}
            >
              <div
                style={{
                  width: `${totalSteps ? (progress.steps / totalSteps) * 100 : 0}%`,
                }}
              />
            </div>
            <span className="small muted">
              {t(
                `Epoca ${progress.epoch} / ${targetEpochs} · mini-batch ${progress.steps} / ${totalSteps}`,
              )}
              {Number.isFinite(progress.loss) &&
                ` · ${t("loss")} ${progress.loss.toFixed(3)} · ${t("accuracy")} ${(progress.accuracy * 100).toFixed(1)}%`}
            </span>
          </div>
          <div className="cnn-sample-picker">
            <label htmlFor="cnn-sample">{t("Immagine da ispezionare")}</label>
            <input
              id="cnn-sample"
              type="range"
              min="0"
              max={images.length - 1}
              value={selected}
              onChange={(event) => setSelected(Number(event.target.value))}
            />
            <span className="small muted">
              {t("Campione")} {selected + 1} / {images.length} ·{" "}
              {t("etichetta reale")} {sample?.label}
            </span>
          </div>
          {sample && forward && (
            <>
              <div className="cnn-stages">
                <section className="cnn-stage">
                  <h3>{t("1 · Immagine di input")}</h3>
                  <p className="small muted">
                    {t(
                      "28×28 pixel in scala di grigi · intensità normalizzata tra 0 e 1.",
                    )}
                  </p>
                  <Heatmap
                    values={Array.from({ length: 28 }, (_, y) =>
                      sample.image.slice(y * 28, (y + 1) * 28),
                    )}
                    label={t(`Immagine MNIST, cifra ${sample.label}`)}
                  />
                  <strong>
                    {t("Etichetta MNIST")}: {sample.label}
                  </strong>
                </section>
                <section className="cnn-stage">
                  <h3>{t("2 · Convolution layer 1")}</h3>
                  <p className="small muted">
                    {t(
                      "Quattro kernel 3×3 inizializzati casualmente; i valori si aggiornano con il gradiente.",
                    )}
                  </p>
                  {renderFilterKernels(1)}
                  <p className="small muted">
                    {t(
                      "Ogni pixel in uscita è la somma dei 9 prodotti tra patch 3×3 e kernel, più il bias.",
                    )}
                  </p>
                  {renderFeatureMaps(forward.convolution1, 1, "convolution")}
                </section>
                <section className="cnn-stage">
                  <h3>{t("3 · ReLU layer 1")}</h3>
                  <p className="small muted">
                    {t(
                      "Attivazione elemento per elemento: max(0, x). I valori negativi diventano zero.",
                    )}
                  </p>
                  {renderFeatureMaps(forward.relu1, 1, "relu")}
                </section>
                <section className="cnn-stage">
                  <h3>{t("4 · Pooling layer 1")}</h3>
                  <p className="small muted">
                    {pooling === "max"
                      ? t(
                          "Massimo in ogni finestra 2×2, stride 2: riduce ogni mappa da 26×26 a 13×13.",
                        )
                      : t(
                          "Media dei 4 valori in ogni finestra 2×2, stride 2: riduce ogni mappa da 26×26 a 13×13.",
                        )}
                  </p>
                  {renderFeatureMaps(forward.pooled1, 1, "pool")}
                </section>
                <section className="cnn-stage">
                  <h3>{t("5 · Repetition of feature extraction")}</h3>
                  <p className="small muted">
                    {t(
                      "Il secondo blocco ripete convoluzione, ReLU e pooling sulle feature del primo blocco.",
                    )}
                  </p>
                  <h4>{t("Convolution layer 2")}</h4>
                  <p className="small muted">
                    {t(
                      "Quattro filtri di uscita; ciascuno combina le quattro mappe del primo pooling con 4 kernel 3×3, producendo mappe 11×11.",
                    )}
                  </p>
                  {renderFilterKernels(2)}
                  {renderFeatureMaps(forward.convolution2, 2, "convolution")}
                  <h4>{t("ReLU layer 2")}</h4>
                  {renderFeatureMaps(forward.relu2, 2, "relu")}
                  <h4>{t("Pooling layer 2")}</h4>
                  <p className="small muted">
                    {pooling === "max"
                      ? t(
                          "Finestre 2×2 valide con stride 2: mappe 11×11 → 5×5; l’ultima riga e colonna non entrano in una finestra.",
                        )
                      : t(
                          "Media delle finestre 2×2 valide con stride 2: mappe 11×11 → 5×5; l’ultima riga e colonna non entrano in una finestra.",
                        )}
                  </p>
                  {renderFeatureMaps(forward.pooled2, 2, "pool")}
                </section>
                <section className="cnn-stage">
                  <h3>{t("6 · Flattening layer")}</h3>
                  <p className="small muted">
                    {t(
                      "Le 4 mappe 5×5 sono appiattite in un vettore ordinato di 100 valori, pronto per i livelli densi.",
                    )}
                  </p>
                  <div
                    className="cnn-flat-vector"
                    aria-label={t("Vettore flatten, 100 valori")}
                  >
                    {forward.flattened.map((value, index) => (
                      <span key={index} title={`${index}: ${value.toFixed(6)}`}>
                        {value.toFixed(3)}
                      </span>
                    ))}
                  </div>
                  <p className="small muted">
                    {t("Dimensione vettore")}: {forward.flattened.length}
                  </p>
                </section>
                <section className="cnn-stage">
                  <h3>{t("7 · Fully connected (dense) layer")}</h3>
                  <p className="small muted">
                    {t(
                      "Ogni neurone calcola la somma pesata dei 100 ingressi più il bias, poi applica ReLU.",
                    )}
                  </p>
                  <div className="cnn-pixel-controls">
                    <label htmlFor="cnn-dense-node">
                      {t("Neurone hidden da ispezionare")}
                      <select
                        id="cnn-dense-node"
                        value={denseNode}
                        onChange={(event) =>
                          setDenseNode(Number(event.target.value))
                        }
                      >
                        {forward.denseHidden.map((_, index) => (
                          <option key={index} value={index}>
                            {index}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                  <div className="cnn-dense-activations">
                    {forward.denseHidden.map((value, index) => (
                      <span
                        key={index}
                        className={index === denseNode ? "selected" : ""}
                      >
                        {index}: {value.toFixed(3)}
                      </span>
                    ))}
                  </div>
                  {denseHiddenTrace && (
                    <DenseCalculation
                      trace={denseHiddenTrace}
                      label={t(
                        "Mostra i 100 prodotti, la somma e il bias di questo neurone",
                      )}
                    />
                  )}
                  <h4>{t("8 · Output layer · Softmax")}</h4>
                  <p className="small muted">
                    {t(
                      "Dieci neuroni calcolano i logits dai 16 valori dense; softmax li converte in probabilità che sommano a 1.",
                    )}
                  </p>
                  <div className="cnn-pixel-controls">
                    <label htmlFor="cnn-output-class">
                      {t("Classe di output da ispezionare")}
                      <select
                        id="cnn-output-class"
                        value={outputClass}
                        onChange={(event) =>
                          setOutputClass(Number(event.target.value))
                        }
                      >
                        {forward.probabilities.map((_, index) => (
                          <option key={index} value={index}>
                            {index}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                  {denseOutputTrace && (
                    <DenseCalculation
                      trace={denseOutputTrace}
                      label={t(
                        "Mostra i 16 prodotti, la somma e il bias per il logit",
                      )}
                    />
                  )}
                  <div className="cnn-predictions">
                    {forward.probabilities.map((probability, label) => (
                      <div className="cnn-prediction" key={label}>
                        <span>{label}</span>
                        <div className="progress">
                          <div style={{ width: `${probability * 100}%` }} />
                        </div>
                        <b>{(probability * 100).toFixed(1)}%</b>
                      </div>
                    ))}
                  </div>
                  <p className="small muted">
                    {t("Softmax")}:
                    {" exp(logit − max) / somma exp(logit − max) "}·{" "}
                    {t("Denominatore softmax")}{" "}
                    {forward.logits
                      .reduce(
                        (sum, value) =>
                          sum + Math.exp(value - Math.max(...forward.logits)),
                        0,
                      )
                      .toFixed(6)}
                  </p>
                  <div className="tablewrap">
                    <table className="data">
                      <thead>
                        <tr>
                          <th>{t("Classe")}</th>
                          <th>{t("Logit")}</th>
                          <th>{t("Esponenziale normalizzato")}</th>
                          <th>{t("Probabilità")}</th>
                        </tr>
                      </thead>
                      <tbody>
                        {forward.probabilities.map((probability, index) => {
                          const maxLogit = Math.max(...forward.logits),
                            exponent = Math.exp(
                              forward.logits[index] - maxLogit,
                            );
                          return (
                            <tr
                              key={index}
                              className={
                                index === outputClass ? "cnn-selected-row" : ""
                              }
                            >
                              <td>{index}</td>
                              <td>{forward.logits[index].toFixed(6)}</td>
                              <td>{exponent.toFixed(6)}</td>
                              <td>{probability.toFixed(6)}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  <p className="small muted">
                    {t("Predizione")}:{" "}
                    <strong>
                      {forward.probabilities.indexOf(
                        Math.max(...forward.probabilities),
                      )}
                    </strong>
                    {" · "}
                    {t("Softmax")} {t("somma")}{" "}
                    {forward.probabilities
                      .reduce((sum, value) => sum + value, 0)
                      .toFixed(6)}
                  </p>
                </section>
              </div>
              {pixelTrace && (
                <section className="cnn-pixel-inspector" aria-live="polite">
                  <h3>{t("Calcolo dettagliato del pixel selezionato")}</h3>
                  <p className="small muted">
                    {t(
                      "Clicca un pixel in una mappa, oppure usa i controlli per selezionare filtro e coordinate. La selezione è collegata tra gli stadi.",
                    )}
                  </p>
                  <div className="cnn-pixel-controls">
                    <label htmlFor="cnn-pixel-layer">
                      {t("Layer")}
                      <select
                        id="cnn-pixel-layer"
                        value={selection.layer}
                        onChange={(event) =>
                          changeLayer(Number(event.target.value) as CnnLayer)
                        }
                      >
                        <option value={1}>{t("Convolution layer 1")}</option>
                        <option value={2}>{t("Convolution layer 2")}</option>
                      </select>
                    </label>
                    <label htmlFor="cnn-pixel-stage">
                      {t("Mappa da ispezionare")}
                      <select
                        id="cnn-pixel-stage"
                        value={selection.stage}
                        onChange={(event) =>
                          changeStage(event.target.value as CnnPixelStage)
                        }
                      >
                        <option value="convolution">{t("Convoluzione")}</option>
                        <option value="relu">{t("ReLU")}</option>
                        <option value="pool">{t("Pooling")}</option>
                      </select>
                    </label>
                    <label htmlFor="cnn-pixel-filter">
                      {t("Filtro")}
                      <select
                        id="cnn-pixel-filter"
                        value={selection.filterIndex}
                        onChange={(event) =>
                          setSelection((current) => ({
                            ...current,
                            filterIndex: Number(event.target.value),
                          }))
                        }
                      >
                        {Array.from({ length: 4 }, (_, index) => (
                          <option key={index} value={index}>
                            {index + 1}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label htmlFor="cnn-pixel-row">
                      {t("Riga")}
                      <input
                        id="cnn-pixel-row"
                        type="number"
                        min="0"
                        max={
                          selection.stage === "pool"
                            ? selection.layer === 1
                              ? 12
                              : 4
                            : selection.layer === 1
                              ? 25
                              : 10
                        }
                        value={selection.row}
                        onChange={(event) =>
                          changeCoordinate("row", event.target.value)
                        }
                      />
                    </label>
                    <label htmlFor="cnn-pixel-column">
                      {t("Colonna")}
                      <input
                        id="cnn-pixel-column"
                        type="number"
                        min="0"
                        max={
                          selection.stage === "pool"
                            ? selection.layer === 1
                              ? 12
                              : 4
                            : selection.layer === 1
                              ? 25
                              : 10
                        }
                        value={selection.column}
                        onChange={(event) =>
                          changeCoordinate("column", event.target.value)
                        }
                      />
                    </label>
                  </div>
                  <p className="cnn-pixel-location">
                    {t("Layer")} {selection.layer} · {t("Filtro")}{" "}
                    {selection.filterIndex + 1} · {t("Coordinate selezionate")}{" "}
                    ({selection.row}, {selection.column}) · {t("indici da 0")}
                    {" · "}
                    {t("Pixel convoluzione collegato")} (
                    {pixelTrace.convolutionRow}, {pixelTrace.convolutionColumn})
                  </p>
                  <div className="cnn-pixel-matrices">
                    {pixelTrace.inputPatches.map((patch, channel) => (
                      <div className="cnn-channel-calculation" key={channel}>
                        <span className="cnn-pixel-operator">
                          {channel > 0 ? "+" : ""}
                        </span>
                        <ValueMatrix
                          values={patch}
                          label={
                            selection.layer === 1
                              ? t("Patch immagine normalizzata (3×3)")
                              : `${t("Patch canale")} ${channel + 1} (3×3)`
                          }
                        />
                        <span className="cnn-pixel-operator">×</span>
                        <ValueMatrix
                          values={pixelTrace.filters[channel]}
                          label={`${t("Pesi del filtro (3×3)")} · ${selection.filterIndex + 1} · ${channel + 1}`}
                        />
                        <span className="cnn-pixel-operator">=</span>
                        <ValueMatrix
                          values={pixelTrace.products[channel]}
                          label={`${t("Prodotti elemento per elemento")} · ${channel + 1}`}
                        />
                      </div>
                    ))}
                  </div>
                  <div className="cnn-pixel-equations">
                    <div>
                      <span>
                        {t(
                          selection.layer === 1
                            ? "Somma dei 9 prodotti"
                            : "Somma dei 36 prodotti",
                        )}
                      </span>
                      <b>{pixelTrace.productSum.toFixed(5)}</b>
                    </div>
                    <div>
                      <span>{t("Bias del filtro")}</span>
                      <b>{pixelTrace.bias.toFixed(5)}</b>
                    </div>
                    <div>
                      <span>{t("Convoluzione: somma + bias")}</span>
                      <b>{pixelTrace.convolutionValue.toFixed(5)}</b>
                    </div>
                    <div>
                      <span>{t("ReLU: max(0, convoluzione)")}</span>
                      <b>{pixelTrace.reluValue.toFixed(5)}</b>
                    </div>
                  </div>
                  <div className="cnn-pool-trace">
                    {pixelTrace.poolValue === null ? (
                      <p className="small muted">
                        {t(
                          "Questo pixel sul bordo non rientra in una finestra pooling 2×2 valida.",
                        )}
                      </p>
                    ) : (
                      <>
                        <div>
                          <strong>
                            {t(
                              pooling === "max"
                                ? "Max pooling"
                                : "Average pooling",
                            )}{" "}
                            · {t("finestra")} 2×2 · {t("coordinate")} (
                            {pixelTrace.poolRow}, {pixelTrace.poolColumn})
                          </strong>
                          <span>
                            {pooling === "average" &&
                              `(${pixelTrace.poolCandidates
                                .flat()
                                .map((candidate) => candidate.relu.toFixed(5))
                                .join(" + ")}) / 4 = `}
                            {t("Valore pooling")}:{" "}
                            <b>{pixelTrace.poolValue.toFixed(5)}</b>
                          </span>
                        </div>
                        <table className="cnn-pool-matrix">
                          <tbody>
                            {pixelTrace.poolCandidates.map(
                              (candidateRow, y) => (
                                <tr key={y}>
                                  {candidateRow.map((candidate) => {
                                    const winner =
                                      pixelTrace.poolWinner !== null &&
                                      candidate.row ===
                                        pixelTrace.poolWinner.row &&
                                      candidate.column ===
                                        pixelTrace.poolWinner.column;
                                    return (
                                      <td
                                        key={`${candidate.row}-${candidate.column}`}
                                        className={
                                          winner ? "cnn-pool-winner" : ""
                                        }
                                        title={`${t("Convoluzione")}: ${candidate.convolution.toFixed(5)}`}
                                      >
                                        <span>
                                          ({candidate.row},{candidate.column})
                                        </span>
                                        <small>
                                          {t("Convoluzione")}:{" "}
                                          {candidate.convolution.toFixed(5)}
                                        </small>
                                        <b>ReLU: {candidate.relu.toFixed(5)}</b>
                                        {winner && (
                                          <em>{t("massimo selezionato")}</em>
                                        )}
                                      </td>
                                    );
                                  })}
                                </tr>
                              ),
                            )}
                          </tbody>
                        </table>
                      </>
                    )}
                  </div>
                </section>
              )}
            </>
          )}
        </>
      )}
      <p className="small muted cnn-attribution">
        {t(
          "Fonte dati: MNIST training set, Yann LeCun et al., mirror compresso IDX nel repository GitHub fgnt/mnist. Il browser scarica i dati; nessuna immagine viene caricata o inviata.",
        )}
      </p>
    </Panel>
  );
}
