import { NEURAL_METHODS } from "../data/catalog.ts";
import { parameterRows } from "../engine/modelParameters.ts";
import { createEngine } from "../engine/createEngine.ts";
import { buildDecisionSurfaces } from "../engine/decisionSurfaces.ts";
import type { Configuration, Dataset, PageId } from "../types.ts";
import type {
  PredictionGrid,
  SolverCommand,
  SolverEvent,
  SolverFrame,
} from "./protocol.ts";

const FRAME_INTERVAL_MS = 125;
const SLICE_BUDGET_MS = 8;
const MAX_STEPS_PER_SLICE = 8;

/** Injected timers make cancellation, throttling, and backpressure deterministic in tests. */
export interface SolverScheduler {
  now(): number;
  schedule(callback: () => void, delay: number): unknown;
  cancel(handle: unknown): void;
}
const browserScheduler: SolverScheduler = {
  now: () => performance.now(),
  schedule: (callback, delay) => setTimeout(callback, delay),
  cancel: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
};

/**
 * Run numerical work on the worker's event loop, not the window event loop.
 *
 * Each slice finishes at an epoch boundary before yielding to worker commands.
 * A particularly expensive epoch can still occupy the worker for a while, but
 * it cannot block typing, pointer gestures, or animation on the main thread.
 * Reset/settings changes terminate the worker, cancelling even such an epoch.
 *
 * There is at most one unacknowledged display frame. The UI acknowledges after
 * React commits it, so a slow/hidden tab never accumulates model copies and
 * obsolete chart updates. Training continues independently of that render rate.
 */
export function createSolverRuntime(
  send: (event: SolverEvent) => void,
  scheduler: SolverScheduler = browserScheduler,
) {
  let engine: ReturnType<typeof createEngine> | null = null;
  let dataset: Dataset,
    config: Configuration,
    page: PageId = "home";
  let running = false,
    disposed = false,
    dirty = false,
    controlId = 0,
    frameId = 0;
  let awaitingAck: number | null = null,
    timer: unknown = null,
    lastFrameAt = -Infinity;
  let selected = 0;
  let diagnostics: SolverFrame["diagnostics"] = null;
  let diagnosticsIteration: number | null = null;
  let rowIndices = new Map<Dataset["data"][number], number>();

  function schedule(delay = 0) {
    if (!disposed && timer === null) timer = scheduler.schedule(tick, delay);
  }

  /**
   * Sample the response field in world coordinates inside the worker. The canvas
   * only projects/colorizes these cells: zooming or dragging never sorts KNN
   * neighbors, evaluates kernels, or runs a neural forward pass on the UI thread.
   */
  function predictionGrid(): PredictionGrid {
    if (!engine)
      throw new Error("Initialize the solver before building its graph.");
    const rows = dataset.data,
      dimensions = rows[0]?.x.length || 2,
      three = dimensions >= 3;
    if (three) {
      const snapshot = engine.snapshot(),
        kind = snapshot.model?.kind;
      return {
        cells: [],
        stepX: 0,
        stepY: 0,
        three: true,
        surfaces: buildDecisionSurfaces(dataset, {
          predict: engine.predict,
          scores: engine.decisionScores,
          linear:
            kind === "logistic" || kind === "softmax" || kind === "perceptron",
          regression: snapshot.regression,
          clustering: snapshot.clustering,
        }),
      };
    }
    const min = [0, 1, 2].map(
      (i) => Math.min(-2, ...rows.map((row) => row.x[i] || 0)) - 0.3,
    );
    const max = [0, 1, 2].map(
      (i) => Math.max(2, ...rows.map((row) => row.x[i] || 0)) + 0.3,
    );
    const resolution = 32,
      cells: PredictionGrid["cells"] = [];
    // Extend 2D coverage for the supported zoom-out view. Outside this finite
    // sampled domain the points/axes remain interactive, without inventing values.
    if (!three)
      for (const i of [0, 1]) {
        const margin = (max[i] - min[i]) * 0.5;
        min[i] -= margin;
        max[i] += margin;
      }
    const stepX = (max[0] - min[0]) / resolution,
      stepY = (max[1] - min[1]) / resolution;
    for (let y = 0; y < resolution; y++)
      for (let x = 0; x < resolution; x++) {
        const input = Array<number>(dimensions).fill(0);
        input[0] = min[0] + x * stepX;
        input[1] = min[1] + y * stepY;
        cells.push({ input, value: engine.predict(input) });
      }
    return { cells, stepX, stepY, three, surfaces: [] };
  }

  function publish() {
    if (!engine || awaitingAck !== null) return;
    const neuralMethod = NEURAL_METHODS.includes(config.method);
    const showNetwork =
      neuralMethod && ["home", "train", "visual", "tutorial"].includes(page);
    const showTrace = showNetwork || page === "tutorial";
    const trace =
      showTrace && !engine.snapshot().issue
        ? engine.trace()
        : { status: "", steps: [] };
    const snapshot = engine.snapshot();
    // An exact loss slice evaluates hundreds of full-batch gradients. Refresh it
    // on pause/single-step/completion, rather than competing with live training.
    if (
      showNetwork &&
      !running &&
      snapshot.fitted &&
      diagnosticsIteration !== snapshot.iteration
    ) {
      diagnostics = engine.neuralDiagnostics();
      diagnosticsIteration = snapshot.iteration;
    }
    const frame: SolverFrame = {
      frameId: ++frameId,
      controlId,
      running,
      snapshot: {
        ...snapshot,
        // Never clone a whole forest or kernel cache onto the UI thread.
        // Its only consumers need a bounded parameter table and a few centers.
        model: null,
        train: [],
        test: [],
      },
      trainIndices: snapshot.train.map((row) => rowIndices.get(row)!),
      testIndices: snapshot.test.map((row) => rowIndices.get(row)!),
      parameters: parameterRows(snapshot.model),
      centers:
        snapshot.model?.kind === "kmeans"
          ? snapshot.model.centroids
          : snapshot.model?.kind === "gmm"
            ? snapshot.model.means
            : [],
      predictions: snapshot.fitted
        ? dataset.data.map((row) => engine!.predict(row.x))
        : [],
      plot:
        snapshot.fitted && ["home", "data", "visual", "tutorial"].includes(page)
          ? predictionGrid()
          : null,
      trace,
      neural: showNetwork ? engine.neuralVisualization() : null,
      diagnostics: showNetwork ? diagnostics : null,
      diagnosticsIteration: showNetwork ? diagnosticsIteration : null,
      inspection:
        page === "tutorial" && snapshot.fitted
          ? engine.visualization(selected)
          : null,
      inspectionIndex: selected,
    };
    awaitingAck = frame.frameId;
    dirty = false;
    lastFrameAt = scheduler.now();
    send({ type: "frame", frame });
  }

  function tick() {
    timer = null;
    if (disposed || !engine) return;
    try {
      if (running) {
        const started = scheduler.now();
        let count = 0;
        do {
          const result = engine.step();
          count++;
          if (result.iteration >= result.total) {
            running = false;
            break;
          }
        } while (
          count < MAX_STEPS_PER_SLICE &&
          scheduler.now() - started < SLICE_BUDGET_MS
        );
        dirty = true;
      }
      if (
        dirty &&
        awaitingAck === null &&
        (!running || scheduler.now() - lastFrameAt >= FRAME_INTERVAL_MS)
      )
        publish();
      if (running) schedule(4);
      else if (dirty && awaitingAck === null) schedule();
    } catch (cause) {
      running = false;
      dirty = false;
      send({
        type: "error",
        controlId,
        message: cause instanceof Error ? cause.message : String(cause),
      });
    }
  }

  function handle(command: SolverCommand) {
    if (disposed) return;
    if (command.type === "initialize") {
      dataset = command.dataset;
      config = command.config;
      page = command.page;
      rowIndices = new Map(dataset.data.map((row, i) => [row, i]));
      engine = createEngine(dataset, config);
      dirty = true;
      schedule();
      return;
    }
    if (!engine) return;
    switch (command.type) {
      case "ack":
        if (command.frameId === awaitingAck) {
          awaitingAck = null;
          if (dirty) schedule();
        }
        break;
      case "page":
        page = command.page;
        dirty = true;
        schedule();
        break;
      case "inspect":
        selected = command.index;
        dirty = true;
        schedule();
        break;
      case "predict":
        send({
          type: "prediction",
          requestId: command.requestId,
          value: engine.snapshot().fitted
            ? engine.predict(command.input)
            : null,
        });
        break;
      case "run":
        controlId = command.controlId;
        if (engine.snapshot().iteration >= engine.snapshot().total) {
          engine.reset();
          diagnostics = null;
          diagnosticsIteration = null;
        }
        running = true;
        dirty = true;
        schedule();
        break;
      case "pause":
        controlId = command.controlId;
        running = false;
        dirty = true;
        schedule();
        break;
      case "step":
        controlId = command.controlId;
        running = false;
        try {
          engine.step();
          dirty = true;
          schedule();
        } catch (cause) {
          send({
            type: "error",
            controlId,
            message: cause instanceof Error ? cause.message : String(cause),
          });
        }
        break;
    }
  }
  return {
    handle,
    dispose() {
      disposed = true;
      running = false;
      if (timer !== null) scheduler.cancel(timer);
      timer = null;
    },
  };
}
