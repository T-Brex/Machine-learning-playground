import { useMemo, useRef, useState } from "react";
import { MULTICLASS_DATASETS, TASKS } from "../data/catalog.ts";
import { parseCsv } from "../data/csv.ts";
import { generateDataset } from "../data/datasets.ts";
import { defaultParameters } from "../data/methods.ts";
import { useSolver } from "./useSolver.ts";
import type {
  Configuration,
  MethodId,
  ParameterValue,
  PageId,
  TaskId,
} from "../types.ts";

const INITIAL: Configuration = {
  task: "classification",
  method: "logistic",
  dataset: "linear",
  featureCount: 2,
  sampleCount: 120,
  classCount: 3,
  noise: 0.35,
  split: 0.8,
  nnOutput: "auto",
  parameters: defaultParameters("logistic"),
};

/**
 * Own experiment settings and delegate every numerical operation to useSolver.
 * The worker persists across navigation and is replaced on configuration/data
 * changes. The synchronous facade below only reads the latest committed display
 * frame: it never fits, predicts, or computes gradients on the main thread.
 */
export function usePlayground(page: PageId) {
  const [config, setConfig] = useState(INITIAL);
  const [dataset, setDataset] = useState(() => generateDataset(INITIAL));
  const [error, setError] = useState("");
  const [resetKey, setResetKey] = useState(0);
  const solver = useSolver(dataset, config, page, resetKey);
  const frame = solver.frame,
    snapshot = frame.snapshot;
  const fileRequest = useRef(0);
  const engine = useMemo(
    () => ({
      snapshot: () => snapshot,
      predict: (input: number[]) =>
        frame.predictions[dataset.data.findIndex((row) => row.x === input)] ??
        NaN,
      trace: () => frame.trace,
      neuralVisualization: () => frame.neural,
      neuralDiagnostics: () => frame.diagnostics,
      visualization: (index: number) =>
        frame.inspectionIndex === index ? frame.inspection : null,
    }),
    [frame, snapshot, dataset],
  );

  /**
   * Replace experiment settings and optionally generate new synthetic rows.
   *
   * Every edit stops the timer and invalidates pending CSV reads before updating
   * configuration. Switching a binary logistic model to a multiclass dataset also
   * selects softmax and its defaults atomically, avoiding an incompatible transient
   * experiment. Dataset regeneration is explicit so a parameter edit can reuse rows.
   */
  function update(patch: Partial<Configuration>, regenerate = false) {
    fileRequest.current++;
    solver.cancel();
    setError("");
    const next = { ...config, ...patch };
    if (
      next.method === "logistic" &&
      MULTICLASS_DATASETS.includes(next.dataset) &&
      Number(next.classCount) > 2
    ) {
      next.method = "softmax";
      next.parameters = defaultParameters("softmax");
    }
    setConfig(next);
    if (regenerate) setDataset(generateDataset(next));
  }
  /** Choose a compatible default model and dataset together, avoiding mixed task configurations. */
  function selectTask(task: TaskId) {
    const method = TASKS[task].preferred;
    const next: Configuration = {
      ...config,
      task,
      method,
      parameters: defaultParameters(method),
      dataset: task === "regression" ? "regression" : "linear",
    };
    fileRequest.current++;
    solver.cancel();
    setError("");
    setConfig(next);
    setDataset(generateDataset(next));
  }
  /**
   * Load a selected file without allowing stale asynchronous results to win.
   *
   * Capture a monotonically increasing request token before awaiting file.text().
   * Any later settings change, task selection, or import invalidates that token. When
   * the read finishes, only the current request may replace the dataset or error.
   * Regression preserves raw continuous targets; classification maps numeric labels
   * into a contiguous vocabulary and retains their original values for display.
   */
  async function importCsv(file: File | undefined) {
    if (!file) return;
    const request = ++fileRequest.current;
    solver.pause();
    try {
      const imported = parseCsv(
        await file.text(),
        config.task === "regression",
      );
      if (request !== fileRequest.current) return;
      const method =
        config.method === "logistic" && imported.classes > 2
          ? "softmax"
          : config.method;
      setConfig({
        ...config,
        method,
        featureCount: imported.data[0].x.length,
        parameters:
          method === config.method
            ? config.parameters
            : defaultParameters(method),
      });
      setDataset(imported);
      setError("");
    } catch (cause) {
      if (request === fileRequest.current)
        setError(cause instanceof Error ? cause.message : String(cause));
    }
  }
  return {
    config,
    dataset,
    engine,
    snapshot,
    running: solver.running,
    ready: solver.ready,
    busy: solver.busy,
    plot: frame.plot,
    predictions: frame.predictions,
    parameters: frame.parameters,
    centers: frame.centers,
    diagnosticsIteration: frame.diagnosticsIteration,
    predict: solver.predict,
    inspect: solver.inspect,
    error: error || solver.error,
    update,
    selectTask,
    importCsv,
    step: solver.step,
    train: solver.train,
    pause: solver.pause,
    reset: () => {
      solver.cancel();
      fileRequest.current++;
      setResetKey((value) => value + 1);
      setError("");
    },
    selectMethod: (method: MethodId) =>
      update({ method, parameters: defaultParameters(method) }),
    parameter: (key: string, value: ParameterValue) =>
      update({ parameters: { ...config.parameters, [key]: value } }),
  };
}
