import { useEffect, useMemo, useRef, useState } from "react";
import { MULTICLASS_DATASETS, TASKS } from "../data/catalog.ts";
import { parseCsv } from "../data/csv.ts";
import { generateDataset } from "../data/datasets.ts";
import { defaultParameters } from "../data/methods.ts";
import { createEngine } from "../engine/createEngine.ts";
import type {
  Configuration,
  MethodId,
  ParameterValue,
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
 * Bridge React state to one imperative numerical experiment.
 *
 * Configuration and dataset changes replace the memoized engine. Epoch updates
 * mutate that engine and increment revision so React reads a fresh snapshot. This
 * keeps large numerical matrices out of component state while preserving explicit
 * render updates. Running state controls a cleanup-managed timer, so page changes
 * can share progress while settings changes and unmounts cancel obsolete work.
 *
 * The returned controller is also the source of the Playground prop type. Keeping
 * that type derived prevents component callbacks from drifting from this API.
 */
export function usePlayground() {
  const [config, setConfig] = useState(INITIAL);
  const [dataset, setDataset] = useState(() => generateDataset(INITIAL));
  const [revision, setRevision] = useState(0);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [resetKey, setResetKey] = useState(0);
  const engine = useMemo(
    () => createEngine(dataset, config),
    [dataset, config, resetKey],
  );
  const snapshot = useMemo(() => engine.snapshot(), [engine, revision]);
  const fileRequest = useRef(0);

  // An experiment change or unmount cancels its pending timer, including StrictMode remounts.
  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => {
      try {
        const current = engine.snapshot();
        if (current.iteration >= current.total) {
          setRunning(false);
          return;
        }
        engine.step();
        setRevision((value) => value + 1);
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : String(cause));
        setRunning(false);
      }
    }, 16);
    return () => clearInterval(timer);
  }, [engine, running]);

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
    setRunning(false);
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
    setRunning(false);
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
    setRunning(false);
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
  /** Commit exactly one numerical update and request a React snapshot refresh. */
  function step() {
    try {
      engine.step();
      setRevision((value) => value + 1);
      setError("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }
  /**
   * Resume remaining work or replace an already completed experiment.
   *
   * The engine itself does not schedule callbacks. Setting running starts the effect's
   * timer, and effect cleanup cancels it on engine replacement or unmount. Completed
   * runs increment resetKey so a new model/split is created before the next tick;
   * paused runs retain their exact model and optimizer buffers.
   */
  function train() {
    if (snapshot.issue) return;
    if (snapshot.iteration >= snapshot.total) setResetKey((value) => value + 1);
    setError("");
    setRunning(true);
  }
  return {
    config,
    dataset,
    engine,
    snapshot,
    running,
    error,
    update,
    selectTask,
    importCsv,
    step,
    train,
    pause: () => setRunning(false),
    reset: () => {
      setRunning(false);
      setResetKey((value) => value + 1);
      setError("");
    },
    selectMethod: (method: MethodId) =>
      update({ method, parameters: defaultParameters(method) }),
    parameter: (key: string, value: ParameterValue) =>
      update({ parameters: { ...config.parameters, [key]: value } }),
  };
}
