import { useEffect, useMemo, useRef, useState } from "react";
import { createEngine } from "../engine/createEngine.js";
import { generateDataset } from "../data/datasets.js";
import { defaultParameters } from "../data/methods.js";
import { TASKS, MULTICLASS_DATASETS } from "../data/catalog.js";
import { parseCsv } from "../data/csv.js";

const INITIAL = {
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

/** React owns settings and scheduling; each engine owns one experiment's model. */
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
        setError(cause.message);
        setRunning(false);
      }
    }, 16);
    return () => clearInterval(timer);
  }, [engine, running]);

  /** Stop training before replacing settings; dataset controls also regenerate synthetic rows. */

  function update(patch, regenerate = false) {
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
  function selectTask(task) {
    const method = TASKS[task].preferred;
    const next = {
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
  /** Ignore a late file read if settings or another import changed while the file was loading. */
  async function importCsv(file) {
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
      if (request === fileRequest.current) setError(cause.message);
    }
  }
  /** Commit exactly one numerical update and request a React snapshot refresh. */
  function step() {
    try {
      engine.step();
      setRevision((value) => value + 1);
      setError("");
    } catch (cause) {
      setError(cause.message);
    }
  }
  /** Resume an unfinished run, or replace a completed engine before starting again. */
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
    selectMethod: (method) =>
      update({ method, parameters: defaultParameters(method) }),
    parameter: (key, value) =>
      update({ parameters: { ...config.parameters, [key]: value } }),
  };
}
