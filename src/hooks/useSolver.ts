import {
  startTransition,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { Configuration, Dataset, PageId, Vector } from "../types.ts";
import { emptyFrame } from "../solver/protocol.ts";
import type {
  SolverCommand,
  SolverEvent,
  SolverFrame,
} from "../solver/protocol.ts";

interface Session {
  worker: Worker;
  pending: Map<
    number,
    { resolve: (value: number | null) => void; reject: (reason: Error) => void }
  >;
}

/**
 * Manage one worker per experiment, with no synchronous numerical work in React.
 *
 * Effects clean up during reset, settings changes, unmount, and StrictMode's
 * development remount. A session identity check drops late messages from an old
 * worker, while control IDs drop a queued "running" frame after a pause request.
 * Display state is a transition; input/pointer state remains urgent.
 */
export function useSolver(
  dataset: Dataset,
  config: Configuration,
  page: PageId,
  resetKey: number,
) {
  const experiment = useMemo(() => ({}), [dataset, config, resetKey]);
  const [received, setReceived] = useState<{
    experiment: object;
    frame: SolverFrame;
  } | null>(null);
  const [control, setControl] = useState({
    experiment,
    controlId: 0,
    running: false,
    busy: false,
    error: "",
  });
  const session = useRef<Session | null>(null),
    sequence = useRef(0),
    requestId = useRef(0);
  const pageRef = useRef(page);
  pageRef.current = page;

  const cancel = useCallback(() => {
    const current = session.current;
    session.current = null;
    if (current) {
      current.worker.terminate();
      for (const pending of current.pending.values())
        pending.reject(new DOMException("Experiment replaced", "AbortError"));
      current.pending.clear();
    }
  }, []);

  useEffect(() => {
    sequence.current = 0;
    const worker = new Worker(
      new URL("../solver/solver.worker.ts", import.meta.url),
      { type: "module" },
    );
    const current: Session = { worker, pending: new Map() };
    session.current = current;
    worker.onmessage = (event: MessageEvent<SolverEvent>) => {
      if (session.current !== current) return;
      const message = event.data;
      if (message.type === "frame") {
        if (message.frame.controlId < sequence.current) {
          // Even rejected frames return their credit, otherwise the worker would
          // wait forever for a React commit that will deliberately never happen.
          worker.postMessage({
            type: "ack",
            frameId: message.frame.frameId,
          } satisfies SolverCommand);
          return;
        }
        const frame = message.frame;
        // Structured cloning loses reference identity. Restore rows from the UI's
        // dataset so tutorial selection still maps between shuffled and source rows.
        frame.snapshot.train = frame.trainIndices.map((i) => dataset.data[i]);
        frame.snapshot.test = frame.testIndices.map((i) => dataset.data[i]);
        startTransition(() => setReceived({ experiment, frame }));
      } else if (message.type === "prediction") {
        current.pending.get(message.requestId)?.resolve(message.value);
        current.pending.delete(message.requestId);
      } else if (message.controlId >= sequence.current) {
        setControl({
          experiment,
          controlId: message.controlId,
          running: false,
          busy: false,
          error: message.message,
        });
      }
    };
    worker.onerror = (event) => {
      if (session.current === current)
        setControl({
          experiment,
          controlId: sequence.current,
          running: false,
          busy: false,
          error: event.message || "Impossibile avviare il solver.",
        });
      for (const pending of current.pending.values())
        pending.reject(new Error(event.message));
      current.pending.clear();
    };
    worker.postMessage({
      type: "initialize",
      dataset,
      config,
      page: pageRef.current,
    } satisfies SolverCommand);
    return () => {
      if (session.current === current) cancel();
    };
  }, [experiment, dataset, config, cancel]);

  const frame = received?.experiment === experiment ? received.frame : null;
  // Acknowledging after commit, rather than immediately on message receipt,
  // bounds React's render queue to one model frame under CPU or tab throttling.
  useEffect(() => {
    if (frame)
      session.current?.worker.postMessage({
        type: "ack",
        frameId: frame.frameId,
      } satisfies SolverCommand);
  }, [frame]);
  useEffect(() => {
    session.current?.worker.postMessage({
      type: "page",
      page,
    } satisfies SolverCommand);
  }, [page]);

  const sendControl = useCallback(
    (type: "run" | "pause" | "step") => {
      if (!session.current) return;
      const controlId = ++sequence.current;
      setControl({
        experiment,
        controlId,
        running: type === "run" || type === "pause",
        busy: true,
        error: "",
      });
      session.current.worker.postMessage({
        type,
        controlId,
      } satisfies SolverCommand);
    },
    [experiment],
  );
  const inspect = useCallback((index: number) => {
    session.current?.worker.postMessage({
      type: "inspect",
      index,
    } satisfies SolverCommand);
  }, []);
  const predict = useCallback((input: Vector): Promise<number | null> => {
    const current = session.current;
    if (!current) return Promise.resolve(null);
    const id = ++requestId.current;
    return new Promise((resolve, reject) => {
      current.pending.set(id, { resolve, reject });
      current.worker.postMessage({
        type: "predict",
        requestId: id,
        input,
      } satisfies SolverCommand);
    });
  }, []);
  // Optimistic controls cover only commands the worker has not acknowledged.
  // Acknowledged status is derived from the SAME committed frame as the graphs.
  // Updating "paused" urgently while its snapshot was still a transition let
  // the displayed iteration advance once after the pause button was disabled.
  const visibleControl =
    control.experiment !== experiment
      ? { running: false, busy: false, error: "" }
      : control.error || !frame || control.controlId > frame.controlId
        ? control
        : { running: frame.running, busy: false, error: "" };
  return {
    frame: frame ?? emptyFrame(),
    ready: frame !== null,
    ...visibleControl,
    cancel,
    inspect,
    predict,
    train: () => sendControl("run"),
    pause: () => sendControl("pause"),
    step: () => sendControl("step"),
  };
}
