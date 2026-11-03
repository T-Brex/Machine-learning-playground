import { createSolverRuntime } from "./runtime.ts";
import type { SolverCommand, SolverEvent } from "./protocol.ts";

/** Vite emits this module as a separate worker bundle in production. */
const runtime = createSolverRuntime((event: SolverEvent) =>
  self.postMessage(event),
);
self.onmessage = (event: MessageEvent<SolverCommand>) =>
  runtime.handle(event.data);
