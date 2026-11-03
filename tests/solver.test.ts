import assert from "node:assert/strict";
import test from "node:test";
import { generateDataset } from "../src/data/datasets.ts";
import { defaultParameters } from "../src/data/methods.ts";
import { createSolverRuntime } from "../src/solver/runtime.ts";
import type { SolverScheduler } from "../src/solver/runtime.ts";
import type { SolverEvent, SolverFrame } from "../src/solver/protocol.ts";
import type { Configuration, PageId } from "../src/types.ts";

/** Virtual time tests the protocol rather than relying on CPU-dependent sleeps. */
class Scheduler implements SolverScheduler {
  time = 0;
  nextId = 0;
  jobs = new Map<number, { at: number; run: () => void }>();
  now() {
    return this.time;
  }
  schedule(run: () => void, delay: number) {
    const id = ++this.nextId;
    this.jobs.set(id, { at: this.time + delay, run });
    return id;
  }
  cancel(id: unknown) {
    this.jobs.delete(Number(id));
  }
  next() {
    const next = [...this.jobs.entries()].sort((a, b) => a[1].at - b[1].at)[0];
    if (!next) return false;
    this.jobs.delete(next[0]);
    this.time = next[1].at;
    next[1].run();
    return true;
  }
}
function setup(epochs = 2000, featureCount = 2, page: PageId = "train") {
  const scheduler = new Scheduler(),
    events: SolverEvent[] = [];
  // Real worker postMessage clones synchronously. Emulate that ownership boundary
  // so subsequent epochs cannot mutate assertions about already-published frames.
  const runtime = createSolverRuntime(
    (event) => events.push(structuredClone(event)),
    scheduler,
  );
  const config: Configuration = {
    task: "classification",
    method: "logistic",
    dataset: "linear",
    featureCount,
    sampleCount: 40,
    classCount: 2,
    noise: 0.2,
    split: 0.8,
    nnOutput: "auto",
    parameters: { ...defaultParameters("logistic"), epochs },
  };
  runtime.handle({
    type: "initialize",
    dataset: generateDataset(config),
    config,
    page,
  });
  const frames = () =>
    events
      .filter(
        (event): event is { type: "frame"; frame: SolverFrame } =>
          event.type === "frame",
      )
      .map((e) => e.frame);
  scheduler.next();
  return { runtime, scheduler, events, frames };
}

test("worker frames are throttled and backpressured without stopping training", () => {
  const { runtime, scheduler, frames } = setup();
  const initial = frames()[0];
  assert.equal(initial.snapshot.iteration, 0);
  runtime.handle({ type: "ack", frameId: initial.frameId });
  runtime.handle({ type: "run", controlId: 1 });
  for (let i = 0; i < 45; i++) scheduler.next();
  assert.equal(frames().length, 2, "Only one frame may wait for a UI commit.");
  const running = frames()[1];
  assert.ok(running.snapshot.iteration > 0);
  assert.equal(running.running, true);
  // Further training took place while React was deliberately not acknowledging.
  runtime.handle({ type: "pause", controlId: 2 });
  scheduler.next();
  assert.equal(frames().length, 2);
  runtime.handle({ type: "ack", frameId: running.frameId });
  scheduler.next();
  const paused = frames().at(-1)!;
  assert.equal(paused.running, false);
  assert.equal(paused.controlId, 2);
  assert.ok(paused.snapshot.iteration > running.snapshot.iteration);
  runtime.handle({ type: "ack", frameId: paused.frameId });
  assert.equal(
    scheduler.next(),
    false,
    "Pause leaves no training timer behind.",
  );
  runtime.handle({ type: "step", controlId: 3 });
  scheduler.next();
  const stepped = frames().at(-1)!;
  assert.equal(stepped.snapshot.iteration, paused.snapshot.iteration + 1);
  runtime.dispose();
  assert.equal(scheduler.jobs.size, 0);
});

test("completion, resume, restart and prediction use the same worker-owned model", () => {
  const { runtime, scheduler, frames, events } = setup(12);
  runtime.handle({ type: "ack", frameId: frames()[0].frameId });
  runtime.handle({ type: "step", controlId: 1 });
  scheduler.next();
  const first = frames().at(-1)!;
  assert.equal(first.snapshot.iteration, 1);
  runtime.handle({ type: "ack", frameId: first.frameId });
  runtime.handle({ type: "run", controlId: 2 });
  for (let i = 0; i < 10; i++) if (!scheduler.next()) break;
  const completed = frames().at(-1)!;
  assert.equal(completed.snapshot.iteration, 12);
  assert.equal(completed.running, false);
  assert.equal(completed.snapshot.losses.length, 12);
  assert.equal(
    completed.snapshot.model,
    null,
    "Numerical models stay in the worker.",
  );
  assert.ok(
    completed.parameters.length > 0,
    "The UI still receives its parameter table.",
  );
  assert.equal(
    completed.trainIndices.length + completed.testIndices.length,
    40,
  );
  assert.equal(
    new Set([...completed.trainIndices, ...completed.testIndices]).size,
    40,
  );
  runtime.handle({ type: "predict", requestId: 42, input: [0.8, 0.5] });
  const result = events.at(-1);
  assert.equal(result?.type, "prediction");
  if (result?.type === "prediction")
    assert.ok(result.value === 0 || result.value === 1);
  runtime.handle({ type: "ack", frameId: completed.frameId });
  runtime.handle({ type: "run", controlId: 3 });
  for (let i = 0; i < 10; i++) if (!scheduler.next()) break;
  const restarted = frames().at(-1)!;
  assert.equal(restarted.snapshot.iteration, 12);
  assert.equal(restarted.snapshot.losses.length, 12);
  runtime.dispose();
});

test("dispose cancels scheduled work and rejects subsequent commands", () => {
  const { runtime, scheduler, frames } = setup();
  runtime.handle({ type: "ack", frameId: frames()[0].frameId });
  runtime.handle({ type: "run", controlId: 1 });
  runtime.dispose();
  assert.equal(scheduler.next(), false);
  runtime.handle({ type: "step", controlId: 2 });
  assert.equal(frames().length, 1);
});

test("classified volumes are calculated only on request and leave the learned model unchanged", () => {
  const { runtime, scheduler, frames } = setup(12, 3, "home");
  runtime.handle({ type: "ack", frameId: frames()[0].frameId });
  runtime.handle({ type: "step", controlId: 1 });
  scheduler.next();
  const before = frames().at(-1)!;
  assert.equal(before.plot?.regionsRequested, false);
  assert.deepEqual(before.plot?.regions, []);
  runtime.handle({ type: "ack", frameId: before.frameId });
  runtime.handle({ type: "regions", enabled: true });
  scheduler.next();
  const enabled = frames().at(-1)!;
  assert.equal(enabled.plot?.regionsRequested, true);
  assert.ok(enabled.plot!.regions.length > 0);
  assert.equal(enabled.snapshot.iteration, before.snapshot.iteration);
  assert.deepEqual(enabled.predictions, before.predictions);
  runtime.handle({ type: "ack", frameId: enabled.frameId });
  runtime.handle({ type: "regions", enabled: false });
  scheduler.next();
  const disabled = frames().at(-1)!;
  assert.equal(disabled.plot?.regionsRequested, false);
  assert.deepEqual(disabled.plot?.regions, []);
  assert.equal(disabled.snapshot.iteration, before.snapshot.iteration);
  runtime.dispose();
});
