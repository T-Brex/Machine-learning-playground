import test from "node:test";
import assert from "node:assert/strict";
import { createEngine } from "../src/engine/createEngine.js";
import { generateDataset } from "../src/data/datasets.js";
import { defaultParameters, PARAMS } from "../src/data/methods.js";
import { TASKS } from "../src/data/catalog.js";
import { parseCsv } from "../src/data/csv.js";

function settings(method, dataset = "linear", extra = {}) {
  return {
    method,
    dataset,
    noise: 0.15,
    sampleCount: 40,
    classCount: 3,
    featureCount: 3,
    split: 0.8,
    nnOutput: "auto",
    parameters: {
      ...defaultParameters(method),
      nEstimators: 3,
      maxIter: 3,
      epochs: 3,
      ...extra,
    },
  };
}
// Exercise every advertised algorithm, including explanations before fitting.
for (const method of Object.keys(PARAMS)) {
  test(`${method}: train, predict, evaluate and explain`, () => {
    const kind =
      TASKS.regression.methods.includes(method) &&
      !["mlp", "boost"].includes(method)
        ? "regression"
        : "linear";
    const config = settings(method, kind);
    const dataset = generateDataset(config),
      engine = createEngine(dataset, config);
    assert.ok(engine.trace().steps.length > 0);
    engine.step();
    const result = engine.snapshot();
    assert.equal(result.iteration, 1);
    assert.ok(Number.isFinite(result.loss));
    assert.ok(Number.isFinite(engine.predict(dataset.data[0].x)));
    assert.ok(Object.keys(result.metrics).length > 0);
    assert.ok(engine.trace().steps.length > 0);
    assert.ok(engine.visualization(0));
  });
}
test("regression converges and engines keep independent models", () => {
  const config = settings("linear", "regression", { learningRate: 0.08 });
  const data = {
    classes: 1,
    data: Array.from({ length: 100 }, (_, i) => {
      const x = (i - 50) / 20;
      return { x: [x, x * x], y: 2 * x + 1 };
    }),
  };
  const a = createEngine(data, config),
    b = createEngine(data, config);
  a.step();
  const before = a.snapshot().loss;
  for (let i = 0; i < 100; i++) a.step();
  assert.ok(a.snapshot().loss < before * 0.05);
  assert.equal(b.snapshot().iteration, 0);
  const all = new Set([...a.snapshot().train, ...a.snapshot().test]);
  assert.equal(all.size, data.data.length);
  assert.equal(
    a.snapshot().train.filter((row) => a.snapshot().test.includes(row)).length,
    0,
  );
});
test("multiclass logistic is blocked; softmax accepts the same data", () => {
  const config = settings("logistic", "threeblobs");
  const data = generateDataset(config);
  assert.throws(() => createEngine(data, config).step(), /binaria/);
  config.method = "softmax";
  config.parameters = defaultParameters("softmax");
  assert.equal(createEngine(data, config).step().iteration, 1);
});
test("CSV preserves regression targets and maps categorical numeric targets", () => {
  const csv = "x1;x2;y\n0;1;10\n1;2;20\n2;3;10\n3;4;20";
  const regression = parseCsv(csv, true),
    classification = parseCsv(csv);
  assert.equal(regression.data[0].y, 10);
  assert.equal(regression.classes, 1);
  assert.deepEqual(classification.labels, [10, 20]);
  assert.equal(classification.classes, 2);
  assert.equal(parseCsv("0,1,10\n1,2,20\n2,3,10\n3,4,20").data.length, 4);
  assert.throws(() => parseCsv("x,y,z\n1,,2\n1,2,3\n1,2,3\n1,2,3"), /mancanti/);
});

// The same nonlinear model families must handle continuous targets as well as labels.
for (const method of ["boost", "mlp"]) {
  test(`${method}: regression output remains continuous`, () => {
    const config = settings(method, "regression");
    const engine = createEngine(generateDataset(config), config);
    engine.step();
    assert.equal(engine.snapshot().regression, true);
    assert.ok(Number.isFinite(engine.snapshot().metrics.MSE));
    assert.ok(engine.trace().steps.length);
    assert.equal(engine.visualization().kind, "regression");
  });
}
test("neural diagnostics preserve parameters and report the current objective at the slice center", () => {
  const config = settings("adam");
  const engine = createEngine(generateDataset(config), config);
  engine.step();
  const before = structuredClone(engine.snapshot().model);
  const diagnostics = engine.neuralDiagnostics();
  assert.deepEqual(engine.snapshot().model, before);
  assert.ok(
    diagnostics.parameters.every((row) => Number.isFinite(row.gradient)),
  );
  assert.ok(
    Math.abs(diagnostics.surface[7][7] - engine.snapshot().loss) < 1e-10,
  );
});
