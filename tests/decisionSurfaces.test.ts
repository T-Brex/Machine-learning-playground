import assert from "node:assert/strict";
import test from "node:test";
import {
  buildDecisionSurfaces,
  spatialDomain,
} from "../src/engine/decisionSurfaces.ts";
import type { SurfaceField } from "../src/engine/decisionSurfaces.ts";
import { createEngine } from "../src/engine/createEngine.ts";
import { generateDataset } from "../src/data/datasets.ts";
import { defaultParameters } from "../src/data/methods.ts";
import type { Configuration, Dataset, MethodId, Vector } from "../src/types.ts";

const dataset: Dataset = {
  classes: 2,
  data: [
    { x: [-1, -1, -1], y: 0 },
    { x: [1, 1, 1], y: 1 },
  ],
};
function field(scores: (input: Vector) => Vector, linear = true): SurfaceField {
  return {
    scores,
    linear,
    regression: false,
    clustering: false,
    predict(input) {
      const values = scores(input);
      return values.indexOf(Math.max(...values));
    },
  };
}

test("binary linear boundaries are full clipped hyperplanes, not three horizontal layers", () => {
  const faces = buildDecisionSurfaces(
    dataset,
    field((x) => [-x[0], x[0]]),
  );
  assert.equal(faces.length, 1);
  assert.equal(faces[0].vertices.length, 4);
  assert.deepEqual(faces[0].classes, [0, 1]);
  for (const p of faces[0].vertices) assert.ok(Math.abs(p[0]) < 1e-8);
  for (const axis of [1, 2])
    assert.ok(
      Math.max(...faces[0].vertices.map((p) => p[axis])) -
        Math.min(...faces[0].vertices.map((p) => p[axis])) >
        4.5,
    );
});
test("tilted planes and nonzero intercepts stay on the learned decision equation", () => {
  const margin = (x: Vector) => x[0] + 2 * x[1] - 3 * x[2] + 0.4;
  const faces = buildDecisionSurfaces(
    dataset,
    field((x) => [0, margin(x)]),
  );
  assert.equal(faces.length, 1);
  for (const vertex of faces[0].vertices)
    assert.ok(Math.abs(margin(vertex)) < 1e-8);
});
test("multiclass hyperplanes are clipped to regions where the tied classes beat every competitor", () => {
  const scores = (x: Vector) => [x[0], x[1], 0];
  const faces = buildDecisionSurfaces(
    { ...dataset, classes: 3 },
    field(scores),
  );
  assert.equal(faces.length, 3);
  for (const face of faces)
    for (const vertex of face.vertices) {
      const values = scores(vertex),
        [a, b] = face.classes;
      assert.ok(Math.abs(values[a] - values[b]) < 1e-8);
      assert.ok(
        values.every((score) => values[a] >= score - 1e-8),
        "A third class must not win on a visible pairwise boundary.",
      );
    }
});
test("hidden dimensions are fixed at dataset means when slicing a higher-dimensional hyperplane", () => {
  const higher = {
    ...dataset,
    data: dataset.data.map((row) => ({ ...row, x: [...row.x, 2] })),
  };
  const faces = buildDecisionSurfaces(
    higher,
    field((x) => [0, x[0] + x[3] - 1]),
  );
  assert.equal(faces.length, 1);
  for (const p of faces[0].vertices) assert.ok(Math.abs(p[0] + 1) < 1e-8);
});
test("identical score fields do not invent a separating plane", () => {
  assert.deepEqual(
    buildDecisionSurfaces(
      dataset,
      field(() => [0, 0]),
    ),
    [],
  );
});
test("nonlinear decision boundaries form a bounded mesh on the zero-margin surface", () => {
  const sphere = (x: Vector) =>
    x.reduce((sum, value) => sum + value * value, 0) - 1;
  const faces = buildDecisionSurfaces(
    dataset,
    field((x) => [0, sphere(x)], false),
  );
  const [min, max] = spatialDomain(dataset);
  assert.ok(faces.length > 100);
  for (const face of faces) {
    assert.equal(face.vertices.length, 3);
    for (const p of face.vertices) {
      assert.ok(
        Math.abs(sphere(p)) < 0.17,
        "Interpolation must approximate the learned curved boundary.",
      );
      p.forEach((value, i) =>
        assert.ok(Number.isFinite(value) && value >= min[i] && value <= max[i]),
      );
    }
  }
});
test("label-only classifiers get a boundary mesh, and clustering does not invent supervised class planes", () => {
  const labelField: SurfaceField = {
    ...field((x) => [0, x[0]], false),
    scores: () => null,
  };
  assert.ok(buildDecisionSurfaces(dataset, labelField).length > 0);
  assert.deepEqual(
    buildDecisionSurfaces(dataset, { ...labelField, clustering: true }),
    [],
  );
});

// Boundary extraction must use the same score ordering as actual model inference.
for (const method of [
  "logistic",
  "softmax",
  "perceptron",
  "svmrbf",
  "svmpoly",
  "mlp",
  "adam",
  "boost",
  "naive",
] as MethodId[]) {
  test(`${method}: decision scores agree with predicted class`, () => {
    const config: Configuration = {
      task: "classification",
      method,
      dataset: "linear",
      featureCount: 3,
      sampleCount: 40,
      classCount: 2,
      noise: 0.2,
      split: 0.8,
      nnOutput: "binary",
      parameters: {
        ...defaultParameters(method),
        threshold: 0.8,
        nEstimators: 3,
      },
    };
    const data = generateDataset(config),
      engine = createEngine(data, config);
    for (let i = 0; i < 20; i++) engine.step();
    for (const row of data.data) {
      const scores = engine.decisionScores(row.x);
      assert.ok(scores);
      const ordered = scores.slice().sort((a, b) => b - a);
      if (ordered[0] - ordered[1] < 1e-9) continue;
      assert.equal(scores.indexOf(Math.max(...scores)), engine.predict(row.x));
    }
  });
}
