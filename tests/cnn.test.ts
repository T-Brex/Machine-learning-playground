import assert from "node:assert/strict";
import test from "node:test";
import {
  createCnnModel,
  forwardCnn,
  parseMnistImages,
  parseMnistLabels,
  trainCnnBatch,
  traceCnnPixel,
  traceDenseHidden,
  traceDenseOutput,
} from "../src/engine/cnn.ts";

test("CNN exposes both convolution blocks, dense layers, and ten normalized scores", () => {
  const model = createCnnModel("max", () => 0.75),
    image = Array<number>(28 * 28).fill(0);
  image[14 * 28 + 14] = 255;
  const result = forwardCnn(model, image);
  assert.deepEqual(
    [
      result.convolution1[0].length,
      result.convolution1[0][0].length,
      result.pooled1[0].length,
      result.convolution2[0].length,
      result.pooled2[0].length,
    ],
    [26, 26, 13, 11, 5],
  );
  assert.equal(result.convolution1.length, 4);
  assert.equal(result.convolution2.length, 4);
  assert.equal(result.flattened.length, 100);
  assert.equal(result.denseHidden.length, 16);
  assert.equal(result.probabilities.length, 10);
  assert.ok(
    Math.abs(result.probabilities.reduce((sum, value) => sum + value, 0) - 1) <
      1e-12,
  );
});

test("one CNN mini-batch updates both convolutions and both dense layers", () => {
  let seed = 123;
  const model = createCnnModel("max", () => {
      seed = (seed * 16807) % 2147483647;
      return seed / 2147483647;
    }),
    before = {
      filter1: model.filters1[0].slice(),
      filter2: model.filters2[0][0].slice(),
      denseHidden: model.denseHiddenWeights[0].slice(),
      denseOutput: model.denseOutputWeights[0].slice(),
    },
    example = {
      image: Array<number>(28 * 28).fill(180),
      label: 3,
    };
  const result = trainCnnBatch(model, [example], 0.05);
  assert.ok(Number.isFinite(result.loss));
  assert.ok(result.accuracy === 0 || result.accuracy === 1);
  assert.notDeepEqual(model.filters1[0], before.filter1);
  assert.notDeepEqual(model.filters2[0][0], before.filter2);
  assert.notDeepEqual(model.denseHiddenWeights[0], before.denseHidden);
  assert.notDeepEqual(model.denseOutputWeights[0], before.denseOutput);

  const averageModel = createCnnModel("average", () => 0.75),
    averageFilter1 = averageModel.filters1[0].slice(),
    averageFilter2 = averageModel.filters2[0][0].slice();
  averageModel.denseOutputWeights = Array.from({ length: 10 }, (_, output) =>
    Array<number>(16).fill((output + 1) * 0.01),
  );
  trainCnnBatch(averageModel, [example], 0.05);
  assert.notDeepEqual(averageModel.filters1[0], averageFilter1);
  assert.notDeepEqual(averageModel.filters2[0][0], averageFilter2);
});

test("max and average pooling expose correct pixel calculations for both blocks", () => {
  const image = Array<number>(28 * 28).fill(120),
    maxModel = createCnnModel("max", () => 0.75),
    maxForward = forwardCnn(maxModel, image),
    maxTrace = traceCnnPixel(
      maxModel,
      image,
      maxForward,
      1,
      0,
      10,
      13,
      "convolution",
    );
  assert.equal(maxTrace.inputPatches.length, 1);
  assert.equal(maxTrace.convolutionValue, maxForward.convolution1[0][10][13]);
  assert.equal(maxTrace.reluValue, maxForward.relu1[0][10][13]);
  assert.ok(maxTrace.poolValue !== null);
  assert.equal(
    maxTrace.poolValue,
    maxForward.pooled1[0][maxTrace.poolRow!][maxTrace.poolColumn!],
  );
  assert.ok(maxTrace.poolWinner !== null);
  assert.equal(
    maxTrace.poolCandidates
      .flat()
      .some(
        (candidate) =>
          candidate.row === maxTrace.poolWinner!.row &&
          candidate.column === maxTrace.poolWinner!.column &&
          candidate.relu === maxTrace.poolValue,
      ),
    true,
  );

  const averageModel = createCnnModel("average", () => 0.75),
    averageForward = forwardCnn(averageModel, image),
    averageTrace = traceCnnPixel(
      averageModel,
      image,
      averageForward,
      2,
      0,
      5,
      5,
      "convolution",
    );
  assert.equal(averageTrace.inputPatches.length, 4);
  assert.equal(averageTrace.filters.length, 4);
  assert.equal(
    averageTrace.convolutionValue,
    averageForward.convolution2[0][5][5],
  );
  assert.equal(averageTrace.poolWinner, null);
  assert.equal(averageTrace.poolValue, averageForward.pooled2[0][2][2]);
  assert.equal(averageTrace.poolCandidates.flat().length, 4);

  const borderTrace = traceCnnPixel(
    averageModel,
    image,
    averageForward,
    2,
    0,
    10,
    10,
    "convolution",
  );
  assert.equal(borderTrace.poolValue, null);
  assert.equal(borderTrace.poolRow, null);
});

test("dense traces show every weighted input and match forward activations", () => {
  const model = createCnnModel("max", () => 0.75),
    forward = forwardCnn(model, Array<number>(28 * 28).fill(180)),
    hidden = traceDenseHidden(model, forward, 2),
    output = traceDenseOutput(model, forward, 3);
  assert.equal(hidden.inputs.length, 100);
  assert.equal(hidden.products.length, 100);
  assert.equal(hidden.value, forward.denseHiddenPreActivation[2]);
  assert.equal(hidden.activation, forward.denseHidden[2]);
  assert.equal(output.inputs.length, 16);
  assert.equal(output.products.length, 16);
  assert.equal(output.value, forward.logits[3]);
  assert.throws(() => traceDenseHidden(model, forward, 16), /neurone denso/);
  assert.throws(() => traceDenseOutput(model, forward, 10), /classe densa/);
});

test("MNIST IDX parser selects a balanced subset and rejects mismatched files", () => {
  const imageCount = 4,
    imageBuffer = new ArrayBuffer(16 + imageCount * 28 * 28),
    imageView = new DataView(imageBuffer),
    labelBuffer = new ArrayBuffer(8 + imageCount),
    labelView = new DataView(labelBuffer);
  imageView.setUint32(0, 2051);
  imageView.setUint32(4, imageCount);
  imageView.setUint32(8, 28);
  imageView.setUint32(12, 28);
  labelView.setUint32(0, 2049);
  labelView.setUint32(4, imageCount);
  new Uint8Array(labelBuffer).set([0, 1, 0, 1], 8);
  const labels = parseMnistLabels(labelBuffer),
    samples = parseMnistImages(imageBuffer, labels, 1);
  assert.deepEqual(
    samples.map((sample) => sample.label),
    [0, 1],
  );
  assert.equal(samples[0].image.length, 28 * 28);
  assert.throws(
    () => parseMnistImages(imageBuffer, labels.subarray(0, 9), 1),
    /non corrispondono/,
  );
});
