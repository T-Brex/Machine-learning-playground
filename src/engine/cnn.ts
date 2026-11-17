export interface MnistDigit {
  image: number[];
  label: number;
}

export type CnnPooling = "max" | "average";
export type CnnLayer = 1 | 2;
export type CnnPixelStage = "convolution" | "relu" | "pool";

export interface CnnModel {
  pooling: CnnPooling;
  filters1: number[][];
  filterBias1: number[];
  filters2: number[][][];
  filterBias2: number[];
  denseHiddenWeights: number[][];
  denseHiddenBias: number[];
  denseOutputWeights: number[][];
  denseOutputBias: number[];
}

export interface CnnForward {
  convolution1: number[][][];
  relu1: number[][][];
  pooled1: number[][][];
  poolIndices1: number[][][];
  convolution2: number[][][];
  relu2: number[][][];
  pooled2: number[][][];
  poolIndices2: number[][][];
  flattened: number[];
  denseHiddenPreActivation: number[];
  denseHidden: number[];
  logits: number[];
  probabilities: number[];
}

export interface CnnPixelTrace {
  layer: CnnLayer;
  stage: CnnPixelStage;
  filterIndex: number;
  row: number;
  column: number;
  convolutionRow: number;
  convolutionColumn: number;
  inputPatches: number[][][];
  filters: number[][][];
  products: number[][][];
  productSum: number;
  bias: number;
  convolutionValue: number;
  reluValue: number;
  poolRow: number | null;
  poolColumn: number | null;
  poolCandidates: {
    row: number;
    column: number;
    convolution: number;
    relu: number;
  }[][];
  poolValue: number | null;
  poolWinner: { row: number; column: number } | null;
}

export interface DenseTrace {
  index: number;
  inputs: number[];
  weights: number[];
  products: number[];
  productSum: number;
  bias: number;
  value: number;
  activation?: number;
}

const IMAGE_SIZE = 28;
const FILTER_SIZE = 3;
const FILTER_COUNT = 4;
const HIDDEN_COUNT = 16;
const CONV1_SIZE = IMAGE_SIZE - FILTER_SIZE + 1;
const POOL1_SIZE = Math.floor(CONV1_SIZE / 2);
const CONV2_SIZE = POOL1_SIZE - FILTER_SIZE + 1;
const POOL2_SIZE = Math.floor(CONV2_SIZE / 2);
const DENSE_INPUTS = FILTER_COUNT * POOL2_SIZE * POOL2_SIZE;

export function parseMnistImages(
  buffer: ArrayBuffer,
  labels: Uint8Array,
  perClass = 80,
): MnistDigit[] {
  const view = new DataView(buffer);
  if (
    buffer.byteLength < 16 ||
    view.getUint32(0) !== 2051 ||
    view.getUint32(8) !== IMAGE_SIZE ||
    view.getUint32(12) !== IMAGE_SIZE
  )
    throw new Error("Il file immagini MNIST non ha un formato IDX valido.");
  const count = view.getUint32(4),
    expectedBytes = 16 + count * IMAGE_SIZE * IMAGE_SIZE;
  if (buffer.byteLength !== expectedBytes || labels.byteLength !== 8 + count)
    throw new Error("I file MNIST sono incompleti o non corrispondono.");
  const labelView = new DataView(
    labels.buffer,
    labels.byteOffset,
    labels.byteLength,
  );
  if (labelView.getUint32(0) !== 2049 || labelView.getUint32(4) !== count)
    throw new Error(
      "Il file delle etichette MNIST non corrisponde alle immagini.",
    );
  const selected: MnistDigit[] = [],
    counts = Array<number>(10).fill(0),
    pixels = new Uint8Array(buffer, 16);
  for (let i = 0; i < count; i++) {
    const label = labels[8 + i];
    if (label > 9) throw new Error("MNIST contiene un'etichetta non valida.");
    if (counts[label] >= perClass) continue;
    const start = i * IMAGE_SIZE * IMAGE_SIZE;
    selected.push({
      label,
      image: Array.from(
        pixels.subarray(start, start + IMAGE_SIZE * IMAGE_SIZE),
      ),
    });
    counts[label]++;
    if (counts.every((value) => value >= perClass)) break;
  }
  if (selected.length === 0)
    throw new Error("Il dataset MNIST non contiene immagini utilizzabili.");
  return selected;
}

export function parseMnistLabels(buffer: ArrayBuffer): Uint8Array {
  if (buffer.byteLength < 8)
    throw new Error("Il file delle etichette MNIST è incompleto.");
  const view = new DataView(buffer);
  if (view.getUint32(0) !== 2049 || buffer.byteLength !== 8 + view.getUint32(4))
    throw new Error(
      "Il file delle etichette MNIST non ha un formato IDX valido.",
    );
  return new Uint8Array(buffer);
}

export function createCnnModel(
  pooling: CnnPooling = "max",
  random: () => number = Math.random,
): CnnModel {
  return {
    pooling,
    filters1: Array.from({ length: FILTER_COUNT }, () =>
      Array.from(
        { length: FILTER_SIZE * FILTER_SIZE },
        () => (random() - 0.5) * 0.6,
      ),
    ),
    filterBias1: Array<number>(FILTER_COUNT).fill(0),
    filters2: Array.from({ length: FILTER_COUNT }, () =>
      Array.from({ length: FILTER_COUNT }, () =>
        Array.from(
          { length: FILTER_SIZE * FILTER_SIZE },
          () => (random() - 0.5) * 0.4,
        ),
      ),
    ),
    filterBias2: Array<number>(FILTER_COUNT).fill(0),
    denseHiddenWeights: Array.from({ length: HIDDEN_COUNT }, () =>
      Array.from({ length: DENSE_INPUTS }, () => (random() - 0.5) * 0.2),
    ),
    denseHiddenBias: Array<number>(HIDDEN_COUNT).fill(0),
    denseOutputWeights: Array.from({ length: 10 }, () =>
      Array.from({ length: HIDDEN_COUNT }, () => (random() - 0.5) * 0.3),
    ),
    denseOutputBias: Array<number>(10).fill(0),
  };
}

function matrix(rows: number, columns: number): number[][] {
  return Array.from({ length: rows }, () => Array<number>(columns).fill(0));
}

function zeroMaps(channels: number, size: number): number[][][] {
  return Array.from({ length: channels }, () => matrix(size, size));
}

function convolve(
  inputs: number[][][],
  filters: number[][][],
  biases: number[],
): number[][][] {
  const inputSize = inputs[0].length,
    outputSize = inputSize - FILTER_SIZE + 1,
    outputs = Array.from({ length: filters.length }, () =>
      matrix(outputSize, outputSize),
    );
  for (let out = 0; out < filters.length; out++)
    for (let y = 0; y < outputSize; y++)
      for (let x = 0; x < outputSize; x++) {
        let value = biases[out];
        for (let channel = 0; channel < inputs.length; channel++)
          for (let fy = 0; fy < FILTER_SIZE; fy++)
            for (let fx = 0; fx < FILTER_SIZE; fx++)
              value +=
                filters[out][channel][fy * FILTER_SIZE + fx] *
                inputs[channel][y + fy][x + fx];
        outputs[out][y][x] = value;
      }
  return outputs;
}

function applyRelu(maps: number[][][]): number[][][] {
  return maps.map((map) =>
    map.map((row) => row.map((value) => Math.max(0, value))),
  );
}

function pool(
  maps: number[][][],
  mode: CnnPooling,
): { values: number[][][]; indices: number[][][] } {
  const inputSize = maps[0].length,
    outputSize = Math.floor(inputSize / 2),
    values = zeroMaps(maps.length, outputSize),
    indices = zeroMaps(maps.length, outputSize);
  for (let channel = 0; channel < maps.length; channel++)
    for (let y = 0; y < outputSize; y++)
      for (let x = 0; x < outputSize; x++) {
        let sum = 0,
          maximum = -Infinity,
          maxIndex = -1;
        for (let dy = 0; dy < 2; dy++)
          for (let dx = 0; dx < 2; dx++) {
            const row = y * 2 + dy,
              column = x * 2 + dx,
              value = maps[channel][row][column];
            sum += value;
            if (value > maximum) {
              maximum = value;
              maxIndex = row * inputSize + column;
            }
          }
        values[channel][y][x] = mode === "max" ? maximum : sum / 4;
        indices[channel][y][x] = mode === "max" ? maxIndex : -1;
      }
  return { values, indices };
}

function dense(
  inputs: number[],
  weights: number[][],
  biases: number[],
): number[] {
  return weights.map((row, i) =>
    row.reduce((sum, weight, j) => sum + weight * inputs[j], biases[i]),
  );
}

export function forwardCnn(model: CnnModel, image: number[]): CnnForward {
  if (image.length !== IMAGE_SIZE * IMAGE_SIZE)
    throw new Error("La CNN richiede immagini MNIST 28×28.");
  const input = [
      Array.from({ length: IMAGE_SIZE }, (_, y) =>
        image
          .slice(y * IMAGE_SIZE, (y + 1) * IMAGE_SIZE)
          .map((value) => value / 255),
      ),
    ],
    convolution1 = convolve(
      input,
      model.filters1.map((filter) => [filter]),
      model.filterBias1,
    ),
    relu1 = applyRelu(convolution1),
    pool1 = pool(relu1, model.pooling),
    convolution2 = convolve(pool1.values, model.filters2, model.filterBias2),
    relu2 = applyRelu(convolution2),
    pool2 = pool(relu2, model.pooling),
    flattened = pool2.values.flatMap((feature) => feature.flat()),
    denseHiddenPreActivation = dense(
      flattened,
      model.denseHiddenWeights,
      model.denseHiddenBias,
    ),
    denseHidden = denseHiddenPreActivation.map((value) => Math.max(0, value)),
    logits = dense(
      denseHidden,
      model.denseOutputWeights,
      model.denseOutputBias,
    ),
    maxLogit = Math.max(...logits),
    exponentials = logits.map((value) => Math.exp(value - maxLogit)),
    total = exponentials.reduce((sum, value) => sum + value, 0);
  return {
    convolution1,
    relu1,
    pooled1: pool1.values,
    poolIndices1: pool1.indices,
    convolution2,
    relu2,
    pooled2: pool2.values,
    poolIndices2: pool2.indices,
    flattened,
    denseHiddenPreActivation,
    denseHidden,
    logits,
    probabilities: exponentials.map((value) => value / total),
  };
}

export function traceCnnPixel(
  model: CnnModel,
  image: number[],
  forward: CnnForward,
  layer: CnnLayer,
  filterIndex: number,
  row: number,
  column: number,
  stage: CnnPixelStage,
): CnnPixelTrace {
  if (image.length !== IMAGE_SIZE * IMAGE_SIZE)
    throw new Error("La CNN richiede immagini MNIST 28×28.");
  if (filterIndex < 0 || filterIndex >= FILTER_COUNT)
    throw new RangeError("Indice del filtro CNN non valido.");
  const convolution = layer === 1 ? forward.convolution1 : forward.convolution2,
    relu = layer === 1 ? forward.relu1 : forward.relu2,
    pooled = layer === 1 ? forward.pooled1 : forward.pooled2,
    indices = layer === 1 ? forward.poolIndices1 : forward.poolIndices2,
    featureSize = convolution[0].length,
    poolSize = pooled[0].length,
    mapSize = stage === "pool" ? poolSize : featureSize;
  if (
    !Number.isInteger(row) ||
    !Number.isInteger(column) ||
    row < 0 ||
    column < 0 ||
    row >= mapSize ||
    column >= mapSize
  )
    throw new RangeError("Coordinate del pixel CNN fuori dalla mappa.");
  const requestedPoolRow = stage === "pool" ? row : Math.floor(row / 2),
    requestedPoolColumn = stage === "pool" ? column : Math.floor(column / 2),
    hasPoolCell = requestedPoolRow < poolSize && requestedPoolColumn < poolSize,
    poolRow = hasPoolCell ? requestedPoolRow : null,
    poolColumn = hasPoolCell ? requestedPoolColumn : null,
    poolWinnerIndex =
      hasPoolCell && model.pooling === "max"
        ? indices[filterIndex][requestedPoolRow][requestedPoolColumn]
        : -1,
    poolWinner =
      poolWinnerIndex >= 0
        ? {
            row: Math.floor(poolWinnerIndex / featureSize),
            column: poolWinnerIndex % featureSize,
          }
        : null,
    convolutionRow = stage === "pool" ? (poolWinner?.row ?? row * 2) : row,
    convolutionColumn =
      stage === "pool" ? (poolWinner?.column ?? column * 2) : column,
    poolCandidates = hasPoolCell
      ? Array.from({ length: 2 }, (_, dy) =>
          Array.from({ length: 2 }, (_, dx) => {
            const candidateRow = requestedPoolRow * 2 + dy,
              candidateColumn = requestedPoolColumn * 2 + dx;
            return {
              row: candidateRow,
              column: candidateColumn,
              convolution:
                convolution[filterIndex][candidateRow][candidateColumn],
              relu: relu[filterIndex][candidateRow][candidateColumn],
            };
          }),
        )
      : [],
    poolValue = hasPoolCell
      ? pooled[filterIndex][requestedPoolRow][requestedPoolColumn]
      : null,
    inputMaps =
      layer === 1
        ? [
            Array.from({ length: IMAGE_SIZE }, (_, y) =>
              image
                .slice(y * IMAGE_SIZE, (y + 1) * IMAGE_SIZE)
                .map((value) => value / 255),
            ),
          ]
        : forward.pooled1,
    filterBanks =
      layer === 1 ? [model.filters1[filterIndex]] : model.filters2[filterIndex],
    inputPatches = inputMaps.map((map) =>
      Array.from({ length: FILTER_SIZE }, (_, fy) =>
        map
          .slice(convolutionRow + fy, convolutionRow + fy + 1)[0]
          .slice(convolutionColumn, convolutionColumn + FILTER_SIZE),
      ),
    ),
    filters = filterBanks.map((bank) =>
      Array.from({ length: FILTER_SIZE }, (_, y) =>
        bank.slice(y * FILTER_SIZE, (y + 1) * FILTER_SIZE),
      ),
    ),
    products = inputPatches.map((patch, channel) =>
      patch.map((patchRow, y) =>
        patchRow.map((value, x) => value * filters[channel][y][x]),
      ),
    ),
    bias =
      layer === 1
        ? model.filterBias1[filterIndex]
        : model.filterBias2[filterIndex];
  let productSum = 0,
    convolutionValue = bias;
  for (const channelProducts of products)
    for (const productRow of channelProducts)
      for (const product of productRow) {
        productSum += product;
        convolutionValue += product;
      }
  return {
    layer,
    stage,
    filterIndex,
    row,
    column,
    convolutionRow,
    convolutionColumn,
    inputPatches,
    filters,
    products,
    productSum,
    bias,
    convolutionValue,
    reluValue: Math.max(0, convolutionValue),
    poolRow,
    poolColumn,
    poolCandidates,
    poolValue,
    poolWinner,
  };
}

export function traceDenseHidden(
  model: CnnModel,
  forward: CnnForward,
  index: number,
): DenseTrace {
  if (!Number.isInteger(index) || index < 0 || index >= HIDDEN_COUNT)
    throw new RangeError("Indice del neurone denso non valido.");
  const weights = model.denseHiddenWeights[index],
    products = forward.flattened.map((value, i) => value * weights[i]),
    productSum = products.reduce((sum, value) => sum + value, 0),
    bias = model.denseHiddenBias[index],
    value = bias + productSum;
  return {
    index,
    inputs: forward.flattened,
    weights,
    products,
    productSum,
    bias,
    value,
    activation: Math.max(0, value),
  };
}

export function traceDenseOutput(
  model: CnnModel,
  forward: CnnForward,
  index: number,
): DenseTrace {
  if (!Number.isInteger(index) || index < 0 || index >= 10)
    throw new RangeError("Indice della classe densa non valido.");
  const weights = model.denseOutputWeights[index],
    products = forward.denseHidden.map((value, i) => value * weights[i]),
    productSum = products.reduce((sum, value) => sum + value, 0),
    bias = model.denseOutputBias[index];
  return {
    index,
    inputs: forward.denseHidden,
    weights,
    products,
    productSum,
    bias,
    value: bias + productSum,
  };
}

function convolutionBackward(
  inputs: number[][][],
  filters: number[][][],
  outputGradient: number[][][],
): {
  inputGradient: number[][][];
  filterGradient: number[][][];
  biasGradient: number[];
} {
  const inputSize = inputs[0].length,
    outputSize = outputGradient[0].length,
    inputGradient = zeroMaps(inputs.length, inputSize),
    filterGradient = filters.map((bank) =>
      bank.map(() => Array<number>(FILTER_SIZE * FILTER_SIZE).fill(0)),
    ),
    biasGradient = Array<number>(filters.length).fill(0);
  for (let out = 0; out < filters.length; out++)
    for (let y = 0; y < outputSize; y++)
      for (let x = 0; x < outputSize; x++) {
        const upstream = outputGradient[out][y][x];
        biasGradient[out] += upstream;
        for (let channel = 0; channel < inputs.length; channel++)
          for (let fy = 0; fy < FILTER_SIZE; fy++)
            for (let fx = 0; fx < FILTER_SIZE; fx++) {
              const weightIndex = fy * FILTER_SIZE + fx,
                weight = filters[out][channel][weightIndex];
              filterGradient[out][channel][weightIndex] +=
                upstream * inputs[channel][y + fy][x + fx];
              inputGradient[channel][y + fy][x + fx] += upstream * weight;
            }
      }
  return { inputGradient, filterGradient, biasGradient };
}

function poolBackward(
  pooledGradient: number[][][],
  indices: number[][][],
  inputSize: number,
  mode: CnnPooling,
): number[][][] {
  const result = zeroMaps(pooledGradient.length, inputSize);
  for (let channel = 0; channel < pooledGradient.length; channel++)
    for (let y = 0; y < pooledGradient[channel].length; y++)
      for (let x = 0; x < pooledGradient[channel][y].length; x++) {
        const gradient = pooledGradient[channel][y][x];
        if (mode === "max") {
          const index = indices[channel][y][x];
          result[channel][Math.floor(index / inputSize)][index % inputSize] +=
            gradient;
        } else
          for (let dy = 0; dy < 2; dy++)
            for (let dx = 0; dx < 2; dx++)
              result[channel][y * 2 + dy][x * 2 + dx] += gradient / 4;
      }
  return result;
}

function reluBackward(
  preActivation: number[][][],
  gradient: number[][][],
): number[][][] {
  return gradient.map((map, channel) =>
    map.map((row, y) =>
      row.map((value, x) => (preActivation[channel][y][x] > 0 ? value : 0)),
    ),
  );
}

function addFilters(target: number[][][], source: number[][][]): void {
  for (let out = 0; out < target.length; out++)
    for (let channel = 0; channel < target[out].length; channel++)
      for (let weight = 0; weight < target[out][channel].length; weight++)
        target[out][channel][weight] += source[out][channel][weight];
}

export function trainCnnBatch(
  model: CnnModel,
  examples: MnistDigit[],
  learningRate: number,
): { loss: number; accuracy: number } {
  if (!examples.length) throw new Error("Il mini-batch CNN è vuoto.");
  const filterGradient1 = model.filters1.map((filter) => [filter.map(() => 0)]),
    biasGradient1 = Array<number>(FILTER_COUNT).fill(0),
    filterGradient2 = model.filters2.map((bank) =>
      bank.map((filter) => filter.map(() => 0)),
    ),
    biasGradient2 = Array<number>(FILTER_COUNT).fill(0),
    hiddenWeightGradient = model.denseHiddenWeights.map((row) =>
      row.map(() => 0),
    ),
    hiddenBiasGradient = Array<number>(HIDDEN_COUNT).fill(0),
    outputWeightGradient = model.denseOutputWeights.map((row) =>
      row.map(() => 0),
    ),
    outputBiasGradient = Array<number>(10).fill(0);
  let loss = 0,
    correct = 0;
  for (const example of examples) {
    const pass = forwardCnn(model, example.image),
      outputGradient = pass.probabilities.map(
        (probability, index) => probability - Number(index === example.label),
      );
    if (
      pass.probabilities.indexOf(Math.max(...pass.probabilities)) ===
      example.label
    )
      correct++;
    loss -= Math.log(Math.max(pass.probabilities[example.label], 1e-12));
    const hiddenGradient = Array<number>(HIDDEN_COUNT).fill(0);
    for (let output = 0; output < 10; output++) {
      outputBiasGradient[output] += outputGradient[output];
      for (let hidden = 0; hidden < HIDDEN_COUNT; hidden++) {
        outputWeightGradient[output][hidden] +=
          outputGradient[output] * pass.denseHidden[hidden];
        hiddenGradient[hidden] +=
          outputGradient[output] * model.denseOutputWeights[output][hidden];
      }
    }
    const hiddenPreGradient = hiddenGradient.map((value, hidden) =>
        pass.denseHiddenPreActivation[hidden] > 0 ? value : 0,
      ),
      flattenedGradient = Array<number>(DENSE_INPUTS).fill(0);
    for (let hidden = 0; hidden < HIDDEN_COUNT; hidden++) {
      hiddenBiasGradient[hidden] += hiddenPreGradient[hidden];
      for (let input = 0; input < DENSE_INPUTS; input++) {
        hiddenWeightGradient[hidden][input] +=
          hiddenPreGradient[hidden] * pass.flattened[input];
        flattenedGradient[input] +=
          hiddenPreGradient[hidden] * model.denseHiddenWeights[hidden][input];
      }
    }
    const pooled2Gradient = Array.from({ length: FILTER_COUNT }, (_, channel) =>
        Array.from({ length: POOL2_SIZE }, (_, y) =>
          flattenedGradient.slice(
            channel * POOL2_SIZE * POOL2_SIZE + y * POOL2_SIZE,
            channel * POOL2_SIZE * POOL2_SIZE + (y + 1) * POOL2_SIZE,
          ),
        ),
      ),
      convolution2Gradient = reluBackward(
        pass.convolution2,
        poolBackward(
          pooled2Gradient,
          pass.poolIndices2,
          CONV2_SIZE,
          model.pooling,
        ),
      ),
      convolution2Back = convolutionBackward(
        pass.pooled1,
        model.filters2,
        convolution2Gradient,
      ),
      convolution1Gradient = reluBackward(
        pass.convolution1,
        poolBackward(
          convolution2Back.inputGradient,
          pass.poolIndices1,
          CONV1_SIZE,
          model.pooling,
        ),
      ),
      input = [
        Array.from({ length: IMAGE_SIZE }, (_, y) =>
          example.image
            .slice(y * IMAGE_SIZE, (y + 1) * IMAGE_SIZE)
            .map((value) => value / 255),
        ),
      ],
      convolution1Back = convolutionBackward(
        input,
        model.filters1.map((filter) => [filter]),
        convolution1Gradient,
      );
    addFilters(filterGradient1, convolution1Back.filterGradient);
    addFilters(filterGradient2, convolution2Back.filterGradient);
    for (let filter = 0; filter < FILTER_COUNT; filter++) {
      biasGradient1[filter] += convolution1Back.biasGradient[filter];
      biasGradient2[filter] += convolution2Back.biasGradient[filter];
    }
  }
  const scale = learningRate / examples.length;
  for (let filter = 0; filter < FILTER_COUNT; filter++) {
    for (let weight = 0; weight < FILTER_SIZE * FILTER_SIZE; weight++)
      model.filters1[filter][weight] -=
        scale * filterGradient1[filter][0][weight];
    model.filterBias1[filter] -= scale * biasGradient1[filter];
    for (let channel = 0; channel < FILTER_COUNT; channel++)
      for (let weight = 0; weight < FILTER_SIZE * FILTER_SIZE; weight++)
        model.filters2[filter][channel][weight] -=
          scale * filterGradient2[filter][channel][weight];
    model.filterBias2[filter] -= scale * biasGradient2[filter];
  }
  for (let hidden = 0; hidden < HIDDEN_COUNT; hidden++) {
    for (let input = 0; input < DENSE_INPUTS; input++)
      model.denseHiddenWeights[hidden][input] -=
        scale * hiddenWeightGradient[hidden][input];
    model.denseHiddenBias[hidden] -= scale * hiddenBiasGradient[hidden];
  }
  for (let output = 0; output < 10; output++) {
    for (let hidden = 0; hidden < HIDDEN_COUNT; hidden++)
      model.denseOutputWeights[output][hidden] -=
        scale * outputWeightGradient[output][hidden];
    model.denseOutputBias[output] -= scale * outputBiasGradient[output];
  }
  return { loss: loss / examples.length, accuracy: correct / examples.length };
}
