/**
 * Shared vocabulary for the UI and numerical engine.
 *
 * A row always contains a dense feature vector and one numerical target. For
 * classification, y is a zero-based class index; for regression it is the raw
 * continuous response. Clustering keeps y only for post-fit evaluation.
 *
 * These contracts intentionally contain no React or DOM types: the same model
 * can be exercised by Node tests or presented by a browser component.
 */
export type Vector = number[];
/** Matrices are row-major: matrix[output][input] for neural weight matrices. */
export type Matrix = Vector[];
export interface Sample {
  x: Vector;
  y: number;
}
export interface Dataset {
  data: Sample[];
  classes: number;
  imported?: boolean;
  /** Original numeric class labels, before mapping them to zero-based indices. */
  labels?: number[];
}
export type TaskId = "classification" | "regression" | "clustering" | "other";
export type MethodId =
  | "logistic"
  | "softmax"
  | "perceptron"
  | "linear"
  | "ridge"
  | "lasso"
  | "elastic"
  | "knn"
  | "svmrbf"
  | "svmpoly"
  | "tree"
  | "forest"
  | "boost"
  | "naive"
  | "mlp"
  | "sgd"
  | "momentum"
  | "nesterov"
  | "adagrad"
  | "rmsprop"
  | "adam"
  | "adamw"
  | "kmeans"
  | "dbscan"
  | "gmm";
export type DatasetId =
  | "linear"
  | "linear_multiclass"
  | "xor"
  | "circles"
  | "moons"
  | "spiral3"
  | "blobs"
  | "threeblobs"
  | "oneclass"
  | "regression"
  | "regression_quadratic"
  | "regression_sine"
  | "regression_interaction"
  | "multiclass_blobs"
  | "multiclass_spiral";
export type PageId =
  "home" | "data" | "train" | "visual" | "method" | "tutorial";
export type ParameterValue = string | number | boolean;
export type Parameters = Record<string, ParameterValue>;
/** Select controls serialize values as strings; numerical normalization happens at the engine boundary. */
export interface Configuration {
  task: TaskId;
  method: MethodId;
  dataset: DatasetId;
  featureCount: number;
  sampleCount: number | string;
  classCount: number;
  noise: number;
  split: number;
  nnOutput: "auto" | "binary" | "softmax";
  parameters: Parameters;
}
export type ParameterSchema = [
  key: string,
  label: string,
  defaultValue: string,
  inputType: string,
];
export interface MethodInfo {
  status: string;
  model: string;
  cost: string;
  update: string;
  explain: string;
}
export type SymbolDefinition = [symbol: string, meaning: string];

/** Regression coefficients live in the standardized polynomial feature space, not raw x. */
export interface LinearModel {
  kind: "linear";
  degree: number;
  means: Vector;
  scales: Vector;
  w: Vector;
  b: number;
}
/** Separate variants let modelOfKind narrow even one individual classifier kind. */
export type LinearClassifier =
  | { kind: "logistic"; w: Matrix; b: Vector }
  | { kind: "softmax"; w: Matrix; b: Vector }
  | { kind: "perceptron"; w: Matrix; b: Vector };
export interface KnnModel {
  kind: "knn";
  rows: Sample[];
}
/** Each class has one-vs-rest dual coefficients and signed training labels; K caches the kernel matrix. */
export interface SvmModel {
  kind: "svm";
  kernel: "rbf" | "poly";
  alpha: Matrix;
  bias: Vector;
  labels: Matrix;
  K: Matrix;
  classes: number;
}
/** A split node's children and threshold exist together; leaf nodes carry predictions only. */
export type ClassTree =
  | { leaf: true; pred: number; counts: Vector }
  | {
      leaf: false;
      feature: number;
      threshold: number;
      left: ClassTree;
      right: ClassTree;
    };
export type RegressionTree =
  | { leaf: true; value: number }
  | {
      leaf: false;
      feature: number;
      threshold: number;
      left: RegressionTree;
      right: RegressionTree;
    };
export interface TreeModel {
  kind: "tree";
  root: ClassTree | null;
}
export interface ForestModel {
  kind: "forest";
  trees: ClassTree[];
  oobVotes: Matrix;
  oobCounts: Vector;
  oobScore: number;
}
export interface BoostModel {
  kind: "boost";
  trees: RegressionTree[][];
  base: Vector;
  regression: boolean;
}
export interface NaiveStatistics {
  prior: number;
  mean: Vector;
  varr: Vector;
  prob: Vector;
  offset: Vector;
  bernoulli: Vector;
}
export interface NaiveModel {
  kind: "naive";
  stats: NaiveStatistics[];
  variant: string;
}
/** Momentum and adaptive optimizers share buffers in the W1,b1,W2,b2 flattening order. */
export interface OptimizerState {
  m: Vector;
  v: Vector;
  vmax: Vector;
  t: number;
}
export interface NeuralWeights {
  W1: Matrix;
  b1: Vector;
  W2: Matrix;
  b2: Vector;
}
export interface NeuralModel extends NeuralWeights {
  kind: "mlp";
  d: number;
  h: number;
  k: number;
  opt: MethodId;
  state: OptimizerState;
  thetaN: number;
}
export interface KmeansModel {
  kind: "kmeans";
  centroids: Matrix;
  labels: Vector;
}
export interface DbscanModel {
  kind: "dbscan";
  labels: Vector;
}
export interface GmmModel {
  kind: "gmm";
  k: number;
  d: number;
  means: Matrix;
  covs: Matrix[];
  mix: Vector;
  resp: Matrix;
}
/**
 * `kind` is a discriminant, not a display label. Narrow it before accessing
 * family-specific fields, so a neural matrix cannot be mistaken for a linear
 * coefficient vector. Several method IDs intentionally share one model shape.
 */
export type Model =
  | LinearModel
  | LinearClassifier
  | KnnModel
  | SvmModel
  | TreeModel
  | ForestModel
  | BoostModel
  | NaiveModel
  | NeuralModel
  | KmeansModel
  | DbscanModel
  | GmmModel;
export type Metrics = Record<string, number | string | undefined>;
export interface EngineState {
  id: MethodId;
  model: Model | null;
  previousModel: Model | null;
  train: Sample[];
  test: Sample[];
  metrics: Metrics;
  iteration: number;
  fitted: boolean;
  gradNorm: number;
  lastLoss: number;
}
export interface Snapshot {
  iteration: number;
  fitted: boolean;
  loss: number;
  losses: Vector;
  trainHistory: Vector;
  testHistory: Vector;
  metrics: Metrics;
  model: Model | null;
  train: Sample[];
  test: Sample[];
  regression: boolean;
  clustering: boolean;
  total: number;
  issue: string;
}
export interface LearningStep {
  title: string;
  detail: string;
  fullDetail?: string;
}
export interface LearningTrace {
  status: string;
  steps: LearningStep[];
}
export interface ChartView {
  zoom: number;
  panX: number;
  panY: number;
  yaw: number;
  pitch: number;
}
/** Hit targets are in CSS pixels, independent from the canvas's device-pixel resolution. */
export interface ChartPoint {
  index: number;
  x: number;
  y: number;
}
export type NumericParameter = (
  key: string,
  fallback: number,
  min?: number,
  max?: number,
) => number;
export interface ReadParameter {
  (key: string, fallback: string): string;
  (key: string, fallback: number): number;
  (key: string, fallback: boolean): boolean;
}
export interface NeuralForward {
  a: Vector;
  z1: Vector;
  h: Vector;
  z2: Vector;
  out: Vector;
}
export interface NeuralGradient {
  g: NeuralWeights;
  loss: number;
}
export interface NeuralVisualization {
  iteration: number;
  input: Vector;
  inputValues: Vector;
  hiddenValues: Vector;
  outputValues: Vector;
  previousInputValues: Vector | null;
  previousHiddenValues: Vector | null;
  previousOutputValues: Vector | null;
  weightsInputHidden: Matrix;
  weightsHiddenOutput: Matrix;
  previousWeightsInputHidden: Matrix | null;
  previousWeightsHiddenOutput: Matrix | null;
}
export interface NeuralDiagnostics {
  parameters: { name: string; value: number; gradient: number }[];
  surface: Matrix;
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

/** A tutorial can expose predictions, signed margins, or residuals for the same row. */
export interface LearningEntry {
  index: number;
  x: Vector;
  y: number;
  selected?: boolean;
  before?: number | null;
  after?: number;
  predicted?: number;
  prediction?: number;
  beforeDistance?: number | null;
  afterDistance?: number | null;
  change?: number | null;
}
export interface DetailRow {
  label: string;
  value: number | string;
  kind?: string;
}
export interface LearningDetail {
  title: string;
  rows: DetailRow[];
}
export interface Boundary {
  x: number;
  y: number;
  intercept: number;
}
export interface LearningVisualization {
  kind:
    | "regression"
    | "classification"
    | "model-regression"
    | "model-classification"
    | "model-clustering";
  iteration: number;
  algorithm?: MethodId;
  dimensions?: number;
  hasPrevious?: boolean;
  hasBoundary?: boolean;
  boundary?: { before: Boundary | null; after: Boundary | null };
  entries?: LearningEntry[];
  points?: LearningEntry[];
  curve?: { x: number; before: number | null; after: number }[];
  selectedIndex?: number;
  selected?: LearningEntry;
  means?: Vector;
  surface?: Matrix;
  domain?: { xMin: number; xMax: number; yMin: number; yMax: number };
  details?: LearningDetail[];
  path?: DetailRow[];
  centroids?: { index: number; x: number; y: number }[];
  supportVectors?: { index: number; alpha: number }[];
  message?: string;
}

/** Derived from the hook so component props cannot drift from the actual controller API. */
export type Playground = ReturnType<
  typeof import("./hooks/usePlayground.ts").usePlayground
>;
export interface PlaygroundProps {
  playground: Playground;
}
