import type { DatasetId, MethodId, PageId, TaskId } from "../types.ts";

/** Task compatibility and display labels live outside UI components and numerical algorithms. */
export const TASKS: Record<
  TaskId,
  { label: string; methods: MethodId[]; preferred: MethodId }
> = {
  classification: {
    label: "Classificazione",
    methods: [
      "logistic",
      "softmax",
      "perceptron",
      "knn",
      "svmrbf",
      "svmpoly",
      "tree",
      "forest",
      "boost",
      "naive",
      "mlp",
    ],
    preferred: "logistic",
  },
  regression: {
    label: "Regressione",
    methods: ["linear", "ridge", "lasso", "elastic", "boost", "mlp"],
    preferred: "ridge",
  },
  clustering: {
    label: "Clustering",
    methods: ["kmeans", "dbscan", "gmm"],
    preferred: "kmeans",
  },
  other: {
    label: "Ottimizzatori",
    methods: [
      "sgd",
      "momentum",
      "nesterov",
      "adagrad",
      "rmsprop",
      "adam",
      "adamw",
    ],
    preferred: "adam",
  },
};
export const METHOD_LABELS: Record<MethodId, string> = {
  logistic: "Regressione logistica binaria",
  softmax: "Regressione softmax",
  perceptron: "Perceptron",
  linear: "Regressione lineare",
  ridge: "Ridge Regression",
  lasso: "Lasso",
  elastic: "Elastic Net",
  knn: "K-Nearest Neighbors",
  svmrbf: "SVM · kernel RBF",
  svmpoly: "SVM · kernel polinomiale",
  tree: "Decision Tree",
  forest: "Random Forest",
  boost: "Gradient Boosting",
  naive: "Naive Bayes",
  mlp: "MLP / Neural Network",
  sgd: "SGD",
  momentum: "SGD + Momentum",
  nesterov: "Nesterov Momentum",
  adagrad: "AdaGrad",
  rmsprop: "RMSProp",
  adam: "Adam",
  adamw: "AdamW",
  kmeans: "K-means",
  dbscan: "DBSCAN",
  gmm: "Gaussian Mixture Model",
};
export const DATASETS: [DatasetId, string][] = [
  ["linear", "Classificazione binaria lineare"],
  ["linear_multiclass", "Settori lineari · multiclasse"],
  ["xor", "XOR"],
  ["circles", "Cerchi concentrici"],
  ["moons", "Due lune"],
  ["spiral3", "Spirali intrecciate"],
  ["blobs", "Due gruppi gaussiani"],
  ["threeblobs", "Gruppi gaussiani multiclasse"],
  ["oneclass", "Una classe · nuvola singola"],
  ["regression", "Lineare"],
  ["regression_quadratic", "Quadratica"],
  ["regression_sine", "Sinusoidale"],
  ["regression_interaction", "Con interazione"],
];
export const MULTICLASS_DATASETS: DatasetId[] = [
  "linear_multiclass",
  "spiral3",
  "threeblobs",
];
export const NEURAL_METHODS: MethodId[] = [
  "mlp",
  "sgd",
  "momentum",
  "nesterov",
  "adagrad",
  "rmsprop",
  "adam",
  "adamw",
];
export const NAVIGATION: [PageId, string][] = [
  ["home", "Panoramica"],
  ["data", "Dataset"],
  ["train", "Apprendimento"],
  ["visual", "Visualizzazioni"],
  ["method", "Formule e metriche"],
  ["tutorial", "Guida didattica"],
];
