import { createTreeAlgorithms } from "./trees.js";
import { createNeuralAlgorithms } from "./neural.js";
import { createLearningInspector } from "./learningInspection.js";
import {
  rawRegressionFeatures,
  regressionVector,
  shuffleCopy,
  vecDist,
  softmax,
  classCounts,
  initCentroids,
  treeClassify,
  buildRegTree,
  treeValue,
  createNaiveStats,
  gaussianScore,
  gaussianPdf,
} from "./math.js";

/**
 * One isolated simulation. Numerical code has no browser or React dependencies.
 * Configuration is copied at creation; changing settings creates a fresh engine.
 * step() learns parameters; inspection can initialize an unfitted model lazily.
 * snapshot() copies chart histories; callers treat model references as read-only.
 */
export function createEngine(dataset, settings) {
  const config = structuredClone(settings),
    data = dataset.data,
    classes = dataset.classes;
  const customRegressionData = Boolean(dataset.imported);
  let losses = [],
    trainAccuracyHistory = [],
    testAccuracyHistory = [];
  /** Discard the current fit and split. The next training step initializes a new experiment. */
  function reset() {
    engine = {
      id: config.method,
      model: null,
      previousModel: null,
      train: [],
      test: [],
      metrics: {},
      iteration: 0,
      fitted: false,
      gradNorm: 0,
      lastLoss: NaN,
    };
  }
  const ENGINE_STATIC = new Set(["knn", "tree", "forest", "naive", "dbscan"]);
  const ENGINE_CLUSTER = new Set(["kmeans", "dbscan", "gmm"]);
  const ENGINE_REGRESSION = new Set(["linear", "ridge", "lasso", "elastic"]);
  let engine = {
    id: null,
    model: null,
    previousModel: null,
    train: [],
    test: [],
    metrics: {},
    iteration: 0,
    fitted: false,
    gradNorm: 0,
    lastLoss: NaN,
  };

  const { buildClassTree, impurity } = createTreeAlgorithms(classes);
  const {
    makeMLP,
    mlpForward,
    flattenMLP,
    unflattenMLP,
    mlpGrad,
    applyOptimizer,
  } = createNeuralAlgorithms({
    pNum,
    isRegressionTask,
    getEngine: () => engine,
  });
  /** Use the copied configuration as the single source of truth for the selected algorithm. */
  function methodId() {
    return config.method;
  }
  /** Clustering consumes feature vectors; labels are retained only for evaluation. */
  function isClustering() {
    return ENGINE_CLUSTER.has(methodId());
  }
  /** Boosting and neural models support both tasks, so inspect their task/output mode. */
  function isRegressionTask(id = methodId()) {
    return (
      ENGINE_REGRESSION.has(id) ||
      (id === "boost" && engine.model?.regression) ||
      ((id === "mlp" ||
        [
          "sgd",
          "momentum",
          "nesterov",
          "adagrad",
          "rmsprop",
          "adam",
          "adamw",
        ].includes(id)) &&
        classes === 1 &&
        config.dataset !== "oneclass")
    );
  }
  /** Reject unsupported supervised problems before any model parameters change. */
  function methodCompatibilityIssue(id = methodId()) {
    if (id === "logistic" && classes !== 2)
      return "La regressione logistica Ã¨ binaria; per piÃ¹ di due classi usa Softmax.";
    const regressionDataset = config.dataset.startsWith("regression");
    if (
      classes < 2 &&
      !regressionDataset &&
      !ENGINE_CLUSTER.has(id) &&
      !ENGINE_REGRESSION.has(id)
    )
      return "Un solo target non definisce un problema di classificazione supervisionata. Passa a Clustering per esplorare questi dati.";
    return "";
  }
  /** Synthetic curved targets use polynomial features; imported data defaults to a linear basis. */
  function regressionBasisDegree() {
    const choice = param("featureDegree", "auto");
    if (choice !== "auto") return Math.max(1, Math.min(5, Number(choice) || 1));
    if (customRegressionData) return 1;
    const dataset = config.dataset;
    return dataset === "regression_quadratic" ||
      dataset === "regression_interaction"
      ? 2
      : dataset === "regression_sine"
        ? 5
        : 1;
  }

  /** Fit feature normalization on training rows only, preventing validation leakage. */

  function makeRegressionModel(rows) {
    const degree = regressionBasisDegree(),
      features = rows.map((row) => rawRegressionFeatures(row.x, degree)),
      means = features[0].map(
        (_, i) =>
          features.reduce((sum, row) => sum + row[i], 0) / (rows.length || 1),
      ),
      scales = features[0].map((_, i) => {
        const variance =
          features.reduce((sum, row) => sum + (row[i] - means[i]) ** 2, 0) /
          (rows.length || 1);
        return Math.sqrt(variance) || 1;
      });
    return {
      kind: "linear",
      degree,
      means,
      scales,
      w: Array(means.length).fill(0),
      b: 0,
    };
  }

  /** For a one-dimensional prediction curve, hold other features at their training means. */

  function regressionPlotInput(x, dimensions) {
    const values = Array(dimensions).fill(0);
    values[0] = x;
    for (let i = 1; i < dimensions; i++)
      values[i] = engine.train.length
        ? engine.train.reduce((sum, row) => sum + (row.x[i] || 0), 0) /
          engine.train.length
        : 0;
    return values;
  }
  /** Return an explicitly supplied hyperparameter, otherwise use the algorithm default. */
  function param(key, fallback) {
    const value = config.parameters[key];
    return value === undefined || value === "" ? fallback : value;
  }
  /** Clamp numerical controls at the engine boundary so model code receives a safe range. */
  function pNum(key, fallback, min = -Infinity, max = Infinity) {
    return Math.max(min, Math.min(max, param(key, fallback)));
  }
  /** HTML selects provide strings; normalize them before using boolean hyperparameters. */
  function pBool(key, fallback = false) {
    const v = param(key, String(fallback));
    return v === "true" || v === true;
  }

  /** Shuffle once and create disjoint training/validation sets before constructing the selected model. */

  function engineInit() {
    const id = methodId(),
      rows = shuffleCopy(data),
      cut = Math.max(
        1,
        Math.min(
          rows.length - 1,
          Math.floor(rows.length * (Number(config.split) || 0.8)),
        ),
      );
    engine = {
      id,
      model: null,
      previousModel: null,
      train: rows.slice(0, cut),
      test: rows.slice(cut),
      metrics: {},
      iteration: 0,
      fitted: false,
      gradNorm: 0,
      lastLoss: NaN,
    };
    if (rows.length < 3) {
      engine.train = rows;
      engine.test = rows;
    }
    const d = data[0]?.x.length || 2,
      k = Math.max(1, classes),
      outputMode = config.nnOutput || "auto",
      mlpOutputs =
        classes === 1 ? 1 : outputMode === "binary" && k === 2 ? 1 : k;
    if (ENGINE_REGRESSION.has(id))
      engine.model = makeRegressionModel(engine.train);
    else if (id === "logistic" || id === "softmax" || id === "perceptron")
      engine.model = {
        kind: id,
        w: Array.from({ length: k }, () => Array(d).fill(0)),
        b: Array(k).fill(0),
      };
    else if (id === "knn") engine.model = { kind: "knn", rows: engine.train };
    else if (id === "svmrbf" || id === "svmpoly") {
      const n = engine.train.length;
      engine.model = {
        kind: "svm",
        kernel: id === "svmrbf" ? "rbf" : "poly",
        alpha: Array.from({ length: k }, () => Array(n).fill(0)),
        bias: Array(k).fill(0),
        labels: Array.from({ length: k }, (_, c) =>
          engine.train.map((r) => (r.y === c ? 1 : -1)),
        ),
        K: Array.from({ length: n }, (_, i) =>
          Array.from({ length: n }, (_, j) =>
            svmKernel(engine.train[i].x, engine.train[j].x, id),
          ),
        ),
        classes: k,
      };
    } else if (id === "tree") engine.model = { kind: "tree", root: null };
    else if (id === "forest") engine.model = { kind: "forest", trees: [] };
    else if (id === "boost")
      engine.model = {
        kind: "boost",
        trees: Array.from({ length: k }, () => []),
        base: Array(k).fill(0),
        regression: classes === 1 && config.dataset !== "oneclass",
      };
    else if (id === "naive") engine.model = { kind: "naive", stats: [] };
    else if (
      id === "mlp" ||
      [
        "sgd",
        "momentum",
        "nesterov",
        "adagrad",
        "rmsprop",
        "adam",
        "adamw",
      ].includes(id)
    )
      engine.model = makeMLP(d, ENGINE_REGRESSION.has(id) ? 1 : mlpOutputs, id);
    else if (id === "kmeans") {
      const k = Math.min(
        Math.round(pNum("clusters", 3, 1, 20)),
        engine.train.length,
      );
      const init =
        param("init", "kmeans++") === "random"
          ? shuffleCopy(engine.train)
              .slice(0, k)
              .map((r) => r.x.slice())
          : initCentroids(engine.train, k);
      engine.model = { kind: "kmeans", centroids: init, labels: [] };
    } else if (id === "dbscan") engine.model = { kind: "dbscan", labels: [] };
    else if (id === "gmm")
      engine.model = makeGMM(engine.train, pNum("components", 3, 1, 12));
    else engine.model = { kind: "linear", w: Array(d).fill(0), b: 0 };
  }
  /** Use accuracy for classification, R-squared for regression, and label purity for clustering. */
  function trendScore(rows) {
    if (!rows.length) return NaN;
    if (isRegressionTask(engine.id)) {
      const mean = rows.reduce((sum, row) => sum + row.y, 0) / rows.length,
        total = rows.reduce((sum, row) => sum + (row.y - mean) ** 2, 0),
        residual = rows.reduce(
          (sum, row) => sum + (enginePredict(row.x) - row.y) ** 2,
          0,
        );
      return total > 0
        ? 100 * (1 - residual / total)
        : residual === 0
          ? 100
          : 0;
    }
    if (isClustering(engine.id)) {
      if (new Set(rows.map((row) => row.y)).size < 2) return NaN;
      const groups = new Map();
      for (const row of rows) {
        const cluster = enginePredict(row.x),
          counts = groups.get(cluster) || new Map();
        counts.set(row.y, (counts.get(row.y) || 0) + 1);
        groups.set(cluster, counts);
      }
      return (
        (100 *
          [...groups.values()].reduce(
            (sum, counts) => sum + Math.max(...counts.values()),
            0,
          )) /
        rows.length
      );
    }
    return (
      (100 * rows.filter((row) => enginePredict(row.x) === row.y).length) /
      rows.length
    );
  }
  /** RBF measures local similarity; the polynomial kernel expands dot products implicitly. */
  function svmKernel(a, b, id) {
    if (id === "svmrbf") {
      const g = pNum("gamma", 0.5, 1e-6, 100);
      let s = 0;
      for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
      return Math.exp(-g * s);
    }
    const g = pNum("gamma", 0.5, 1e-6, 100),
      c = pNum("coef0", 1, -100, 100),
      deg = Math.round(pNum("degree", 3, 1, 8));
    let dot = 0;
    for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
    return Math.pow(g * dot + c, deg);
  }

  /** Estimate class priors and per-feature distributions from the training split. */

  function fitNaive() {
    const variant = param("variant", "gaussian"),
      alpha = pNum("smoothing", 1, 0, 100);
    engine.model.variant = variant;
    engine.model.stats = createNaiveStats(
      engine.train,
      Math.max(1, classes),
      variant,
      alpha,
    );
  }
  /** Accumulate log-probabilities to avoid underflow when multiplying feature likelihoods. */
  function naiveScore(x, st, variant) {
    if (variant === "gaussian") return gaussianScore(x, st);
    let v = Math.log(st.prior + 1e-12);
    for (let f = 0; f < x.length; f++) {
      if (variant === "bernoulli") {
        const q = Math.max(1e-9, Math.min(1 - 1e-9, st.bernoulli[f])),
          bit = x[f] > 0 ? 1 : 0;
        v += bit * Math.log(q) + (1 - bit) * Math.log(1 - q);
      } else
        v +=
          Math.max(0, x[f] + (st.offset?.[f] || 0)) *
          Math.log(Math.max(st.prob[f], 1e-12));
    }
    return v;
  }

  /** Start mixture means at spread-out centroids and use identity covariance matrices. */

  function makeGMM(rows, k) {
    const d = rows[0]?.x.length || 2,
      means = initCentroids(rows, k);
    return {
      kind: "gmm",
      k,
      d,
      means,
      covs: Array.from({ length: k }, () =>
        Array.from({ length: d }, (_, i) =>
          Array.from({ length: d }, (_, j) => (i === j ? 1 : 0)),
        ),
      ),
      mix: Array(k).fill(1 / k),
      resp: [],
    };
  }

  /** Run one update for the selected family, then publish its loss and validation metrics. */

  function fitOneEpoch() {
    const id = engine.id,
      m = engine.model,
      rows = engine.train,
      d = rows[0]?.x.length || 2;
    // Linear families descend the normalized-feature objective; L1 uses soft thresholding.
    if (ENGINE_REGRESSION.has(id)) {
      const lr = pNum("learningRate", 0.01, 1e-6, 2),
        lambda = id === "linear" ? 0 : pNum("lambda", 0.1, 0, 100),
        ratio =
          id === "lasso"
            ? 1
            : id === "linear" || id === "ridge"
              ? 0
              : pNum("l1Ratio", 0.5, 0, 1),
        batchMode = param("batchSize", "Full batch"),
        batch =
          batchMode === "Full batch"
            ? rows.length
            : batchMode === "Mini-batch"
              ? Math.min(16, rows.length)
              : 1,
        samples =
          batchMode === "Full batch"
            ? rows
            : batchMode === "Mini-batch" || batchMode === "SGD"
              ? shuffleCopy(rows).slice(0, batch)
              : rows,
        features = samples.map((row) => regressionVector(m, row.x)),
        gw = Array(m.w.length).fill(0);
      let gb = 0;
      for (let r = 0; r < samples.length; r++) {
        const e =
          m.w.reduce((sum, value, i) => sum + value * features[r][i], m.b) -
          samples[r].y;
        gb += e;
        for (let i = 0; i < m.w.length; i++) gw[i] += e * features[r][i];
      }
      for (let i = 0; i < m.w.length; i++) {
        gw[i] = gw[i] / samples.length + lambda * (1 - ratio) * m.w[i];
        m.w[i] -= lr * gw[i];
        if (ratio > 0) {
          const cut = lr * lambda * ratio;
          m.w[i] = Math.sign(m.w[i]) * Math.max(0, Math.abs(m.w[i]) - cut);
        }
      }
      m.b -= (lr * gb) / samples.length;
      // Cross-entropy gradients are (predicted probability - one-hot target) times the input.
    } else if (id === "logistic" || id === "softmax") {
      const k = Math.max(1, classes),
        lr = pNum("learningRate", 0.08, 1e-6, 2),
        lambda = pNum("lambda", 0, 0, 100),
        batchMode = param("batchSize", "Full batch"),
        batch =
          batchMode === "Full batch"
            ? rows.length
            : batchMode === "Mini-batch"
              ? Math.min(16, rows.length)
              : 1,
        samples =
          batchMode === "Full batch" ? rows : shuffleCopy(rows).slice(0, batch);
      const gw = m.w.map((r) => r.map(() => 0)),
        gb = Array(k).fill(0);
      for (const r of samples) {
        const prob = softmax(
            m.w.map((v, c) => v.reduce((s, t, i) => s + t * r.x[i], m.b[c])),
          ),
          t = Array.from({ length: k }, (_, c) => (r.y === c ? 1 : 0));
        for (let c = 0; c < k; c++) {
          const e = prob[c] - t[c];
          gb[c] += e;
          for (let i = 0; i < d; i++) gw[c][i] += e * r.x[i];
        }
      }
      for (let c = 0; c < k; c++) {
        for (let i = 0; i < d; i++)
          m.w[c][i] -= lr * (gw[c][i] / samples.length + lambda * m.w[c][i]);
        m.b[c] -= (lr * gb[c]) / samples.length;
      }
    } else if (id === "perceptron") {
      const lr = pNum("learningRate", 0.1, 1e-6, 2),
        order = pBool("shuffle", true) ? shuffleCopy(rows) : rows;
      for (const r of order) {
        const scores = m.w.map((v, c) =>
            v.reduce((s, t, i) => s + t * r.x[i], m.b[c]),
          ),
          guess = scores.indexOf(Math.max(...scores));
        if (guess !== r.y) {
          for (let i = 0; i < d; i++) {
            m.w[r.y][i] += lr * r.x[i];
            m.w[guess][i] -= lr * r.x[i];
          }
          m.b[r.y] += lr;
          m.b[guess] -= lr;
        }
      }
    } else if (id === "svmrbf" || id === "svmpoly") {
      const tol = pNum("tolerance", 0.001, 1e-8, 1),
        C = pNum("C", 1, 1e-4, 1e4),
        svm = m,
        n = rows.length;
      for (let c = 0; c < svm.classes; c++) {
        const a = svm.alpha[c],
          y = svm.labels[c];
        for (let i = 0; i < n; i++) {
          let f = 0;
          for (let j = 0; j < n; j++) f += a[j] * y[j] * svm.K[j][i];
          const kii = Math.max(1e-8, svm.K[i][i]),
            other = f - a[i] * y[i] * kii;
          const next = Math.max(
            0,
            Math.min(C, (1 - y[i] * (other + svm.bias[c])) / kii),
          );
          if (Math.abs(next - a[i]) > tol) a[i] = next;
        }
        const free = [];
        for (let i = 0; i < n; i++)
          if (a[i] > 1e-7 && a[i] < C - 1e-7) {
            let f = 0;
            for (let j = 0; j < n; j++) f += a[j] * y[j] * svm.K[j][i];
            free.push(y[i] - f);
          }
        if (free.length)
          svm.bias[c] = free.reduce((s, v) => s + v, 0) / free.length;
      }
    } else if (id === "tree") {
      if (!m.root) {
        const fs = Array.from({ length: d }, (_, i) => i);
        m.root = buildClassTree(
          rows,
          0,
          Math.round(pNum("maxDepth", 5, 1, 30)),
          Math.round(pNum("minSplit", 2, 2, 1000)),
          Math.round(pNum("minLeaf", 1, 1, 500)),
          fs,
          param("criterion", "gini"),
        );
      }
    } else if (id === "forest") {
      if (!m.trees.length) {
        // Bootstrap each tree independently; out-of-bag rows estimate ensemble generalization.
        const nTrees = Math.round(pNum("nEstimators", 100, 1, 300)),
          depth = Math.round(pNum("maxDepth", 8, 1, 30)),
          featureMode = param("maxFeatures", "sqrt"),
          bootstrap = pBool("bootstrap", true),
          minLeaf = Math.round(pNum("minLeaf", 1, 1, 500)),
          criterion = param("criterion", "gini");
        const all = Array.from({ length: d }, (_, i) => i),
          nf =
            featureMode === "all"
              ? d
              : featureMode === "log2"
                ? Math.ceil(Math.log2(d))
                : Math.ceil(Math.sqrt(d));
        m.oobVotes = Array.from({ length: rows.length }, () =>
          Array(Math.max(1, classes)).fill(0),
        );
        m.oobCounts = Array(rows.length).fill(0);
        for (let t = 0; t < nTrees; t++) {
          const indices = bootstrap
              ? Array.from({ length: rows.length }, () =>
                  Math.floor(Math.random() * rows.length),
                )
              : rows.map((_, i) => i),
            inBag = new Set(indices),
            sample = indices.map((i) => rows[i]);
          const tree = buildClassTree(
            sample,
            0,
            depth,
            2,
            minLeaf,
            all,
            criterion,
            featureMode !== "all",
            nf,
          );
          m.trees.push(tree);
          if (bootstrap) {
            rows.forEach((r, i) => {
              if (!inBag.has(i)) {
                m.oobVotes[i][treeClassify(tree, r.x)]++;
                m.oobCounts[i]++;
              }
            });
          }
        }
        const valid = rows.map((r, i) => i).filter((i) => m.oobCounts[i] > 0);
        m.oobScore = valid.length
          ? valid.filter(
              (i) =>
                m.oobVotes[i].indexOf(Math.max(...m.oobVotes[i])) === rows[i].y,
            ).length / valid.length
          : NaN;
      }
    } else if (id === "boost") {
      const reg = m.regression,
        lr = pNum("learningRate", 0.1, 1e-4, 1),
        depth = Math.round(pNum("maxDepth", 3, 1, 10)),
        fraction = pNum("subsample", 1, 0.1, 1),
        stageRows =
          fraction < 1
            ? shuffleCopy(rows).slice(
                0,
                Math.max(2, Math.ceil(rows.length * fraction)),
              )
            : rows;
      if (m.trees[0].length < Math.round(pNum("nEstimators", 100, 1, 300))) {
        if (reg) {
          if (!m.trees[0].length)
            m.base[0] = rows.reduce((s, r) => s + r.y, 0) / rows.length;
          const residual = stageRows.map((r) => ({
            x: r.x,
            y:
              r.y -
              (m.base[0] +
                m.trees[0].reduce((s, t) => s + lr * treeValue(t, r.x), 0)),
          }));
          m.trees[0].push(
            buildRegTree(
              residual,
              0,
              depth,
              1,
              Array.from({ length: d }, (_, i) => i),
            ),
          );
        } else {
          const k = Math.max(1, classes),
            probs = rows.map((r) => boostProb(m, r.x)),
            counts = classCounts(rows, k);
          if (!m.trees[0].length)
            m.base = counts.map((c) => Math.log((c + 1) / (rows.length + k)));
          for (let c = 0; c < k; c++) {
            const residual = stageRows.map((r) => ({
              x: r.x,
              y: (r.y === c ? 1 : 0) - boostProb(m, r.x)[c],
            }));
            m.trees[c].push(
              buildRegTree(
                residual,
                0,
                depth,
                1,
                Array.from({ length: d }, (_, i) => i),
              ),
            );
          }
        }
      }
    } else if (id === "naive") {
      if (!m.stats.length) fitNaive();
    } else if (
      id === "mlp" ||
      [
        "sgd",
        "momentum",
        "nesterov",
        "adagrad",
        "rmsprop",
        "adam",
        "adamw",
      ].includes(id)
    ) {
      const batchMode = param("batchSize", "Full batch"),
        batch =
          batchMode === "Full batch"
            ? rows.length
            : batchMode === "Mini-batch"
              ? Math.min(32, rows.length)
              : batchMode === "SGD"
                ? 1
                : Math.max(
                    1,
                    Math.min(rows.length, Math.round(Number(batchMode) || 1)),
                  ),
        samples =
          batchMode === "Full batch"
            ? rows
            : (pBool("shuffle", true) ? shuffleCopy(rows) : rows.slice()).slice(
                0,
                batch,
              ),
        theta = flattenMLP(m);
      let grad;
      // Nesterov measures the gradient at a look-ahead point, then restores the current weights.
      if (m.opt === "nesterov") {
        const look = theta.map((v, i) => v + m.state.m[i]);
        unflattenMLP(m, look);
        const lookGrad = mlpGrad(m, samples);
        grad = flattenMLP(m, lookGrad.g);
        unflattenMLP(m, theta);
      } else {
        const currentGrad = mlpGrad(m, samples);
        grad = flattenMLP(m, currentGrad.g);
      }
      unflattenMLP(m, applyOptimizer(theta, grad, m, m.opt));
    } else if (id === "kmeans") {
      // Lloyd iteration: assign to nearest centers, then replace each center by its assigned mean.
      const k = m.centroids.length,
        assign = rows.map((r) =>
          m.centroids
            .map((c) => vecDist(r.x, c) ** 2)
            .indexOf(Math.min(...m.centroids.map((c) => vecDist(r.x, c) ** 2))),
        ),
        sums = Array.from({ length: k }, () => Array(d).fill(0)),
        cnt = Array(k).fill(0);
      rows.forEach((r, i) => {
        cnt[assign[i]]++;
        for (let f = 0; f < d; f++) sums[assign[i]][f] += r.x[f];
      });
      for (let c = 0; c < k; c++) {
        if (cnt[c]) m.centroids[c] = sums[c].map((v) => v / cnt[c]);
        else
          m.centroids[c] =
            rows[Math.floor(Math.random() * rows.length)].x.slice();
      }
      m.labels = assign;
    } else if (id === "dbscan") {
      // DBSCAN expands connected dense neighborhoods; -2 means unseen, -1 means noise.
      if (!m.labels.length) {
        const eps = pNum("eps", 0.5, 1e-4, 100),
          minPts = Math.round(pNum("minPts", 5, 1, 1000)),
          metric = param("metric", "euclidean"),
          labels = Array(rows.length).fill(-2);
        let cluster = 0;
        const neigh = (i) =>
          rows
            .map((r, j) => (vecDist(rows[i].x, r.x, metric) <= eps ? j : -1))
            .filter((j) => j >= 0);
        for (let i = 0; i < rows.length; i++) {
          if (labels[i] !== -2) continue;
          let ns = neigh(i);
          if (ns.length < minPts) {
            labels[i] = -1;
            continue;
          }
          labels[i] = cluster;
          const queue = ns.slice();
          for (let q = 0; q < queue.length; q++) {
            const j = queue[q];
            if (labels[j] === -1) labels[j] = cluster;
            if (labels[j] !== -2) continue;
            labels[j] = cluster;
            const more = neigh(j);
            if (more.length >= minPts)
              for (const z of more) if (!queue.includes(z)) queue.push(z);
          }
          cluster++;
        }
        m.labels = labels;
      }
    } else if (id === "gmm") {
      const k = m.k,
        d = m.d,
        reg = pNum("regCovar", 1e-6, 1e-10, 1),
        mode = param("covariance", "full");
      if (!m.resp.length)
        m.resp = rows.map((r) => {
          const ds = m.means.map((c) => vecDist(r.x, c));
          const a = ds.indexOf(Math.min(...ds));
          return Array.from({ length: k }, (_, j) => (j === a ? 1 : 0));
        });
      // EM alternates posterior responsibilities (E-step) and weighted moments (M-step).
      const resp = rows.map((r) =>
        m.means.map((mu, c) => m.mix[c] * gaussianPdf(r.x, mu, m.covs[c])),
      );
      for (let i = 0; i < resp.length; i++) {
        const s = resp[i].reduce((a, b) => a + b, 0) || 1;
        resp[i] = resp[i].map((v) => v / s);
      }
      m.resp = resp;
      for (let c = 0; c < k; c++) {
        const nk = resp.reduce((s, r) => s + r[c], 0) || 1e-8;
        m.mix[c] = nk / rows.length;
        m.means[c] = Array.from(
          { length: d },
          (_, f) => rows.reduce((s, r, i) => s + resp[i][c] * r.x[f], 0) / nk,
        );
        const cov = Array.from({ length: d }, () => Array(d).fill(0));
        for (let i = 0; i < rows.length; i++)
          for (let f = 0; f < d; f++)
            for (let g = 0; g < d; g++)
              cov[f][g] +=
                (resp[i][c] *
                  (rows[i].x[f] - m.means[c][f]) *
                  (rows[i].x[g] - m.means[c][g])) /
                nk;
        if (mode === "diag" || mode === "spherical") {
          for (let f = 0; f < d; f++)
            for (let g = 0; g < d; g++) if (f !== g) cov[f][g] = 0;
          if (mode === "spherical") {
            const v = cov.reduce((s, r, i) => s + r[i], 0) / d;
            for (let f = 0; f < d; f++) cov[f][f] = v;
          }
        }
        for (let f = 0; f < d; f++) cov[f][f] += reg;
        m.covs[c] = cov;
      }
    }
    // Publish after the update so charts, metrics, and the model describe the same iteration.
    const loss = engineLoss();
    engine.lastLoss = loss;
    losses.push(loss);
    if (losses.length > 500) losses.shift();
    engine.iteration++;
    engine.fitted = true;
    trainAccuracyHistory.push(trendScore(engine.train));
    testAccuracyHistory.push(trendScore(engine.test));
    computeEngineMetrics();
  }
  /** Combine the base logits and all fitted weak learners before softmax normalization. */
  function boostProb(m, x) {
    return softmax(
      m.base.map(
        (v, c) =>
          v +
          m.trees[c].reduce(
            (s, t) => s + pNum("learningRate", 0.1, 1e-4, 1) * treeValue(t, x),
            0,
          ),
      ),
    );
  }
  /** Predict from the fitted family using every supplied feature; never update the model here. */
  function enginePredict(x) {
    const id = engine.id,
      m = engine.model;
    if (!m) return 0;
    if (ENGINE_REGRESSION.has(id)) {
      const features = regressionVector(m, x);
      return m.w.reduce((sum, value, i) => sum + value * features[i], m.b);
    }
    if (id === "logistic" || id === "softmax") {
      const scores = m.w.map((v, c) =>
          v.reduce((s, t, i) => s + t * (x[i] || 0), m.b[c]),
        ),
        pr = softmax(scores);
      return id === "logistic" && classes === 2
        ? pr[1] >= pNum("threshold", 0.5, 0, 1)
          ? 1
          : 0
        : pr.indexOf(Math.max(...pr));
    }
    if (id === "perceptron") {
      const scores = m.w.map((v, c) =>
        v.reduce((s, t, i) => s + t * (x[i] || 0), m.b[c]),
      );
      return scores.indexOf(Math.max(...scores));
    }
    if (id === "knn") {
      const k = Math.round(pNum("k", 5, 1, engine.train.length || data.length)),
        metric = param("metric", "euclidean"),
        weighted = param("weights", "uniform") === "distance",
        votes = Array(Math.max(1, classes)).fill(0);
      engine.train
        .map((r) => ({ y: r.y, d: vecDist(r.x, x, metric) }))
        .sort((a, b) => a.d - b.d)
        .slice(0, k)
        .forEach((r) => (votes[r.y] += weighted ? 1 / (r.d + 1e-8) : 1));
      return votes.indexOf(Math.max(...votes));
    }
    if (id === "svmrbf" || id === "svmpoly") {
      const scores = m.alpha.map(
        (a, c) =>
          m.bias[c] +
          a.reduce(
            (s, v, i) =>
              s + v * m.labels[c][i] * svmKernel(engine.train[i].x, x, id),
            0,
          ),
      );
      return scores.indexOf(Math.max(...scores));
    }
    if (id === "tree") return treeClassify(m.root, x);
    if (id === "forest") {
      const votes = Array(Math.max(1, classes)).fill(0);
      m.trees.forEach((t) => votes[treeClassify(t, x)]++);
      return votes.indexOf(Math.max(...votes));
    }
    if (id === "boost") {
      if (m.regression)
        return (
          m.base[0] +
          m.trees[0].reduce(
            (s, t) => s + pNum("learningRate", 0.1, 1e-4, 1) * treeValue(t, x),
            0,
          )
        );
      const pr = boostProb(m, x);
      return pr.indexOf(Math.max(...pr));
    }
    if (id === "naive") {
      const s = m.stats.map((st) => naiveScore(x, st, m.variant));
      return s.indexOf(Math.max(...s));
    }
    if (
      id === "mlp" ||
      [
        "sgd",
        "momentum",
        "nesterov",
        "adagrad",
        "rmsprop",
        "adam",
        "adamw",
      ].includes(id)
    ) {
      const out = mlpForward(m, x).out;
      if (isRegressionTask(id)) return out[0];
      if (m.k === 1) return out[0] >= 0.5 ? 1 : 0;
      return out.indexOf(Math.max(...out));
    }
    if (id === "kmeans") {
      const ds = m.centroids.map((c) => vecDist(c, x));
      return ds.indexOf(Math.min(...ds));
    }
    if (id === "dbscan") {
      let best = -1,
        bd = Infinity;
      for (let i = 0; i < engine.train.length; i++) {
        const d0 = vecDist(engine.train[i].x, x);
        if (d0 < bd && m.labels[i] >= 0) {
          bd = d0;
          best = m.labels[i];
        }
      }
      return bd <= pNum("eps", 0.5, 1e-4, 100) ? best : -1;
    }
    if (id === "gmm") {
      const scores = m.means.map(
        (mu, c) => m.mix[c] * gaussianPdf(x, mu, m.covs[c]),
      );
      return scores.indexOf(Math.max(...scores));
    }
    return 0;
  }

  /** Compute the training objective, including regularization when the selected method uses it. */

  function engineLoss() {
    const id = engine.id,
      m = engine.model,
      rows = engine.train;
    if (!rows.length) return 0;
    if (ENGINE_REGRESSION.has(id)) {
      const mse =
        rows.reduce((s, r) => s + (enginePredict(r.x) - r.y) ** 2, 0) /
        rows.length;
      const lambda = id === "linear" ? 0 : pNum("lambda", 0.1, 0, 100),
        ratio =
          id === "lasso"
            ? 1
            : id === "linear" || id === "ridge"
              ? 0
              : pNum("l1Ratio", 0.5, 0, 1);
      return (
        mse / 2 +
        lambda *
          (ratio * m.w.reduce((s, v) => s + Math.abs(v), 0) +
            (1 - ratio) * 0.5 * m.w.reduce((s, v) => s + v * v, 0))
      );
    }
    if (id === "logistic" || id === "softmax") {
      return (
        rows.reduce((sum, r) => {
          const pr = softmax(
            m.w.map((v, c) => v.reduce((s, t, i) => s + t * r.x[i], m.b[c])),
          );
          return sum - Math.log(Math.max(1e-12, pr[r.y]));
        }, 0) /
          rows.length +
        (pNum("lambda", 0, 0, 100) *
          m.w.reduce((s, v) => s + v.reduce((a, b) => a + b * b, 0), 0)) /
          2
      );
    }
    if (id === "kmeans")
      return (
        rows.reduce(
          (s, r) =>
            s + Math.min(...m.centroids.map((c) => vecDist(r.x, c) ** 2)),
          0,
        ) / rows.length
      );
    if (id === "dbscan")
      return m.labels.filter((v) => v === -1).length / rows.length;
    if (id === "gmm")
      return -rows.reduce(
        (s, r) =>
          s +
          Math.log(
            Math.max(
              1e-300,
              m.means.reduce(
                (v, mu, c) => v + m.mix[c] * gaussianPdf(r.x, mu, m.covs[c]),
                0,
              ),
            ),
          ),
        0,
      );
    if (id === "svmrbf" || id === "svmpoly") {
      let s = 0;
      for (let c = 0; c < m.classes; c++)
        for (let i = 0; i < rows.length; i++) {
          const f =
            m.bias[c] +
            m.alpha[c].reduce(
              (a, v, j) => a + v * m.labels[c][j] * m.K[j][i],
              0,
            );
          s += Math.max(0, 1 - m.labels[c][i] * f);
        }
      return s / rows.length;
    }
    if (
      id === "mlp" ||
      [
        "sgd",
        "momentum",
        "nesterov",
        "adagrad",
        "rmsprop",
        "adam",
        "adamw",
      ].includes(id)
    )
      return mlpGrad(m, rows).loss;
    if (id === "boost" && m.regression)
      return (
        rows.reduce((s, r) => s + (enginePredict(r.x) - r.y) ** 2, 0) /
        rows.length
      );
    if (isClustering())
      return (
        rows.reduce((s, r) => s + (enginePredict(r.x) >= 0 ? 0 : 1), 0) /
        rows.length
      );
    let wrong = rows.filter((r) => enginePredict(r.x) !== r.y).length;
    return wrong / rows.length;
  }
  /** Evaluate held-out rows without adding a training-only regularization penalty. */
  function evaluationLoss(rows) {
    if (!rows.length) return NaN;
    const id = engine.id,
      m = engine.model;
    if (ENGINE_REGRESSION.has(id) || (id === "boost" && m.regression))
      return (
        rows.reduce((s, r) => s + (enginePredict(r.x) - r.y) ** 2, 0) /
        rows.length
      );
    if (id === "logistic" || id === "softmax")
      return (
        rows.reduce((s, r) => {
          const pr = softmax(
            m.w.map((v, c) => v.reduce((a, t, i) => a + t * r.x[i], m.b[c])),
          );
          return s - Math.log(Math.max(1e-12, pr[r.y]));
        }, 0) / rows.length
      );
    if (
      id === "mlp" ||
      [
        "sgd",
        "momentum",
        "nesterov",
        "adagrad",
        "rmsprop",
        "adam",
        "adamw",
      ].includes(id)
    )
      return mlpGrad(m, rows).loss;
    if (id === "boost" && !m.regression)
      return (
        rows.reduce(
          (s, r) => s - Math.log(Math.max(1e-12, boostProb(m, r.x)[r.y])),
          0,
        ) / rows.length
      );
    return rows.filter((r) => enginePredict(r.x) !== r.y).length / rows.length;
  }
  /** Compute task-specific metrics on validation rows; labels never influence clustering fits. */
  function computeEngineMetrics() {
    const id = engine.id,
      rows = engine.test.length ? engine.test : engine.train;
    if (!rows.length) return;
    const metrics = {};
    if (isRegressionTask(id)) {
      const err = rows.map((r) => enginePredict(r.x) - r.y),
        mse = err.reduce((s, v) => s + v * v, 0) / err.length,
        mae = err.reduce((s, v) => s + Math.abs(v), 0) / err.length,
        mean = rows.reduce((s, r) => s + r.y, 0) / rows.length,
        ss = rows.reduce((s, r) => s + (r.y - mean) ** 2, 0),
        r2 = ss ? 1 - err.reduce((s, v) => s + v * v, 0) / ss : 0;
      Object.assign(metrics, {
        MSE: mse,
        RMSE: Math.sqrt(mse),
        MAE: mae,
        "RÂ²": r2,
      });
    } else if (isClustering()) {
      const labels = rows.map((r) => enginePredict(r.x)),
        clusters = [...new Set(labels)],
        noise = labels.filter((v) => v < 0).length;
      metrics["Numero cluster"] = clusters.filter((v) => v >= 0).length;
      metrics["Punti rumore"] = noise;
      metrics["Inertia"] = id === "kmeans" ? engineLoss() : NaN;
      metrics["Cluster sizes"] = [...new Set(labels)]
        .map((c) => `${c}:${labels.filter((v) => v === c).length}`)
        .join(" Â· ");
      metrics["Log-likelihood"] = id === "gmm" ? -engineLoss() : NaN;
      if (clusters.length > 1 && clusters.length < rows.length) {
        const sample = rows.slice(0, Math.min(80, rows.length)),
          labs = sample.map((r) => enginePredict(r.x));
        let sil = 0,
          count = 0;
        for (let i = 0; i < sample.length; i++) {
          const ci = labs[i];
          if (ci < 0) continue;
          const sums = new Map(),
            nums = new Map();
          for (let j = 0; j < sample.length; j++) {
            if (i === j || labs[j] < 0) continue;
            const c = labs[j];
            sums.set(c, (sums.get(c) || 0) + vecDist(sample[i].x, sample[j].x));
            nums.set(c, (nums.get(c) || 0) + 1);
          }
          const a = (sums.get(ci) || 0) / (nums.get(ci) || 1);
          let bmin = Infinity;
          for (const [c, sum] of sums)
            if (c !== ci) bmin = Math.min(bmin, sum / nums.get(c));
          if (Number.isFinite(bmin)) {
            sil += (bmin - a) / Math.max(a, bmin, 1e-12);
            count++;
          }
        }
        metrics.Silhouette = count ? sil / count : NaN;
      }
      if (id === "gmm") {
        const n = rows.length,
          d = rows[0].x.length,
          k = engine.model.k,
          params = k * (d + (d * (d + 1)) / 2) + k - 1;
        metrics.AIC = 2 * params + 2 * engineLoss();
        metrics.BIC = Math.log(n) * params + 2 * engineLoss();
      }
    } else {
      const k = Math.max(1, classes),
        cm = Array.from({ length: k }, () => Array(k).fill(0));
      let correct = 0;
      rows.forEach((r) => {
        const p = Math.max(0, Math.min(k - 1, enginePredict(r.x)));
        cm[r.y][p]++;
        if (p === r.y) correct++;
      });
      const precision = [],
        recall = [],
        f1 = [];
      for (let c = 0; c < k; c++) {
        const tp = cm[c][c],
          fp = cm.reduce((s, row, i) => s + (i !== c ? row[c] : 0), 0),
          fn = cm[c].reduce((s, v, j) => s + (j !== c ? v : 0), 0),
          pr = tp / (tp + fp || 1),
          re = tp / (tp + fn || 1);
        precision.push(pr);
        recall.push(re);
        f1.push((2 * pr * re) / (pr + re || 1));
      }
      metrics.Accuracy = correct / rows.length;
      metrics.Precision = precision.reduce((a, b) => a + b, 0) / k;
      metrics.Recall = recall.reduce((a, b) => a + b, 0) / k;
      metrics.F1 = f1.reduce((a, b) => a + b, 0) / k;
      metrics["Macro-F1"] = metrics.F1;
      metrics["Micro-F1"] = metrics.Accuracy;
      metrics["Confusion matrix"] = cm.map((r) => r.join(" ")).join(" | ");
      if ((id === "svmrbf" || id === "svmpoly") && k === 2) {
        const scores = rows
            .map((r) => {
              const m = engine.model;
              return [0, 1].map(
                (c) =>
                  m.bias[c] +
                  m.alpha[c].reduce(
                    (sum, a, i) =>
                      sum +
                      a *
                        m.labels[c][i] *
                        svmKernel(engine.train[i].x, r.x, id),
                    0,
                  ),
              );
            })
            .map((v) => v[1] - v[0]),
          pos = [],
          neg = [];
        scores.forEach((v, i) => (rows[i].y === 1 ? pos : neg).push(v));
        if (pos.length && neg.length) {
          let wins = 0;
          for (const a of pos)
            for (const b of neg) wins += a > b ? 1 : a === b ? 0.5 : 0;
          metrics["ROC-AUC"] = wins / (pos.length * neg.length);
        }
      }
    }
    metrics["Training loss"] = engine.lastLoss;
    metrics["Validation loss"] = evaluationLoss(engine.test);
    if (id === "boost" && !engine.model.regression)
      metrics["Log-loss"] = evaluationLoss(rows);
    if (id === "forest") metrics["OOB score"] = engine.model.oobScore;
    metrics["Gradient norm"] = engine.gradNorm;
    engine.metrics = metrics;
  }
  /** Initialize lazily, preserve the previous model for explanations, and apply one real update. */
  function trainOne() {
    if (!engine.model || engine.id !== methodId()) engineInit();
    const id = engine.id;
    if (ENGINE_STATIC.has(id) && engine.fitted) return;
    if (
      id === "boost" &&
      engine.model.trees[0].length >=
        Math.round(pNum("nEstimators", 100, 1, 300))
    )
      return;
    if (id === "dbscan" && engine.fitted) return;
    const neuralMethods = [
        "mlp",
        "sgd",
        "momentum",
        "nesterov",
        "adagrad",
        "rmsprop",
        "adam",
        "adamw",
      ],
      trackable =
        isRegressionTask(id) ||
        neuralMethods.includes(id) ||
        (["logistic", "softmax", "perceptron"].includes(id) && classes === 2);
    engine.previousModel = trackable
      ? JSON.parse(JSON.stringify(engine.model))
      : null;
    fitOneEpoch();
  }
  /** Some models fit once; iterative models use epochs, iterations, or ensemble stages. */
  function runTotal() {
    const id = methodId();
    if (ENGINE_STATIC.has(id)) return 1;
    if (id === "svmrbf" || id === "svmpoly")
      return Math.round(pNum("maxIter", 1000, 1, 3000));
    if (id === "boost") return Math.round(pNum("nEstimators", 100, 1, 300));
    if (id === "kmeans" || id === "gmm")
      return Math.round(pNum("maxIter", 100, 1, 500));
    if (id === "dbscan") return 1;
    return Math.round(pNum("epochs", 120, 1, 2000));
  }
  const {
    getAlgorithmLearningTrace,
    getAlgorithmLearningVisualization,
    getNeuralNetworkLearningVisualization,
  } = createLearningInspector({
    config,
    ENGINE_REGRESSION,
    ENGINE_CLUSTER,
    getEngine: () => engine,
    data,
    classes,
    reset,
    methodId,
    isRegressionTask,
    rawRegressionFeatures,
    regressionVector,
    regressionPlotInput,
    param,
    pNum,
    vecDist,
    softmax,
    engineInit,
    svmKernel,
    impurity,
    treeClassify,
    treeValue,
    createNaiveStats,
    naiveScore,
    mlpForward,
    flattenMLP,
    unflattenMLP,
    mlpGrad,
    gaussianPdf,
    boostProb,
    enginePredict,
    engineLoss,
    evaluationLoss,
  });

  /** Copy metric histories so React can render a stable view of the latest completed step. */

  function snapshot() {
    return {
      iteration: engine.iteration,
      fitted: engine.fitted,
      loss: engine.lastLoss,
      losses: losses.slice(),
      trainHistory: trainAccuracyHistory.slice(),
      testHistory: testAccuracyHistory.slice(),
      metrics: { ...engine.metrics },
      model: engine.model,
      train: engine.train,
      test: engine.test,
      regression: isRegressionTask(),
      clustering: isClustering(),
      total: runTotal(),
      issue: methodCompatibilityIssue(),
    };
  }

  /** Evaluate a two-weight loss slice on a copy, so inspection cannot change training. */
  function neuralDiagnostics() {
    if (!engine.model || engine.model.kind !== "mlp") return null;
    const model = engine.model;
    const result = mlpGrad(model, engine.train);
    const values = flattenMLP(model),
      gradients = flattenMLP(model, result.g);
    const names = [];
    model.W1.forEach((row, j) =>
      row.forEach((_, i) => names.push(`W1[${j},${i}]`)),
    );
    model.b1.forEach((_, i) => names.push(`b1[${i}]`));
    model.W2.forEach((row, j) =>
      row.forEach((_, i) => names.push(`W2[${j},${i}]`)),
    );
    model.b2.forEach((_, i) => names.push(`b2[${i}]`));
    const copy = structuredClone(model),
      surface = [],
      resolution = 15;
    for (let y = 0; y < resolution; y++) {
      const row = [];
      for (let x = 0; x < resolution; x++) {
        const point = values.slice();
        point[0] += -1 + (2 * x) / (resolution - 1);
        point[1] += -1 + (2 * y) / (resolution - 1);
        unflattenMLP(copy, point);
        row.push(mlpGrad(copy, engine.train).loss);
      }
      surface.push(row);
    }
    return {
      parameters: values.map((value, i) => ({
        name: names[i],
        value,
        gradient: gradients[i],
      })),
      surface,
      xMin: values[0] - 1,
      xMax: values[0] + 1,
      yMin: values[1] - 1,
      yMax: values[1] + 1,
    };
  }
  return {
    snapshot,
    predict: enginePredict,
    trace: getAlgorithmLearningTrace,
    visualization: getAlgorithmLearningVisualization,
    neuralVisualization: getNeuralNetworkLearningVisualization,
    neuralDiagnostics,
    step() {
      const issue = methodCompatibilityIssue();
      if (issue) throw new Error(issue);
      trainOne();
      return snapshot();
    },
    reset() {
      reset();
      losses = [];
      trainAccuracyHistory = [];
      testAccuracyHistory = [];
    },
  };
}
