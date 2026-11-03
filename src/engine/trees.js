import { classCounts, shuffleCopy } from "./math.js";
/** Recursive classification tree construction, also used by forests. */
export function createTreeAlgorithms(classes) {
  /** Measure label mixture within a node using Gini or entropy. */
  function impurity(rows, criterion = "gini") {
    if (!rows.length) return 0;
    const c = classCounts(rows, Math.max(1, classes)),
      p = c.filter((x) => x > 0).map((x) => x / rows.length);
    return criterion === "gini"
      ? 1 - p.reduce((s, v) => s + v * v, 0)
      : -p.reduce((s, v) => s + v * Math.log2(v), 0);
  }
  /** Scan adjacent feature values and choose the threshold with the greatest weighted impurity reduction. */
  function bestTreeSplit(rows, features, minLeaf, criterion) {
    let best = null,
      base = impurity(rows, criterion);
    for (const f of features) {
      const sorted = rows
        .map((r) => ({ x: r.x[f], y: r.y }))
        .sort((a, b) => a.x - b.x);
      for (let i = minLeaf; i <= sorted.length - minLeaf; i++) {
        if (i === sorted.length || sorted[i - 1].x === sorted[i].x) continue;
        const t = (sorted[i - 1].x + sorted[i].x) / 2;
        const left = rows.filter((r) => r.x[f] <= t),
          right = rows.length - left.length;
        if (left.length < minLeaf || right < minLeaf) continue;
        const score =
          (left.length * impurity(left, criterion) +
            right *
              impurity(
                rows.filter((r) => r.x[f] > t),
                criterion,
              )) /
          rows.length;
        if (!best || score < best.score)
          best = { feature: f, threshold: t, score, gain: base - score };
      }
    }
    return best;
  }
  /** Recursively split until depth, sample-count, or purity constraints require a leaf. */
  function buildClassTree(
    rows,
    depth,
    maxDepth,
    minSplit,
    minLeaf,
    features,
    criterion,
    randomFeatures = false,
    featureCount = null,
  ) {
    const counts = classCounts(rows, Math.max(1, classes)),
      pred = counts.indexOf(Math.max(...counts));
    const node = { leaf: true, pred, counts };
    if (
      depth >= maxDepth ||
      rows.length < minSplit ||
      counts.filter((v) => v > 0).length <= 1
    )
      return node;
    let fs = features.slice();
    if (randomFeatures && fs.length > 1) {
      const n = Math.max(
        1,
        Math.min(fs.length, featureCount || Math.ceil(Math.sqrt(fs.length))),
      );
      fs = shuffleCopy(fs).slice(0, n);
    }
    const split = bestTreeSplit(rows, fs, minLeaf, criterion);
    if (!split || split.gain < 1e-10) return node;
    const left = rows.filter((r) => r.x[split.feature] <= split.threshold),
      right = rows.filter((r) => r.x[split.feature] > split.threshold);
    if (!left.length || !right.length) return node;
    return {
      leaf: false,
      feature: split.feature,
      threshold: split.threshold,
      left: buildClassTree(
        left,
        depth + 1,
        maxDepth,
        minSplit,
        minLeaf,
        features,
        criterion,
        randomFeatures,
      ),
      right: buildClassTree(
        right,
        depth + 1,
        maxDepth,
        minSplit,
        minLeaf,
        features,
        criterion,
        randomFeatures,
      ),
    };
  }
  return { buildClassTree, impurity };
}
