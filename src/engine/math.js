/** Box-Muller sampling and bounded sigmoid used throughout the models. */
export function randn() {
  let u = 0,
    v = 0;
  while (!u) u = Math.random();
  while (!v) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
export function sigmoid(z) {
  return 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, z))));
}

// Pure numerical helpers shared by the model families.
/** Name polynomial and pairwise interaction terms in the same order as their numeric values. */
export function regressionFeatureNames(dimensions, degree) {
  const names = [];
  for (let i = 0; i < dimensions; i++) {
    names.push(`x${i + 1}`);
    for (let power = 2; power <= degree; power++)
      names.push(`x${i + 1}^${power}`);
  }
  if (degree >= 2)
    for (let i = 0; i < dimensions; i++)
      for (let j = i + 1; j < dimensions; j++)
        names.push(`x${i + 1}·x${j + 1}`);
  return names;
}
/** Expand each feature by degree and add pairwise interactions for nonlinear regression. */
export function rawRegressionFeatures(x, degree) {
  const values = [];
  for (let i = 0; i < x.length; i++) {
    values.push(x[i] || 0);
    for (let power = 2; power <= degree; power++)
      values.push((x[i] || 0) ** power);
  }
  if (degree >= 2)
    for (let i = 0; i < x.length; i++)
      for (let j = i + 1; j < x.length; j++)
        values.push((x[i] || 0) * (x[j] || 0));
  return values;
}
/** Apply training-set centering and scaling to the expanded features of a new input. */
export function regressionVector(model, x) {
  const raw = rawRegressionFeatures(x, model.degree);
  return raw.map(
    (value, i) => (value - (model.means[i] || 0)) / (model.scales[i] || 1),
  );
}
/** Fisher-Yates shuffle on a copy; preserve the original dataset ordering for tables. */
export function shuffleCopy(a) {
  const r = a.slice();
  for (let i = r.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [r[i], r[j]] = [r[j], r[i]];
  }
  return r;
}
/** Measure Euclidean or Manhattan distance over all dimensions, padding absent coordinates with zero. */
export function vecDist(a, b, metric = "euclidean") {
  let s = 0;
  if (metric === "manhattan") {
    for (let i = 0; i < a.length; i++) s += Math.abs((a[i] || 0) - (b[i] || 0));
    return s;
  }
  for (let i = 0; i < a.length; i++) s += ((a[i] || 0) - (b[i] || 0)) ** 2;
  return Math.sqrt(s);
}
/** Subtract the largest logit before exponentiation to keep normalization numerically stable. */
export function softmax(z) {
  const m = Math.max(...z),
    e = z.map((v) => Math.exp(Math.max(-40, Math.min(40, v - m)))),
    s = e.reduce((a, b) => a + b, 0) || 1;
  return e.map((v) => v / s);
}
/** Count labels within a fixed class vocabulary, including classes absent from a sample. */
export function classCounts(rows, k) {
  const c = Array(k).fill(0);
  rows.forEach((d) => {
    if (d.y >= 0 && d.y < k) c[d.y]++;
  });
  return c;
}
/** K-means++ initialization favors points far from centers already selected. */
export function initCentroids(rows, k) {
  k = Math.min(Math.max(1, k), rows.length);
  const out = [];
  if (!rows.length) return out;
  out.push(rows[Math.floor(Math.random() * rows.length)].x.slice());
  while (out.length < k) {
    const ds = rows.map((r) =>
      Math.min(...out.map((c) => vecDist(r.x, c) ** 2)),
    );
    const sum = ds.reduce((a, b) => a + b, 0);
    if (sum <= 1e-12) {
      out.push(rows[Math.floor(Math.random() * rows.length)].x.slice());
      continue;
    }
    let t = Math.random() * sum,
      idx = 0;
    while (idx < ds.length - 1 && (t -= ds[idx]) > 0) idx++;
    out.push(rows[idx].x.slice());
  }
  return out;
}
/** Follow learned feature thresholds until reaching a classification leaf. */
export function treeClassify(node, x) {
  while (node && !node.leaf)
    node = (x[node.feature] || 0) <= node.threshold ? node.left : node.right;
  return node?.pred ?? 0;
}
/** Fit a residual tree by minimizing within-leaf squared error, as used by boosting. */
export function buildRegTree(rows, depth, maxDepth, minLeaf, features) {
  const mean = rows.reduce((s, r) => s + r.y, 0) / (rows.length || 1),
    node = { leaf: true, value: mean };
  if (depth >= maxDepth || rows.length < 2 * minLeaf) return node;
  let best = null;
  for (const f of features) {
    const sorted = rows.slice().sort((a, b) => a.x[f] - b.x[f]);
    let sum = rows.reduce((s, r) => s + r.y, 0),
      sq = rows.reduce((s, r) => s + r.y * r.y, 0),
      leftSum = 0,
      leftSq = 0;
    for (let i = 1; i < sorted.length; i++) {
      leftSum += sorted[i - 1].y;
      leftSq += sorted[i - 1].y ** 2;
      if (
        i < minLeaf ||
        sorted.length - i < minLeaf ||
        sorted[i - 1].x[f] === sorted[i].x[f]
      )
        continue;
      const n1 = i,
        n2 = sorted.length - i,
        sse1 = leftSq - (leftSum * leftSum) / n1,
        rightSum = sum - leftSum,
        rightSq = sq - leftSq,
        sse2 = rightSq - (rightSum * rightSum) / n2,
        score = sse1 + sse2;
      if (!best || score < best.score)
        best = { f, t: (sorted[i - 1].x[f] + sorted[i].x[f]) / 2, score };
    }
  }
  if (!best) return node;
  const left = rows.filter((r) => r.x[best.f] <= best.t),
    right = rows.filter((r) => r.x[best.f] > best.t);
  if (!left.length || !right.length) return node;
  return {
    leaf: false,
    feature: best.f,
    threshold: best.t,
    left: buildRegTree(left, depth + 1, maxDepth, minLeaf, features),
    right: buildRegTree(right, depth + 1, maxDepth, minLeaf, features),
  };
}
/** Follow learned thresholds and return the mean response stored at a regression leaf. */
export function treeValue(node, x) {
  while (node && !node.leaf)
    node = (x[node.feature] || 0) <= node.threshold ? node.left : node.right;
  return node?.value ?? 0;
}
/** Estimate priors and Gaussian, Bernoulli, or multinomial feature statistics per class. */
export function createNaiveStats(rows, k, variant, alpha) {
  const d = rows[0]?.x.length || 2;
  return Array.from({ length: k }, (_, c) => {
    const classRows = rows.filter((r) => r.y === c),
      count = classRows.length,
      n = count || 1,
      mean = Array(d).fill(0),
      varr = Array(d).fill(0),
      counts = Array(d).fill(0);
    for (let f = 0; f < d; f++) {
      mean[f] = classRows.reduce((s, r) => s + r.x[f], 0) / n;
      varr[f] = classRows.reduce((s, r) => s + (r.x[f] - mean[f]) ** 2, 0) / n;
      counts[f] = classRows.reduce(
        (s, r) =>
          s +
          (variant === "bernoulli"
            ? r.x[f] > 0
              ? 1
              : 0
            : Math.max(0, r.x[f])),
        0,
      );
    }
    const offset = Array.from({ length: d }, (_, f) =>
      variant === "multinomial" && count
        ? Math.max(0, -Math.min(...classRows.map((r) => r.x[f]))) + 1e-9
        : 0,
    );
    if (variant === "multinomial") {
      for (let f = 0; f < d; f++)
        counts[f] = classRows.reduce((sum, r) => sum + r.x[f] + offset[f], 0);
    }
    const total = counts.reduce((a, b) => a + b, 0);
    return {
      prior: count / rows.length,
      mean,
      varr: varr.map((v) => Math.max(1e-8, v + alpha * 1e-6)),
      offset,
      prob: counts.map((v) => (v + alpha) / (total + alpha * d || 1)),
      bernoulli: counts.map((v) => (v + alpha) / (n + 2 * alpha)),
    };
  });
}
/** Evaluate diagonal Gaussian log-likelihood plus the class log-prior. */
export function gaussianScore(x, st) {
  let v = Math.log(st.prior + 1e-12);
  for (let f = 0; f < x.length; f++)
    v -=
      0.5 *
      (Math.log(2 * Math.PI * st.varr[f]) +
        (x[f] - st.mean[f]) ** 2 / st.varr[f]);
  return v;
}
/** Gauss-Jordan elimination with pivoting returns both inverse and determinant for GMM densities. */
export function invertMatrix(a) {
  const n = a.length,
    m = a.map((r, i) =>
      r.slice().concat(Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))),
    );
  let det = 1;
  for (let i = 0; i < n; i++) {
    let pivot = i;
    for (let r = i + 1; r < n; r++)
      if (Math.abs(m[r][i]) > Math.abs(m[pivot][i])) pivot = r;
    if (Math.abs(m[pivot][i]) < 1e-12) return null;
    if (pivot !== i) {
      [m[pivot], m[i]] = [m[i], m[pivot]];
      det = -det;
    }
    const q = m[i][i];
    det *= q;
    for (let j = 0; j < 2 * n; j++) m[i][j] /= q;
    for (let r = 0; r < n; r++)
      if (r !== i) {
        const f = m[r][i];
        for (let j = 0; j < 2 * n; j++) m[r][j] -= f * m[i][j];
      }
  }
  return { inv: m.map((r) => r.slice(n)), det };
}
/** Evaluate a multivariate Gaussian density with a numerical floor for nearly singular fits. */
export function gaussianPdf(x, mu, cov) {
  const inv = invertMatrix(cov);
  if (!inv || inv.det <= 1e-15) return 1e-300;
  const z = x.map((v, i) => v - mu[i]);
  let q = 0;
  for (let i = 0; i < z.length; i++)
    for (let j = 0; j < z.length; j++) q += z[i] * inv.inv[i][j] * z[j];
  return Math.max(
    1e-300,
    Math.exp(-0.5 * q) / Math.sqrt(Math.pow(2 * Math.PI, x.length) * inv.det),
  );
}
