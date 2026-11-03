import { randn, sigmoid, softmax } from "./math.js";
/** Forward pass, backpropagation, and optimizers share one parameter layout. */
export function createNeuralAlgorithms({ pNum, isRegressionTask, getEngine }) {
  /** Initialize a one-hidden-layer tanh network and optimizer buffers with a shared flat layout. */
  function makeMLP(d, k, id) {
    const h = Math.round(pNum("hidden", 8, 1, 128)),
      thetaN = h * d + h + k * h + k;
    const W1 = Array.from({ length: h }, () =>
        Array.from({ length: d }, () => randn() * 0.25),
      ),
      b1 = Array(h).fill(0),
      W2 = Array.from({ length: k }, () =>
        Array.from({ length: h }, () => randn() * 0.25),
      ),
      b2 = Array(k).fill(0);
    return {
      kind: "mlp",
      d,
      h,
      k,
      W1,
      b1,
      W2,
      b2,
      opt: id === "mlp" ? "adam" : id,
      state: {
        m: Array(thetaN).fill(0),
        v: Array(thetaN).fill(0),
        vmax: Array(thetaN).fill(0),
        t: 0,
      },
      thetaN,
    };
  }
  /** Compute affine transforms and tanh activations, followed by regression, sigmoid, or softmax output. */
  function mlpForward(m, x) {
    const a = x.slice(),
      z1 = m.W1.map((row, j) =>
        row.reduce((s, v, i) => s + v * (a[i] || 0), m.b1[j]),
      ),
      h = z1.map(Math.tanh),
      z2 = m.W2.map((row, k) => row.reduce((s, v, j) => s + v * h[j], m.b2[k]));
    return {
      a,
      z1,
      h,
      z2,
      out: isRegressionTask(getEngine().id)
        ? z2
        : m.k === 1
          ? [sigmoid(z2[0])]
          : softmax(z2),
    };
  }
  /** Use a fixed W1, b1, W2, b2 ordering for both parameter vectors and gradient vectors. */
  function flattenMLP(m, g) {
    const arr = [];
    for (let j = 0; j < m.h; j++)
      for (let i = 0; i < m.d; i++) arr.push(g ? g.W1[j][i] : m.W1[j][i]);
    for (let j = 0; j < m.h; j++) arr.push(g ? g.b1[j] : m.b1[j]);
    for (let k = 0; k < m.k; k++)
      for (let j = 0; j < m.h; j++) arr.push(g ? g.W2[k][j] : m.W2[k][j]);
    for (let k = 0; k < m.k; k++) arr.push(g ? g.b2[k] : m.b2[k]);
    return arr;
  }
  /** Write a flat optimizer result back into the same matrix and bias layout. */
  function unflattenMLP(m, v) {
    let q = 0;
    for (let j = 0; j < m.h; j++)
      for (let i = 0; i < m.d; i++) m.W1[j][i] = v[q++];
    for (let j = 0; j < m.h; j++) m.b1[j] = v[q++];
    for (let k = 0; k < m.k; k++)
      for (let j = 0; j < m.h; j++) m.W2[k][j] = v[q++];
    for (let k = 0; k < m.k; k++) m.b2[k] = v[q++];
  }
  /** Backpropagate output error through tanh, then average gradients over the selected batch. */
  function mlpGrad(m, rows) {
    const g = {
        W1: m.W1.map((r) => r.map(() => 0)),
        b1: Array(m.h).fill(0),
        W2: m.W2.map((r) => r.map(() => 0)),
        b2: Array(m.k).fill(0),
      },
      regression = isRegressionTask(getEngine().id);
    let loss = 0;
    for (const row of rows) {
      const f = mlpForward(m, row.x),
        target = regression
          ? [row.y]
          : m.k === 1
            ? [row.y === 1 ? 1 : 0]
            : Array.from({ length: m.k }, (_, k) => (row.y === k ? 1 : 0)),
        dz = f.out.map((v, k) => v - target[k]);
      if (regression) loss += 0.5 * (f.out[0] - row.y) ** 2;
      else
        loss -= target.reduce(
          (s, t, k) => s + t * Math.log(f.out[k] + 1e-12),
          0,
        );
      // The output derivative is prediction minus target for softmax cross-entropy or MSE.
      const dh = Array(m.h).fill(0);
      for (let k = 0; k < m.k; k++) {
        g.b2[k] += dz[k];
        for (let j = 0; j < m.h; j++) {
          g.W2[k][j] += dz[k] * f.h[j];
          dh[j] += dz[k] * m.W2[k][j];
        }
      }
      for (let j = 0; j < m.h; j++) {
        // tanh derivative: 1 - activation squared. Chain it with the upstream gradient.
        const dz1 = dh[j] * (1 - f.h[j] * f.h[j]);
        g.b1[j] += dz1;
        for (let i = 0; i < m.d; i++) g.W1[j][i] += dz1 * (row.x[i] || 0);
      }
    }
    const n = Math.max(1, rows.length);
    for (const block of [g.W1, g.W2])
      for (const row of block) for (let i = 0; i < row.length; i++) row[i] /= n;
    for (const block of [g.b1, g.b2])
      for (let i = 0; i < block.length; i++) block[i] /= n;
    return { g, loss: loss / n };
  }
  /** Apply SGD or an adaptive update. Adam bias correction uses the completed update count. */
  function applyOptimizer(theta, grad, m, optimizer) {
    const s = m.state,
      p = (key) =>
        pNum(
          key,
          {
            learningRate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            momentum: 0.9,
            rho: 0.9,
            weightDecay: 0.01,
          }[key],
        );
    const lr = p("learningRate"),
      eps = p("epsilon"),
      mu = p("momentum"),
      b1 = p("beta1"),
      b2 = p("beta2"),
      rho = p("rho");
    s.t++;
    const out = theta.slice();
    let norm = 0;
    for (let i = 0; i < grad.length; i++) norm += grad[i] * grad[i];
    getEngine().gradNorm = Math.sqrt(norm);
    for (let i = 0; i < grad.length; i++) {
      const g = grad[i];
      if (optimizer === "sgd") out[i] -= lr * g;
      else if (optimizer === "momentum") {
        s.m[i] = mu * s.m[i] + g;
        out[i] -= lr * s.m[i];
      } else if (optimizer === "nesterov") {
        s.m[i] = mu * s.m[i] - lr * g;
        out[i] += s.m[i];
      } else if (optimizer === "adagrad") {
        s.v[i] += g * g;
        out[i] -= (lr * g) / (Math.sqrt(s.v[i]) + eps);
      } else if (optimizer === "rmsprop") {
        s.v[i] = rho * s.v[i] + (1 - rho) * g * g;
        out[i] -= (lr * g) / (Math.sqrt(s.v[i]) + eps);
      } else {
        s.m[i] = b1 * s.m[i] + (1 - b1) * g;
        s.v[i] = b2 * s.v[i] + (1 - b2) * g * g;
        // Correct the zero-initialized moment estimates before dividing by their scale.
        const mh = s.m[i] / (1 - Math.pow(b1, s.t)),
          vh = s.v[i] / (1 - Math.pow(b2, s.t));
        out[i] -= (lr * mh) / (Math.sqrt(vh) + eps);
        if (optimizer === "adamw") out[i] -= lr * p("weightDecay") * theta[i];
      }
    }
    return out;
  }
  return {
    makeMLP,
    mlpForward,
    flattenMLP,
    unflattenMLP,
    mlpGrad,
    applyOptimizer,
  };
}
