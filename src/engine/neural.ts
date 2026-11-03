import type {
  EngineState,
  MethodId,
  NeuralForward,
  NeuralGradient,
  NeuralModel,
  NeuralWeights,
  NumericParameter,
  Sample,
  Vector,
} from "../types.ts";
import { randn, sigmoid, softmax } from "./math.ts";

/**
 * Build neural operations around explicit experiment dependencies.
 *
 * getEngine() reads the current state rather than a captured model reference: resets
 * replace the state object. pNum supplies normalized hyperparameters, and task
 * inspection chooses continuous versus probabilistic outputs. This factory has no
 * browser or React dependency and can be reused by numerical tests.
 */
export function createNeuralAlgorithms({
  pNum,
  isRegressionTask,
  getEngine,
}: {
  pNum: NumericParameter;
  isRegressionTask: (id?: MethodId | null) => boolean;
  getEngine: () => EngineState;
}) {
  /**
   * Allocate a one-hidden-layer network and its optimizer memory.
   *
   * For d inputs, h hidden neurons, and k outputs, the parameter count is h*d+h+k*h+k.
   * W1 has shape h-by-d and W2 has shape k-by-h; biases match their output layers.
   * Small independent Gaussian weights break symmetry between hidden neurons.
   *
   * Optimizer buffers follow exactly the same flattening order as the parameters.
   * Sharing the layout prevents a gradient for one connection from updating another.
   */
  function makeMLP(d: number, k: number, id: MethodId): NeuralModel {
    const h = Math.round(pNum("hidden", 8, 1, 128)),
      thetaN = h * d + h + k * h + k;
    const W1 = Array.from({ length: h }, () =>
        Array.from({ length: d }, () => randn() * 0.25),
      ),
      b1 = Array<number>(h).fill(0),
      W2 = Array.from({ length: k }, () =>
        Array.from({ length: h }, () => randn() * 0.25),
      ),
      b2 = Array<number>(k).fill(0);
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
        m: Array<number>(thetaN).fill(0),
        v: Array<number>(thetaN).fill(0),
        vmax: Array<number>(thetaN).fill(0),
        t: 0,
      },
      thetaN,
    };
  }
  /**
   * Evaluate affine -> tanh -> affine -> task-specific output.
   *
   * z1 = W1*x+b1, h = tanh(z1), and z2 = W2*h+b2. Regression keeps z2 as a continuous
   * response. A one-output binary classifier applies sigmoid; multiclass output uses
   * softmax, whose entries sum to one. Intermediate values are returned because both
   * backpropagation and the educational network diagram need the same computation.
   *
   * @param m Network parameters and dimensions; no weights are changed by this call.
   * @param x Raw input coordinates, with missing coordinates treated as zero.
   */
  function mlpForward(m: NeuralModel, x: Vector): NeuralForward {
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
  /**
   * Serialize weights or matching gradients in the optimizer's canonical order.
   *
   * Order: all W1 rows, b1, all W2 rows, then b2. Supplying g serializes gradients;
   * omitting it serializes parameter values. This shared traversal is essential:
   * a matching number of elements alone would not detect a reordered parameter bug.
   */
  function flattenMLP(m: NeuralModel, g?: NeuralWeights): Vector {
    const arr = [];
    for (let j = 0; j < m.h; j++)
      for (let i = 0; i < m.d; i++) arr.push(g ? g.W1[j][i] : m.W1[j][i]);
    for (let j = 0; j < m.h; j++) arr.push(g ? g.b1[j] : m.b1[j]);
    for (let k = 0; k < m.k; k++)
      for (let j = 0; j < m.h; j++) arr.push(g ? g.W2[k][j] : m.W2[k][j]);
    for (let k = 0; k < m.k; k++) arr.push(g ? g.b2[k] : m.b2[k]);
    return arr;
  }
  /**
   * Restore a parameter vector using exactly the inverse flattenMLP layout.
   *
   * The vector is an internal optimizer result with a known length thetaN. Writing
   * back into existing matrices retains the model's architecture and lets optimizer
   * state remain aligned with each scalar parameter across successive updates.
   */
  function unflattenMLP(m: NeuralModel, v: Vector): void {
    let q = 0;
    for (let j = 0; j < m.h; j++)
      for (let i = 0; i < m.d; i++) m.W1[j][i] = v[q++];
    for (let j = 0; j < m.h; j++) m.b1[j] = v[q++];
    for (let k = 0; k < m.k; k++)
      for (let j = 0; j < m.h; j++) m.W2[k][j] = v[q++];
    for (let k = 0; k < m.k; k++) m.b2[k] = v[q++];
  }
  /**
   * Compute a batch-average loss and its gradients by reverse-mode differentiation.
   *
   * For the supported output/loss pair, the output derivative is prediction-target.
   * Accumulate W2 and b2 gradients, propagate through W2, then multiply by the tanh
   * derivative 1-h^2 to obtain hidden preactivation derivatives. Those derivatives
   * produce W1 and b1 gradients through the original input coordinates.
   *
   * Accumulate over rows before dividing by batch size. This keeps learning-rate
   * interpretation consistent when switching between full-batch and mini-batch modes.
   * The returned gradient has the same matrix/bias shape as the model and does not
   * itself update any parameter.
   */
  function mlpGrad(m: NeuralModel, rows: Sample[]): NeuralGradient {
    const g = {
        W1: m.W1.map((r) => r.map(() => 0)),
        b1: Array<number>(m.h).fill(0),
        W2: m.W2.map((r) => r.map(() => 0)),
        b2: Array<number>(m.k).fill(0),
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
      const dh = Array<number>(m.h).fill(0);
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
  /**
   * Transform a parameter vector using the chosen first-order optimizer.
   *
   * SGD subtracts alpha*g. Momentum stores a running direction; Nesterov receives a
   * gradient already measured at a look-ahead position. AdaGrad accumulates squared
   * gradients, while RMSProp exponentially discounts old squared gradients. Adam uses
   * both first and second moments and corrects their initial zero-buffer bias.
   *
   * AdamW applies weight decay separately from the adaptive gradient step. The small
   * epsilon stabilizes division by near-zero scales. Only optimizer buffers and the
   * reported gradient norm change here; the caller writes the returned vector back.
   */
  function applyOptimizer(
    theta: Vector,
    grad: Vector,
    m: NeuralModel,
    optimizer: MethodId,
  ): Vector {
    const s = m.state,
      p = (
        key:
          | "learningRate"
          | "beta1"
          | "beta2"
          | "epsilon"
          | "momentum"
          | "rho"
          | "weightDecay",
      ) =>
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
