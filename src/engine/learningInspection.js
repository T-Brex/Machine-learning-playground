/** Numerical explanations and visualization data derived from the live model. */
import { sigmoid } from "./math.js";
export function createLearningInspector({
  config,
  ENGINE_REGRESSION,
  ENGINE_CLUSTER,
  getEngine,
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
}) {
  /** Describe actual intermediate values from the first training row, without applying a training update. */
  function getAlgorithmLearningTrace() {
    const id = methodId();
    if (getEngine().id !== id || !getEngine().model) {
      reset();
      engineInit();
    }
    const m = getEngine().model,
      row = getEngine().train[0] || data[0];
    if (!m || !row)
      return {
        status: "Genera o carica un dataset per vedere i calcoli.",
        steps: [],
      };
    const x = row.x,
      steps = [],
      fmt = (v) =>
        Number.isFinite(v)
          ? Math.abs(v) < 1e-4 && v !== 0
            ? v.toExponential(3)
            : v.toFixed(4)
          : String(v),
      vec = (v) => `[${v.map(fmt).join(", ")}]`,
      matrix = (a, r = 3, c = 4) =>
        `[${a
          .slice(0, r)
          .map(
            (row) =>
              `[${row.slice(0, c).map(fmt).join(", ")}${row.length > c ? ", …" : ""}]`,
          )
          .join(", ")}${a.length > r ? ", …" : ""}]`,
      add = (title, detail, fullDetail) =>
        steps.push({ title, detail, ...(fullDetail ? { fullDetail } : {}) });
    const paramSummary = Object.entries(config.parameters)
      .map(([key, value]) => `${key}=${value}`)
      .join(" · ");
    add(
      "1 · Campione e parametri",
      `Campione di training reale: x = ${vec(x)}${Number.isFinite(row.y) ? `, target y = ${fmt(row.y)}` : ""}.\nTraining ${getEngine().train.length} campioni, validazione ${getEngine().test.length}; parametri selezionati: ${paramSummary || "nessun iperparametro aggiuntivo"}.`,
    );
    if (ENGINE_REGRESSION.has(id)) {
      const phi = rawRegressionFeatures(x, m.degree),
        z = regressionVector(m, x),
        parts = z.map((v, i) => `${fmt(m.w[i])}·${fmt(v)}`);
      const pred = m.w.reduce((s, v, i) => s + v * z[i], m.b),
        err = pred - row.y,
        lambda = id === "linear" ? 0 : pNum("lambda", 0.1, 0, 100),
        ratio =
          id === "lasso"
            ? 1
            : id === "linear" || id === "ridge"
              ? 0
              : pNum("l1Ratio", 0.5, 0, 1),
        grad = z.map((v, i) => err * v + lambda * (1 - ratio) * m.w[i]);
      add(
        "2 · Costruzione del vettore di feature",
        `φ(x) grezzo = ${vec(phi)}\nMedie del training = ${vec(m.means)}\nScale del training = ${vec(m.scales)}\nφ standardizzato = (φ−media)/scala = ${vec(z)}.`,
      );
      add(
        "3 · Prodotto pesi × feature e previsione",
        `Coefficienti w = ${vec(m.w)}; bias b = ${fmt(m.b)}.\nŷ = wᵀφ(x)+b = ${parts.join(" + ")} + ${fmt(m.b)} = ${fmt(pred)}.\nTarget = ${fmt(row.y)}; residuo e = ŷ−y = ${fmt(err)}; errore quadratico sul campione = e² = ${fmt(err * err)}.`,
      );
      add(
        "4 · Loss, regolarizzazione e gradiente",
        `MSE/2 sul training + penalità ${id}: λ=${fmt(lambda)}, ρ=${fmt(ratio)}.\nGradiente per campione (prima della media sul batch): ∂L/∂w ≈ e·φ + λ(1−ρ)w = ${vec(grad)}; ∂L/∂b ≈ e = ${fmt(err)}.\nAggiornamento reale: passo learningRate=${fmt(pNum("learningRate", 0.01, 1e-6, 2))}; w ← w−αg, poi soglia prossimale L1=${fmt(pNum("learningRate", 0.01, 1e-6, 2) * lambda * ratio)}; b ← b−α·media(e).`,
      );
    } else if (id === "logistic" || id === "softmax" || id === "perceptron") {
      const scores = m.w.map((w, c) =>
        w.reduce((s, v, i) => s + v * (x[i] || 0), m.b[c]),
      );
      if (classes === 2) {
        const diff = m.w[1].map((value, index) => value - m.w[0][index]),
          norm = Math.hypot(...diff),
          signed =
            norm > 1e-12
              ? ((scores[1] - scores[0]) * (row.y === 1 ? 1 : -1)) / norm
              : null;
        add(
          "Distanza dalla frontiera",
          signed === null
            ? "I pesi sono ancora nulli: dopo un aggiornamento sarà definita una distanza perpendicolare alla retta."
            : `Distanza firmata dal lato della classe corretta = ${row.y === 1 ? "" : "−"}[z₁−z₀]/‖w₁−w₀‖ = ${fmt(signed)}. Positiva significa che il campione è dal lato corretto; il confronto dopo il passo mostra se si è avvicinato o allontanato.`,
        );
      }
      if (id === "perceptron") {
        const predicted = scores.indexOf(Math.max(...scores)),
          lr = pNum("learningRate", 0.1, 1e-6, 2),
          trueRow = m.w[row.y] || [],
          gradient = x.map((v) => lr * v);
        add(
          "2 · Prodotto scalare per ogni classe",
          `Punteggi zₖ=wₖᵀx+bₖ = ${vec(scores)}.\nŷ = argmax(zₖ) = classe ${predicted}; classe reale = ${row.y}.`,
        );
        add(
          "3 · Aggiornamento Perceptron",
          predicted === row.y
            ? `La previsione è corretta: nessuna modifica. Learning rate α=${fmt(lr)}.`
            : `Errore: per la classe vera w_${row.y} ← w_${row.y}+αx = ${vec(trueRow)} + ${vec(gradient)}; per la classe predetta w_${predicted} ← w_${predicted}−αx. I bias corrispondenti aumentano/diminuiscono di α=${fmt(lr)}.`,
        );
      } else {
        const probs = softmax(scores),
          target = Array.from({ length: classes }, (_, c) =>
            row.y === c ? 1 : 0,
          ),
          loss = -Math.log(Math.max(1e-12, probs[row.y] || 0)),
          grad = probs.map((p, c) => x.map((v) => (p - target[c]) * v));
        add(
          "2 · Matrice dei pesi × input",
          `W (prime righe) = ${matrix(m.w)}; b = ${vec(m.b)}.\nz = Wx+b = ${vec(scores)}.\nSoftmax(z) = exp(zₖ)/Σⱼexp(zⱼ) = ${vec(probs)}; target one-hot = ${vec(target)}.`,
          `Matrice W completa (${m.w.length}×${m.w[0]?.length || 0}) = ${matrix(m.w, m.w.length, m.w[0]?.length || 0)}\nBias completo b = ${vec(m.b)}\nPunteggi z = ${vec(scores)}\nProbabilità softmax = ${vec(probs)}\nTarget one-hot = ${vec(target)}`,
        );
        add(
          "3 · Errore e gradiente",
          `Cross-entropy del campione = −log(p_classe_vera) = −log(${fmt(probs[row.y] || 0)}) = ${fmt(loss)}.\n∂L/∂z = p−y = ${vec(probs.map((p, c) => p - target[c]))}.\n∂L/∂W per il campione = (p−y)xᵀ = ${matrix(grad)}; il motore media questi gradienti sul batch e applica il learning rate.`,
          `Gradiente completo ∂L/∂W = ${matrix(grad, grad.length, grad[0]?.length || 0)}\nGradiente bias ∂L/∂b = ${vec(probs.map((p, c) => p - target[c]))}`,
        );
        add(
          "4 · Aggiornamento dei parametri",
          `W ← W−α(∇W+λW), b ← b−α∇b; α=${fmt(pNum("learningRate", 0.08, 1e-6, 2))}, λ=${fmt(pNum("lambda", 0, 0, 100))}.\nIl vettore di pesi aggiornato è quello visibile nei risultati; premi “Esegui un passo reale” per applicare un batch al modello.`,
        );
      }
    } else if (id === "knn") {
      const metric = param("metric", "euclidean"),
        k = Math.round(pNum("k", 5, 1, getEngine().train.length)),
        weighted = param("weights", "uniform") === "distance",
        allDistances = getEngine()
          .train.map((r, i) => ({ i, y: r.y, d: vecDist(r.x, x, metric) }))
          .sort((a, b) => a.d - b.d),
        neighbors = allDistances.slice(0, k),
        votes = Array(Math.max(1, classes)).fill(0);
      neighbors.forEach((n) => (votes[n.y] += weighted ? 1 / (n.d + 1e-8) : 1));
      add(
        "2 · Distanze dal campione",
        `Metrica: ${metric}. Per ogni xᵢ calcola d(x,xᵢ); vicini più prossimi (indice, distanza, classe):\n${neighbors.map((n) => `x${n.i}: d=${fmt(n.d)}, y=${n.y}`).join("\n")}.`,
        `Tutte le distanze ordinate (indice, distanza, classe):\n${allDistances.map((n, i) => `${i < k ? "* " : ""}x${n.i}: d=${fmt(n.d)}, y=${n.y}`).join("\n")}\n* I primi k=${k} contribuiscono al voto.`,
      );
      add(
        "3 · Voto dei k vicini",
        `k=${k}, pesatura=${weighted ? "1/(d+ε)" : "voto uniforme"}.\nVoti per classe = ${vec(votes)}; classe predetta = argmax(voti) = ${votes.indexOf(Math.max(...votes))}.\nKNN memorizza i dati: non aggiorna pesi con gradient descent.`,
      );
    } else if (id === "svmrbf" || id === "svmpoly") {
      const contributions = m.alpha.map((alpha, c) => {
        let score = m.bias[c],
          lines = [],
          allLines = [];
        alpha.forEach((a, i) => {
          if (a <= 1e-8 || i >= getEngine().train.length) return;
          const support = getEngine().train[i],
            kernel = svmKernel(support.x, x, id),
            term = a * m.labels[c][i] * kernel;
          score += term;
          const line = `α${i}=${fmt(a)} · y=${m.labels[c][i]} · K=${fmt(kernel)} → ${fmt(term)}`;
          allLines.push(line);
          if (lines.length < 4) lines.push(line);
        });
        return { score, lines, allLines };
      });
      add(
        "2 · Kernel tra support vector e campione",
        id === "svmrbf"
          ? `K(x,z)=exp(−γ‖x−z‖²), γ=${fmt(pNum("gamma", 0.5, 1e-6, 100))}.`
          : `K(x,z)=(γxᵀz+r)^d, γ=${fmt(pNum("gamma", 0.5, 1e-6, 100))}, r=${fmt(pNum("coef0", 1, -100, 100))}, d=${Math.round(pNum("degree", 3, 1, 8))}.`,
      );
      add(
        "3 · Somma dei contributi e decisione",
        `${contributions.map((c, i) => `classe ${i}: b=${fmt(m.bias[i])}; ${c.lines.length ? c.lines.join(" + ") : "nessun support vector attivo"}; f(x)=${fmt(c.score)}`).join("\n")}\nLa classe col punteggio massimo è ${enginePredict(x)}. C=${fmt(pNum("C", 1, 1e-4, 1e4))} limita i moltiplicatori duali.`,
        `Contributi di tutti i support vector:\n${contributions.map((c, i) => `Classe ${i}, bias=${fmt(m.bias[i])}:\n${c.allLines.length ? c.allLines.join("\n") : "nessun support vector attivo"}\nPunteggio risultante f(x)=${fmt(c.score)}`).join("\n\n")}`,
      );
      add(
        "4 · Margine e perdita",
        `Per la classe corretta y=${row.y}, margine y·f(x)=${fmt((row.y === 0 ? -1 : 1) * contributions[row.y]?.score)}; hinge loss = max(0,1−margine). I moltiplicatori α dei support vector sono quelli del modello realmente addestrato.`,
      );
    } else if (id === "tree" || id === "forest") {
      const criterion = param("criterion", "gini"),
        route = (root) => {
          let node = root,
            subset = getEngine().train,
            path = [];
          while (node && !node.leaf) {
            const value = x[node.feature] || 0,
              left = value <= node.threshold,
              leftRows = subset.filter(
                (r) => r.x[node.feature] <= node.threshold,
              ),
              rightRows = subset.filter(
                (r) => r.x[node.feature] > node.threshold,
              ),
              parentImpurity = impurity(subset, criterion),
              weighted =
                (leftRows.length * impurity(leftRows, criterion) +
                  rightRows.length * impurity(rightRows, criterion)) /
                (subset.length || 1),
              gain = parentImpurity - weighted;
            path.push(
              `x${node.feature + 1}=${fmt(value)} ${left ? "≤" : ">"} soglia ${fmt(node.threshold)} → ${left ? "sinistra" : "destra"}; impurità ${criterion}: ${fmt(parentImpurity)} → pesata ${fmt(weighted)}, riduzione ${fmt(gain)}; conteggi figli ${leftRows.length}/${rightRows.length}`,
            );
            subset = left ? leftRows : rightRows;
            node = left ? node.left : node.right;
          }
          return { pred: node?.pred ?? 0, path, node, subset };
        };
      if (id === "tree") {
        const r = route(m.root);
        add(
          "2 · Percorso degli split",
          m.root
            ? r.path.join("\n")
            : `Il tree non ha ancora split: il modello verrà costruito al training.`,
        );
        add(
          "3 · Foglia e previsione",
          m.root
            ? `Campioni nella foglia = ${r.subset.length}; conteggi classe = ${vec(r.node.counts)}; frequenze = ${vec(r.node.counts.map((n) => n / (r.subset.length || 1)))}; classe assegnata = argmax conteggi = ${r.pred}.`
            : "Esegui un passo reale per costruire gli split e mostrare la foglia raggiunta.",
        );
      } else {
        const allPaths = m.trees.map((t, i) => {
            const r = route(t);
            return `Albero ${i + 1}: ${r.path.join(" → ") || "foglia radice"} ⇒ classe ${r.pred}`;
          }),
          paths = allPaths.slice(0, 5),
          votes = Array(Math.max(classes, 1)).fill(0);
        m.trees.forEach((t) => votes[route(t).pred]++);
        add(
          "2 · Predizione dei singoli alberi",
          paths.length
            ? paths.join("\n")
            : "La foresta non ha ancora alberi: esegui il training.",
          allPaths.join("\n"),
        );
        add(
          "3 · Voto aggregato",
          `Voti (tutti gli alberi) = ${vec(votes)}; classe finale = ${votes.indexOf(Math.max(...votes))}. Alberi addestrati: ${m.trees.length}.`,
        );
      }
    } else if (id === "boost") {
      const rate = pNum("learningRate", 0.1, 1e-4, 1);
      if (m.regression) {
        const stageValues = m.trees[0].map((t) => rate * treeValue(t, x)),
          prediction = m.base[0] + stageValues.reduce((a, b) => a + b, 0);
        add(
          "2 · Valore base e correzioni",
          `F₀=${fmt(m.base[0])}; learning rate η=${fmt(rate)}.\nContributi degli stadi η·hₘ(x) = ${vec(stageValues.slice(0, 8))}${stageValues.length > 8 ? " …" : ""}.`,
          `Tutti i contributi degli stadi: ${vec(stageValues)}\nF₀=${fmt(m.base[0])}; somma correzioni=${fmt(stageValues.reduce((a, b) => a + b, 0))}; previsione=${fmt(prediction)}.`,
        );
        add(
          "3 · Somma progressiva",
          `ŷ=F₀+Σηhₘ(x)=${fmt(m.base[0])}+${fmt(stageValues.reduce((a, b) => a + b, 0))}=${fmt(prediction)}; target=${fmt(row.y)}, residuo=${fmt(prediction - row.y)}.`,
        );
      } else {
        const score = m.base.map(
            (v, c) =>
              v + m.trees[c].reduce((s, t) => s + rate * treeValue(t, x), 0),
          ),
          probs = softmax(score),
          details = m.trees
            .map(
              (trees, c) =>
                `Classe ${c}: base=${fmt(m.base[c])}; contributi ${vec(trees.map((t) => rate * treeValue(t, x)))}; logit=${fmt(score[c])}`,
            )
            .join("\n");
        add(
          "2 · Punteggi per classe",
          `Punteggio iniziale + correzioni sequenziali ηhₘ(x), η=${fmt(rate)}:\nz=${vec(score)}.\nOgni albero corregge i residui/gradienti rimasti agli stadi precedenti.`,
          details,
        );
        add(
          "3 · Normalizzazione e classe",
          `Softmax(z)=${vec(probs)}; classe prevista=${probs.indexOf(Math.max(...probs))}; stadi per classe=${vec(m.trees.map((t) => t.length))}.`,
        );
      }
    } else if (id === "naive") {
      const variant = param("variant", "gaussian"),
        stats = m.stats.length
          ? m.stats
          : createNaiveStats(
              getEngine().train,
              Math.max(1, classes),
              variant,
              pNum("smoothing", 1, 0, 100),
            ),
        scores = stats.map((st) => naiveScore(x, st, variant)),
        prob = softmax(scores);
      const likelihoodDetails = stats
        .map((st, c) => {
          const terms = x.map((value, f) => {
            if (variant === "gaussian")
              return (
                -0.5 *
                (Math.log(2 * Math.PI * st.varr[f]) +
                  (value - st.mean[f]) ** 2 / st.varr[f])
              );
            if (variant === "bernoulli") {
              const q = Math.max(1e-9, Math.min(1 - 1e-9, st.bernoulli[f])),
                bit = value > 0 ? 1 : 0;
              return bit * Math.log(q) + (1 - bit) * Math.log(1 - q);
            }
            return (
              Math.max(0, value + (st.offset?.[f] || 0)) *
              Math.log(Math.max(st.prob[f], 1e-12))
            );
          });
          return `Classe ${c}: log prior=${fmt(Math.log(st.prior + 1e-12))}; termini feature=${vec(terms)}; somma=${fmt(scores[c])}`;
        })
        .join("\n");
      add(
        "2 · Statistiche per classe",
        stats
          .map(
            (st, c) =>
              `Classe ${c}: prior=${fmt(st.prior)}; media=${vec(st.mean)}; varianza=${vec(st.varr)}; probabilità feature=${vec(st.prob)}.`,
          )
          .join("\n"),
        `Variante ${variant}; smoothing α=${fmt(pNum("smoothing", 1, 0, 100))}.\n${stats.map((st, c) => `Classe ${c}: prior=${fmt(st.prior)}; μ=${vec(st.mean)}; σ²=${vec(st.varr)}; probabilità multinomiali=${vec(st.prob)}; Bernoulli=${vec(st.bernoulli)}`).join("\n")}`,
      );
      add(
        "3 · Log-probabilità e decisione",
        `Variante=${variant}. log P(y|x) (a costante vicino) = ${vec(scores)}.\nNormalizzazione delle probabilità = ${vec(prob)}; classe scelta=${scores.indexOf(Math.max(...scores))}.`,
        likelihoodDetails,
      );
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
      const theta = flattenMLP(m);
      let traceModel = m;
      if (m.opt === "nesterov") {
        const lookahead = theta.map((v, i) => v + (m.state.m[i] || 0));
        const copy = {
          ...m,
          W1: m.W1.map((r) => r.slice()),
          b1: m.b1.slice(),
          W2: m.W2.map((r) => r.slice()),
          b2: m.b2.slice(),
        };
        unflattenMLP(copy, lookahead);
        traceModel = copy;
      }
      const f = mlpForward(traceModel, x),
        g = mlpGrad(traceModel, [row]),
        grad = flattenMLP(m, g.g),
        atLookahead = m.opt === "nesterov" ? "Look-ahead θ+vₜ₋₁. " : "";
      add(
        "2 · Primo strato: W⁽¹⁾x+b⁽¹⁾ e tanh",
        `${atLookahead}W⁽¹⁾ (porzione) = ${matrix(traceModel.W1)}; b⁽¹⁾ = ${vec(traceModel.b1.slice(0, 6))}.\nz⁽¹⁾ = W⁽¹⁾x+b⁽¹⁾ = ${vec(f.z1.slice(0, 6))}; a⁽¹⁾=tanh(z⁽¹⁾) = ${vec(f.h.slice(0, 6))}.`,
        `${atLookahead}W⁽¹⁾ completa (${m.h}×${m.d}) = ${matrix(traceModel.W1, traceModel.W1.length, traceModel.W1[0]?.length || 0)}\nb⁽¹⁾ = ${vec(traceModel.b1)}\nx = ${vec(x)}\nz⁽¹⁾ = W⁽¹⁾x+b⁽¹⁾ = ${vec(f.z1)}\na⁽¹⁾ = tanh(z⁽¹⁾) = ${vec(f.h)}`,
      );
      add(
        "3 · Strato di output",
        `${atLookahead}W⁽²⁾ (porzione) = ${matrix(traceModel.W2)}; b⁽²⁾=${vec(traceModel.b2)}.\nz⁽²⁾=W⁽²⁾a⁽¹⁾+b⁽²⁾=${vec(f.z2)}; ŷ=${vec(f.out)}.\nJ del campione (${classes === 1 ? "MSE/2" : "cross-entropy"})=${fmt(g.loss)}.`,
        `W⁽²⁾ completa (${m.k}×${m.h}) = ${matrix(traceModel.W2, traceModel.W2.length, traceModel.W2[0]?.length || 0)}\nb⁽²⁾ = ${vec(traceModel.b2)}\nz⁽²⁾ = W⁽²⁾a⁽¹⁾+b⁽²⁾ = ${vec(f.z2)}\nŷ = ${vec(f.out)}; J = ${fmt(g.loss)}`,
      );
      const state = m.state,
        index = 0,
        lr = pNum("learningRate", 0.001, 1e-6, 2),
        eps = pNum("epsilon", 1e-8),
        mu = pNum("momentum", 0.9),
        beta1 = pNum("beta1", 0.9),
        beta2 = pNum("beta2", 0.999),
        rho = pNum("rho", 0.9),
        g0 = grad[0] || 0,
        theta0 = theta[0] || 0,
        t = state.t + 1;
      let delta = 0,
        updateMath = "";
      if (m.opt === "sgd") {
        delta = -lr * g0;
        updateMath = `Δθ₀=−αg₀=−${fmt(lr)}·${fmt(g0)}=${fmt(delta)}`;
      } else if (m.opt === "momentum") {
        const v = mu * state.m[index] + g0;
        delta = -lr * v;
        updateMath = `v₀=${fmt(mu)}·${fmt(state.m[index])}+${fmt(g0)}=${fmt(v)}; Δθ₀=−αv₀=${fmt(delta)}`;
      } else if (m.opt === "nesterov") {
        const v = mu * state.m[index] - lr * g0;
        delta = v;
        updateMath = `v₀=${fmt(mu)}·${fmt(state.m[index])}−${fmt(lr)}·${fmt(g0)}=${fmt(v)}; Δθ₀=v₀=${fmt(delta)}`;
      } else if (m.opt === "adagrad") {
        const acc = state.v[index] + g0 * g0;
        delta = (-lr * g0) / (Math.sqrt(acc) + eps);
        updateMath = `r₀=${fmt(state.v[index])}+g₀²=${fmt(acc)}; Δθ₀=−αg₀/(√r₀+ε)=${fmt(delta)}`;
      } else if (m.opt === "rmsprop") {
        const acc = rho * state.v[index] + (1 - rho) * g0 * g0;
        delta = (-lr * g0) / (Math.sqrt(acc) + eps);
        updateMath = `r₀=${fmt(rho)}·${fmt(state.v[index])}+(1−${fmt(rho)})·${fmt(g0 * g0)}=${fmt(acc)}; Δθ₀=${fmt(delta)}`;
      } else {
        const mt = beta1 * state.m[index] + (1 - beta1) * g0,
          vt = beta2 * state.v[index] + (1 - beta2) * g0 * g0,
          mhat = mt / (1 - Math.pow(beta1, t)),
          vhat = vt / (1 - Math.pow(beta2, t));
        delta = (-lr * mhat) / (Math.sqrt(vhat) + eps);
        updateMath = `m₀=${fmt(mt)}, v₀=${fmt(vt)}; m̂₀=${fmt(mhat)}, v̂₀=${fmt(vhat)}; ΔAdam=${fmt(delta)}`;
        if (m.opt === "adamw") {
          const decay = pNum("weightDecay", 0.01);
          delta -= lr * decay * theta0;
          updateMath += `; Δdecay=−α·λ·θ₀=${fmt(-lr * decay * theta0)}`;
        }
      }
      add(
        "4 · Backpropagation e ottimizzatore",
        `Gradiente ∇θ del campione: ${vec(grad.slice(0, 8))}${grad.length > 8 ? " …" : ""}; parametri θ corrispondenti: ${vec(theta.slice(0, 8))}${theta.length > 8 ? " …" : ""}.\nOttimizzatore=${m.opt}; learning rate=${fmt(lr)}. Prossimo aggiornamento del primo parametro (senza applicarlo): ${updateMath}; θ₀ nuovo=${fmt(theta0 + delta)}.`,
        `∇W⁽¹⁾ = ${matrix(g.g.W1, g.g.W1.length, g.g.W1[0]?.length || 0)}\n∇b⁽¹⁾ = ${vec(g.g.b1)}\n∇W⁽²⁾ = ${matrix(g.g.W2, g.g.W2.length, g.g.W2[0]?.length || 0)}\n∇b⁽²⁾ = ${vec(g.g.b2)}\nW⁽¹⁾ corrente = ${matrix(m.W1, m.W1.length, m.W1[0]?.length || 0)}\nb⁽¹⁾ corrente = ${vec(m.b1)}\nW⁽²⁾ corrente = ${matrix(m.W2, m.W2.length, m.W2[0]?.length || 0)}\nb⁽²⁾ corrente = ${vec(m.b2)}\nCalcolo del primo elemento: ${updateMath}; θ₀ nuovo=${fmt(theta0 + delta)}. Il valore è illustrativo del prossimo passo, senza mutare il modello.`,
      );
    } else if (id === "kmeans") {
      const distances = m.centroids.map((c) => vecDist(c, x)),
        cluster = distances.indexOf(Math.min(...distances)),
        delta = x.map((v, i) => v - m.centroids[cluster][i]);
      add(
        "2 · Distanza da ogni centroide",
        `Centroidi μₖ = ${matrix(m.centroids)}.\nd(x,μₖ) = ${vec(distances)}; il centro più vicino è μ_${cluster} (cluster ${cluster}).`,
      );
      const groups = m.centroids.map((_, c) =>
          getEngine().train.filter((r) => enginePredict(r.x) === c),
        ),
        sums = groups.map((group) =>
          x.map((_, f) => group.reduce((sum, r) => sum + r.x[f], 0)),
        ),
        means = groups.map((group, c) =>
          group.length ? sums[c].map((v) => v / group.length) : m.centroids[c],
        );
      add(
        "3 · Assegnazione e aggiornamento centroide",
        `Assegnazione: argminₖ‖x−μₖ‖² = ${cluster}; residuo vettoriale x−μ = ${vec(delta)}.\nIl passo Lloyd riassegna tutti i punti e aggiorna μₖ come media dei vettori assegnati. Inertia training=${fmt(engineLoss())}.`,
        `Conteggi per cluster = ${vec(groups.map((g) => g.length))}\nSomme vettoriali Σxᵢ = ${matrix(sums, sums.length, sums[0]?.length || 0)}\nCentroidi correnti μ = ${matrix(m.centroids, m.centroids.length, m.centroids[0]?.length || 0)}\nMedie (prossimo centroide stimato) = ${matrix(means, means.length, means[0]?.length || 0)}\nAssegnazioni di tutti i campioni:\n${getEngine()
          .train.map(
            (r, i) => `x${i}: ${vec(r.x)} → cluster ${enginePredict(r.x)}`,
          )
          .join("\n")}`,
      );
    } else if (id === "dbscan") {
      const eps = pNum("eps", 0.5, 1e-4, 100),
        minPts = Math.round(pNum("minPts", 5, 1, 1000)),
        metric = param("metric", "euclidean"),
        neighbors = getEngine()
          .train.map((r, i) => ({
            i,
            d: vecDist(r.x, x, metric),
            label: m.labels[i],
          }))
          .filter((r) => r.d <= eps),
        nearest = getEngine()
          .train.map((r, i) => ({
            i,
            d: vecDist(r.x, x, metric),
            label: m.labels[i],
          }))
          .filter((r) => r.label >= 0)
          .sort((a, b) => a.d - b.d)[0];
      add(
        "2 · Regione ε e densità",
        `ε=${fmt(eps)}, minPts=${minPts}, metrica=${metric}.\nVicini entro ε: ${neighbors.length}${
          neighbors.length
            ? ` (indici ${neighbors
                .slice(0, 8)
                .map((n) => n.i)
                .join(", ")}${neighbors.length > 8 ? ", …" : ""})`
            : ""
        }; confronto con minPts ${neighbors.length >= minPts ? "soddisfatto" : "non soddisfatto"}.`,
        `Distanze da tutti i campioni (indice, distanza, etichetta):\n${getEngine()
          .train.map((r, i) => ({
            i,
            d: vecDist(r.x, x, metric),
            label: m.labels[i],
          }))
          .map(
            (v) =>
              `x${v.i}: d=${fmt(v.d)}, cluster precedente=${v.label ?? "non ancora assegnato"}${v.d <= eps ? " · entro ε" : ""}`,
          )
          .join("\n")}`,
      );
      add(
        "3 · Nucleo, bordo o rumore",
        `Regola di assegnazione del modello: nearest labeled core ${nearest ? `x${nearest.i} a distanza ${fmt(nearest.d)}, cluster ${nearest.label}` : "non trovato"}.\nPrevisione corrente=${enginePredict(x) < 0 ? "rumore (−1)" : `cluster ${enginePredict(x)}`}. DBSCAN non ottimizza una loss differenziabile.`,
      );
    } else if (id === "gmm") {
      const weighted = m.means.map(
          (mu, c) => m.mix[c] * gaussianPdf(x, mu, m.covs[c]),
        ),
        total = weighted.reduce((a, b) => a + b, 0),
        posterior = weighted.map((v) => v / (total || 1));
      add(
        "2 · Densità di ogni componente",
        `Matrice di covarianza Σ (componente 1, porzione) = ${matrix(m.covs[0] || [])}.\nPesi πₖ=${vec(m.mix)}; μ (porzione)=${matrix(m.means)}.\nπₖ·N(x|μₖ,Σₖ) = ${vec(weighted)}.`,
        `Pesi πₖ = ${vec(m.mix)}\nMedie μₖ = ${matrix(m.means, m.means.length, m.means[0]?.length || 0)}\n${m.covs.map((cov, c) => `Covarianza Σ${c} = ${matrix(cov, cov.length, cov[0]?.length || 0)}\nDensità N(x|μ${c},Σ${c})=${fmt(gaussianPdf(x, m.means[c], cov))}; contributo π${c}N=${fmt(weighted[c])}`).join("\n")}`,
      );
      add(
        "3 · E-step: responsabilità",
        `rₖ=πₖNₖ/ΣⱼπⱼNⱼ = ${vec(posterior)}; somma responsabilità=${fmt(posterior.reduce((a, b) => a + b, 0))}.\nLa componente più probabile è ${posterior.indexOf(Math.max(...posterior))}.`,
      );
      add(
        "4 · M-step e obiettivo",
        `M-step: usa rₖ come pesi per aggiornare μₖ, Σₖ e πₖ con tutti i campioni.\nNegative log-likelihood training=${fmt(engineLoss())}; iterazione EM=${getEngine().iteration}.`,
      );
    } else {
      add(
        "2 · Modello",
        `Tipo modello: ${m.kind}. Calcolo in ingresso x=${vec(x)}.`,
      );
      add(
        "3 · Risultato",
        `Previsione effettiva del motore = ${enginePredict(x)}.`,
      );
    }
    if (getEngine().previousModel && ENGINE_REGRESSION.has(id)) {
      const previous = getEngine().previousModel,
        oldPrediction = previous.w.reduce(
          (sum, value, index) =>
            sum + value * regressionVector(previous, x)[index],
          previous.b,
        ),
        newPrediction = m.w.reduce(
          (sum, value, index) => sum + value * regressionVector(m, x)[index],
          m.b,
        ),
        delta = m.w.map((value, index) => value - previous.w[index]);
      add(
        "Aggiornamento realmente applicato",
        `Confronto del passo ${getEngine().iteration - 1} → ${getEngine().iteration}: Δw = ${vec(delta)}, Δb = ${fmt(m.b - previous.b)}.\nSul campione x selezionato: previsione ${fmt(oldPrediction)} → ${fmt(newPrediction)}; residuo assoluto ${fmt(Math.abs(row.y - oldPrediction))} → ${fmt(Math.abs(row.y - newPrediction))}. Il batch può migliorare alcuni campioni e peggiorarne altri.`,
      );
    } else if (
      getEngine().previousModel &&
      ["logistic", "softmax", "perceptron"].includes(id) &&
      classes === 2
    ) {
      const previous = getEngine().previousModel,
        deltaW = m.w.map((weights, c) =>
          weights.map((value, index) => value - previous.w[c][index]),
        ),
        deltaB = m.b.map((value, c) => value - previous.b[c]),
        margin = (model) => {
          const scores = model.w.map((weights, c) =>
              weights.reduce(
                (sum, value, index) => sum + value * (x[index] || 0),
                model.b[c],
              ),
            ),
            norm = Math.hypot(
              ...model.w[row.y].map(
                (value, index) => value - model.w[1 - row.y][index],
              ),
            );
          return norm > 1e-12
            ? (scores[row.y] - scores[1 - row.y]) / norm
            : null;
        },
        oldMargin = margin(previous),
        newMargin = margin(m);
      add(
        "Aggiornamento realmente applicato",
        `Confronto del passo ${getEngine().iteration - 1} → ${getEngine().iteration}: ΔW = ${matrix(deltaW, deltaW.length, deltaW[0]?.length || 0)}, Δb = ${vec(deltaB)}.\nSul campione selezionato, distanza firmata dal lato corretto: ${oldMargin === null ? "non definita (norma iniziale zero)" : fmt(oldMargin)} → ${newMargin === null ? "non definita (norma zero)" : fmt(newMargin)}${oldMargin === null ? "" : ` (${newMargin >= oldMargin ? "migliora" : "peggiora"})`}.`,
      );
    }
    if (getEngine().fitted) {
      const metrics = Object.entries(getEngine().metrics || {})
        .filter(
          ([, value]) => typeof value !== "number" || Number.isFinite(value),
        )
        .map(
          ([key, value]) =>
            `${key}: ${typeof value === "number" ? fmt(value) : value}`,
        )
        .join("\n");
      add(
        "Valutazione sul set di validazione",
        metrics || `Validation loss=${fmt(evaluationLoss(getEngine().test))}.`,
      );
    }
    return {
      status: getEngine().fitted
        ? `Traccia del modello addestrato · passo ${getEngine().iteration}. Il campione mostrato è un esempio concreto; le metriche finali usano l’intero set di validazione.`
        : "Modello inizializzato ma non addestrato. I calcoli mostrano i parametri iniziali; esegui un passo reale per vedere l’aggiornamento. Il training applica i calcoli al batch configurato.",
      steps,
    };
  }
  /** Build boundaries, residuals, neighborhoods, or model surfaces for the selected training sample. */
  function getAlgorithmLearningVisualization(selectedSampleIndex = 0) {
    const id = methodId();
    if (getEngine().id !== id || !getEngine().model) {
      reset();
      engineInit();
    }
    const rows = getEngine().train,
      m = getEngine().model,
      selectedRowIndex = rows.length
        ? Math.max(
            0,
            Math.min(
              rows.length - 1,
              Math.round(Number(selectedSampleIndex) || 0),
            ),
          )
        : 0;
    if (isRegressionTask(id)) {
      const predict = (model, x) => {
        if (ENGINE_REGRESSION.has(id)) {
          const features = regressionVector(model, x);
          return model.w.reduce(
            (sum, weight, index) => sum + weight * features[index],
            model.b,
          );
        }
        if (id === "boost")
          return (
            model.base[0] +
            model.trees[0].reduce(
              (sum, tree) =>
                sum + pNum("learningRate", 0.1, 1e-4, 1) * treeValue(tree, x),
              0,
            )
          );
        return mlpForward(model, x).out[0];
      };
      const entries = rows.map((row, index) => {
        const prediction = predict(getEngine().model, row.x),
          previousPrediction = getEngine().previousModel
            ? predict(getEngine().previousModel, row.x)
            : null,
          residual = row.y - prediction,
          previousResidual =
            previousPrediction === null ? null : row.y - previousPrediction;
        return {
          index: index + 1,
          x: row.x,
          y: row.y,
          before: previousPrediction,
          after: prediction,
          beforeDistance: previousResidual,
          afterDistance: residual,
          change:
            previousResidual === null
              ? null
              : Math.abs(previousResidual) - Math.abs(residual),
          selected:
            index ===
            Math.max(
              0,
              Math.min(
                rows.length - 1,
                Math.round(Number(selectedSampleIndex) || 0),
              ),
            ),
        };
      });
      const xValues = rows.map((row) => row.x[0] || 0),
        xMin = Math.min(...xValues),
        xMax = Math.max(...xValues),
        curve = Array.from({ length: 100 }, (_, index) => {
          const x = xMin + ((xMax - xMin) * index) / 99,
            features = regressionPlotInput(x, rows[0]?.x.length || 1);
          return {
            x,
            before: getEngine().previousModel
              ? predict(getEngine().previousModel, features)
              : null,
            after: predict(getEngine().model, features),
          };
        });
      return {
        kind: "regression",
        iteration: getEngine().iteration,
        hasPrevious: !!getEngine().previousModel,
        dimensions: rows[0]?.x.length || 1,
        entries,
        curve,
      };
    }
    if (["logistic", "softmax", "perceptron"].includes(id) && classes === 2) {
      const means = Array.from(
        { length: rows[0]?.x.length || 0 },
        (_, index) =>
          rows.reduce((sum, row) => sum + (row.x[index] || 0), 0) /
          (rows.length || 1),
      );
      const geometry = (model) => {
        if (!model) return null;
        const weights = model.w[1].map(
            (value, index) => value - model.w[0][index],
          ),
          bias = model.b[1] - model.b[0],
          norm = Math.hypot(...weights);
        if (norm < 1e-12) return null;
        const sliceBias =
          bias +
          weights
            .slice(2)
            .reduce(
              (sum, value, index) => sum + value * (means[index + 2] || 0),
              0,
            );
        return { weights, bias, norm, intercept: sliceBias };
      };
      const current = geometry(getEngine().model),
        previous = geometry(getEngine().previousModel);
      if (!current)
        return {
          kind: "classification",
          iteration: getEngine().iteration,
          hasPrevious: !!previous,
          hasBoundary: false,
          entries: rows.map((row, index) => ({
            index: index + 1,
            x: row.x,
            y: row.y,
            beforeDistance: null,
            afterDistance: null,
            change: null,
            selected: index === selectedRowIndex,
          })),
          message:
            "Con pesi ancora nulli non esiste una retta decisionale definita. Esegui un passo reale per iniziare a visualizzarla.",
        };
      const distance = (model, row) => {
        if (!model) return null;
        const score = model.weights.reduce(
          (sum, value, index) => sum + value * (row.x[index] || 0),
          model.bias,
        );
        return ((row.y === 1 ? 1 : -1) * score) / model.norm;
      };
      const entries = rows.map((row, index) => {
        const beforeDistance = distance(previous, row),
          afterDistance = distance(current, row);
        return {
          index: index + 1,
          x: row.x,
          y: row.y,
          beforeDistance,
          afterDistance,
          change:
            beforeDistance === null ? null : afterDistance - beforeDistance,
          predicted: enginePredict(row.x),
          selected: index === selectedRowIndex,
        };
      });
      return {
        kind: "classification",
        iteration: getEngine().iteration,
        hasPrevious: !!previous,
        hasBoundary: true,
        boundary: {
          before: previous
            ? {
                x: previous.weights[0],
                y: previous.weights[1],
                intercept: previous.intercept,
              }
            : null,
          after: {
            x: current.weights[0],
            y: current.weights[1],
            intercept: current.intercept,
          },
        },
        entries,
        message:
          rows[0]?.x.length > 2
            ? "La retta è una sezione 2D con le altre feature fissate alla loro media; le distanze numeriche sono calcolate nello spazio completo delle feature."
            : "Le distanze firmate sono perpendicolari alla retta e positive sul lato della classe 1.",
      };
    }
    const selectedIndex = Math.max(
        0,
        Math.min(rows.length - 1, Math.round(Number(selectedSampleIndex) || 0)),
      ),
      selected = rows[selectedIndex],
      dimensions = selected?.x.length || 2,
      means = Array.from(
        { length: dimensions },
        (_, index) =>
          rows.reduce((sum, row) => sum + (row.x[index] || 0), 0) /
          (rows.length || 1),
      ),
      xValues = rows.map((row) => row.x[0] || 0),
      yValues = rows.map((row) => row.x[1] || 0),
      xMin = Math.min(...xValues),
      xMax = Math.max(...xValues),
      yMin = Math.min(...yValues),
      yMax = Math.max(...yValues),
      xMargin = (xMax - xMin || 1) * 0.12,
      yMargin = (yMax - yMin || 1) * 0.12,
      gridSize = 24,
      surface = [];
    for (let iy = 0; iy < gridSize; iy++) {
      const y =
          yMin -
          yMargin +
          ((yMax - yMin + 2 * yMargin) * (iy + 0.5)) / gridSize,
        line = [];
      for (let ix = 0; ix < gridSize; ix++) {
        const x =
            xMin -
            xMargin +
            ((xMax - xMin + 2 * xMargin) * (ix + 0.5)) / gridSize,
          features = means.slice();
        features[0] = x;
        if (features.length > 1) features[1] = y;
        line.push(enginePredict(features));
      }
      surface.push(line);
    }
    const points = rows.map((row, index) => ({
      index,
      x: row.x,
      y: row.y,
      prediction: enginePredict(row.x),
      selected: index === selectedIndex,
    }));
    const details = [];
    if (id === "knn") {
      const metric = param("metric", "euclidean"),
        k = Math.min(rows.length, Math.round(pNum("k", 5, 1, rows.length))),
        neighbors = rows
          .map((row, index) => ({
            index,
            distance: vecDist(selected.x, row.x, metric),
            label: row.y,
          }))
          .filter((row) => row.index !== selectedIndex)
          .sort((a, b) => a.distance - b.distance)
          .slice(0, k);
      details.push({
        title: `I ${k} vicini più prossimi · ${metric}`,
        rows: neighbors.map((neighbor) => ({
          label: `campione ${neighbor.index + 1} · classe ${neighbor.label}`,
          value: neighbor.distance,
          kind: "distance",
        })),
      });
    } else if (id === "svmrbf" || id === "svmpoly") {
      const support = rows
        .map((row, index) => ({
          index,
          alpha: Math.max(...m.alpha.map((values) => values[index] || 0)),
          similarity: svmKernel(row.x, selected.x, id),
        }))
        .filter((item) => item.alpha > 1e-8)
        .sort((a, b) => b.alpha - a.alpha)
        .slice(0, 32);
      details.push({
        title:
          "Support vector attivi · α e similarità kernel col campione selezionato",
        rows: support.map((item) => ({
          label: `campione ${item.index + 1} · α ${item.alpha.toFixed(4)}`,
          value: item.similarity,
          kind: "similarity",
        })),
      });
    } else if (id === "tree") {
      let node = m.root,
        path = [];
      while (node && !node.leaf) {
        const value = selected.x[node.feature] || 0,
          left = value <= node.threshold;
        path.push({
          label: `x${node.feature + 1} = ${value.toFixed(3)} ${left ? "≤" : ">"} soglia ${node.threshold.toFixed(3)}`,
          value: node.threshold,
          kind: "split",
        });
        node = left ? node.left : node.right;
      }
      details.push({
        title: "Percorso del campione nell’albero",
        rows: path.length
          ? path.map((item, index) => ({
              label: `${index + 1}. ${item.label}`,
              value: item.value,
              kind: "split",
            }))
          : [
              {
                label: "Radice foglia · nessuno split appreso",
                value: selected.y,
                kind: "split",
              },
            ],
      });
    } else if (id === "forest") {
      const votes = Array(Math.max(1, classes)).fill(0);
      m.trees.forEach((tree) => votes[treeClassify(tree, selected.x)]++);
      details.push({
        title: "Voti degli alberi per il campione selezionato",
        rows: votes.map((value, index) => ({
          label: `classe ${index}`,
          value,
          kind: "vote",
        })),
      });
    } else if (id === "boost") {
      const scores = m.regression
        ? [
            m.base[0] +
              m.trees[0].reduce(
                (sum, tree) =>
                  sum +
                  pNum("learningRate", 0.1, 1e-4, 1) *
                    treeValue(tree, selected.x),
                0,
              ),
          ]
        : boostProb(m, selected.x);
      details.push({
        title: m.regression
          ? "Contributi degli alberi alla previsione"
          : "Probabilità per classe",
        rows: m.regression
          ? [
              {
                label: "Valore di base",
                value: m.base[0],
                kind: "contribution",
              },
              ...m.trees[0].map((tree, index) => ({
                label: `albero ${index + 1}`,
                value:
                  pNum("learningRate", 0.1, 1e-4, 1) *
                  treeValue(tree, selected.x),
                kind: "contribution",
              })),
            ]
          : scores.map((value, index) => ({
              label: `classe ${index}`,
              value,
              kind: "probability",
            })),
      });
    } else if (id === "naive") {
      const scores = m.stats.map((stats) =>
          naiveScore(selected.x, stats, m.variant),
        ),
        probabilities = softmax(scores);
      details.push({
        title: "Punteggio e probabilità per classe",
        rows: scores.map((value, index) => ({
          label: `classe ${index} · score ${value.toFixed(4)}`,
          value: probabilities[index],
          kind: "probability",
        })),
      });
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
      const forward = mlpForward(m, selected.x);
      details.push({
        title: "Attivazioni della rete sul campione selezionato",
        rows: [
          ...forward.a.map((value, index) => ({
            label: `input x${index + 1}`,
            value,
            kind: "activation",
          })),
          ...forward.h.map((value, index) => ({
            label: `hidden h${index + 1} · tanh`,
            value,
            kind: "activation",
          })),
          ...forward.out.map((value, index) => ({
            label: `output ŷ${index + 1}`,
            value,
            kind: "activation",
          })),
        ],
      });
    } else if (id === "kmeans") {
      details.push({
        title: "Distanza del campione dai centroidi",
        rows: m.centroids.map((centroid, index) => ({
          label: `centroide ${index + 1} · ${enginePredict(selected.x) === index ? "assegnato" : ""}`,
          value: vecDist(selected.x, centroid),
          kind: "distance",
        })),
      });
    } else if (id === "dbscan") {
      const label = m.labels[selectedIndex],
        neighbors = rows
          .map((row, index) => ({
            index,
            distance: vecDist(selected.x, row.x, param("metric", "euclidean")),
            label: m.labels[index],
          }))
          .filter(
            (item) =>
              item.index !== selectedIndex &&
              item.distance <= pNum("eps", 0.5, 1e-4, 100),
          )
          .sort((a, b) => a.distance - b.distance)
          .slice(0, 32);
      details.push({
        title: `Vicini entro ε · punto ${label === -1 ? "rumore" : `cluster ${label}`} · ε=${pNum("eps", 0.5, 1e-4, 100).toFixed(3)}`,
        rows: neighbors.map((item) => ({
          label: `campione ${item.index + 1} · ${item.label < 0 ? "rumore" : `cluster ${item.label}`}`,
          value: item.distance,
          kind: "distance",
        })),
      });
    } else if (id === "gmm") {
      const scores = m.means.map(
          (mean, index) =>
            m.mix[index] * gaussianPdf(selected.x, mean, m.covs[index]),
        ),
        total = scores.reduce((sum, value) => sum + value, 0),
        probabilities = scores.map((value) => value / (total || 1));
      details.push({
        title: "Componenti gaussiane e responsabilità del campione",
        rows: m.means.map((mean, index) => ({
          label: `componente ${index + 1} · π=${m.mix[index].toFixed(3)} · μ=[${mean
            .slice(0, 2)
            .map((value) => value.toFixed(2))
            .join(", ")}]`,
          value: probabilities[index],
          kind: "probability",
        })),
      });
    }
    const path = m.kind === "tree" ? details[0]?.rows || [] : [],
      centroids =
        id === "kmeans"
          ? m.centroids.map((center, index) => ({
              index,
              x: center[0] || 0,
              y: center[1] || 0,
            }))
          : id === "gmm"
            ? m.means.map((center, index) => ({
                index,
                x: center[0] || 0,
                y: center[1] || 0,
              }))
            : [],
      supportVectors =
        id === "svmrbf" || id === "svmpoly"
          ? rows
              .map((row, index) => ({
                index,
                alpha: Math.max(...m.alpha.map((values) => values[index] || 0)),
              }))
              .filter((item) => item.alpha > 1e-8)
          : [];
    return {
      kind:
        ENGINE_REGRESSION.has(id) || (id === "boost" && m.regression)
          ? "model-regression"
          : ENGINE_CLUSTER.has(id)
            ? "model-clustering"
            : "model-classification",
      algorithm: id,
      iteration: getEngine().iteration,
      selectedIndex,
      selected: {
        index: selectedIndex,
        x: selected.x,
        y: selected.y,
        prediction: enginePredict(selected.x),
      },
      dimensions,
      means,
      surface,
      domain: {
        xMin: xMin - xMargin,
        xMax: xMax + xMargin,
        yMin: yMin - yMargin,
        yMax: yMax + yMargin,
      },
      points,
      details,
      path,
      centroids,
      supportVectors,
      message:
        dimensions > 2
          ? "La mappa mostra la sezione sulle feature x₁ e x₂; le altre feature sono fissate alle medie del training. Le distanze e le decisioni del punto selezionato usano tutte le feature."
          : "Sfondo = risposta prevista dal modello; punti = dati osservati. Seleziona un campione dal controllo o dalla tabella per ispezionarlo.",
    };
  }
  /** Expose current and previous activations and weights for the neural-network drawing. */
  function getNeuralNetworkLearningVisualization() {
    const id = methodId(),
      neuralMethods = [
        "mlp",
        "sgd",
        "momentum",
        "nesterov",
        "adagrad",
        "rmsprop",
        "adam",
        "adamw",
      ];
    if (!neuralMethods.includes(id)) return null;
    if (getEngine().id !== id || !getEngine().model) engineInit();
    const model = getEngine().model;
    if (model?.kind !== "mlp") return null;
    const x = (getEngine().train[0] || data[0])?.x || Array(model.d).fill(0),
      forward = mlpForward(model, x),
      previous =
        getEngine().previousModel?.kind === "mlp"
          ? getEngine().previousModel
          : null,
      previousForward = previous ? mlpForward(previous, x) : null;
    return {
      iteration: getEngine().iteration,
      input: x.slice(),
      inputValues: forward.a,
      hiddenValues: forward.h,
      outputValues: forward.out,
      previousInputValues: previousForward?.a || null,
      previousHiddenValues: previousForward?.h || null,
      previousOutputValues: previousForward?.out || null,
      weightsInputHidden: model.W1,
      weightsHiddenOutput: model.W2,
      previousWeightsInputHidden: previous?.W1 || null,
      previousWeightsHiddenOutput: previous?.W2 || null,
    };
  }
  return {
    getAlgorithmLearningTrace,
    getAlgorithmLearningVisualization,
    getNeuralNetworkLearningVisualization,
  };
}
