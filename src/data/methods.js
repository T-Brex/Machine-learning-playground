/** Algorithm formulas, glossary, and editable parameter schemas. */
const METHOD_INFO = {
  logistic: {
    status:
      "Implementato per classificazione binaria; per più classi usa Softmax.",
    model: "p(y=1|x)=σ((w₁−w₀)ᵀx+b₁−b₀)",
    cost: "J = −(1/m)Σᵢ[yᵢlog(pᵢ)+(1−yᵢ)log(1−pᵢ)] + regolarizzazione",
    update: "θ ← θ − α∇J<br>θ = (w₁−w₀, b₁−b₀)",
    explain:
      "Stima la probabilità di una delle due classi con la funzione sigmoide. Per classificare più di due classi seleziona Softmax. Metriche: accuracy, precision, recall, F1 e matrice di confusione.",
  },
  softmax: {
    status: "Implementato nel motore come classificatore lineare multiclasse.",
    model: "p(y=k|x)= exp(wₖᵀx+bₖ) / Σⱼ exp(wⱼᵀx+bⱼ)",
    cost: "J = −(1/m)ΣᵢΣₖ yᵢₖ log(pᵢₖ) + (λ/2)‖W‖₂²",
    update: "W ← W − α[(1/m)Σᵢ(pᵢ−yᵢ)xᵢᵀ + λW]<br>b ← b − α(1/m)Σᵢ(pᵢ−yᵢ)",
    explain:
      "Un vettore di probabilità per ogni classe. Metriche: accuracy, macro/micro precision, recall, F1 e confusion matrix.",
  },
  perceptron: {
    status:
      "Implementato nel motore per classificazione binaria e multiclasse.",
    model: "ŷ = argmaxₖ(wₖᵀx+bₖ)",
    cost: "Aggiorna solo se ŷᵢ ≠ yᵢ",
    update: "wᵧ ← wᵧ + αx ; wŷ ← wŷ − αx<br>bᵧ ← bᵧ + α ; bŷ ← bŷ − α",
    explain:
      "Aggiorna i parametri quando un esempio è classificato male. Metriche: accuracy, precision, recall, F1.",
  },
  linear: {
    status: "Regressione lineare con feature originali o trasformate.",
    model: "ŷ = wᵀφ(x)+b",
    cost: "J = MSE/2 = (1/2m)Σᵢ(ŷᵢ−yᵢ)²",
    update: "w ← w − α(1/m)Σᵢeᵢφ(xᵢ)<br>b ← b − α(1/m)Σᵢeᵢ",
    explain:
      "Il grado automatico usa una base lineare, quadratica o polinomiale in base al dataset; puoi impostarlo manualmente. Metriche: MSE, RMSE, MAE e R².",
  },
  ridge: {
    status:
      "Ridge Regression implementata con feature originali o trasformate.",
    model: "ŷ = wᵀφ(x)+b",
    cost: "J = MSE/2 + λ‖w‖₂²/2",
    update: "w ← w − α[(1/m)Σᵢeᵢφ(xᵢ)+λw]<br>b ← b − α(1/m)Σᵢeᵢ",
    explain:
      "Regolarizzazione L2 sulla base selezionata: riduce i coefficienti senza azzerarli in generale. Puoi scegliere il grado delle feature.",
  },
  lasso: {
    status:
      "Lasso implementato con aggiornamento prossimale e feature trasformate.",
    model: "ŷ = wᵀφ(x)+b",
    cost: "J = MSE/2 + λ‖w‖₁",
    update:
      "u ← w − α∇(MSE/2)<br>w ← sign(u)·max(0,|u|−αλ)<br>b ← b − α∂(MSE/2)/∂b",
    explain:
      "Regolarizzazione L1 sulla base selezionata: può azzerare coefficienti e selezionare termini polinomiali. Puoi scegliere il grado delle feature.",
  },
  elastic: {
    status: "Elastic Net implementato con penalità L1 e L2.",
    model: "ŷ = wᵀφ(x)+b",
    cost: "J = MSE/2 + λ[ρ‖w‖₁ + (1−ρ)‖w‖₂²/2]",
    update:
      "w ← w − α[∇(MSE/2)+λ(1−ρ)w]<br>w ← proxₐλρ(w) ; b ← b − α∂(MSE/2)/∂b",
    explain:
      "Combina selezione delle feature (L1) e shrinkage (L2) sulla base scelta. Il grado automatico si adatta ai dataset sintetici; puoi modificarlo.",
  },
  knn: {
    status:
      "Implementato nel motore base; classificazione multiclasse e distanza sulle feature supportate.",
    model: "ŷ(x)= moda{yᵢ : xᵢ ∈ Nₖ(x)}",
    cost: "Nessuna funzione di costo ottimizzata: KNN è un metodo instance-based.",
    update:
      "Nessun aggiornamento dei pesi. Si memorizzano gli esempi; la previsione usa i k vicini.",
    explain:
      "Metriche: accuracy, precision, recall, F1, confusion matrix. Parametri: k, metrica e pesatura dei vicini.",
  },
  svmrbf: {
    status:
      "SVM con kernel RBF implementata tramite aggiornamento duale semplificato.",
    model: "K(x,z)=exp(−γ||x−z||²)",
    cost: "Hinge loss sul punteggio kernel<br>Lᵢ=max(0,1−yᵢf(xᵢ))",
    update:
      "αᵢ ← clip((1−yᵢ(otherᵢ+b))/Kᵢᵢ, 0, C)<br>b ← media(yᵢ−f(xᵢ)) sui support vector liberi",
    explain:
      "Kernel RBF produce frontiere non lineari. Parametri: C, γ, tolleranza, criterio multiclasse. Metriche di classificazione.",
  },
  svmpoly: {
    status:
      "SVM con kernel polinomiale implementata tramite aggiornamento duale semplificato.",
    model: "K(x,z)=(γxᵀz+r)ᵈ",
    cost: "Hinge loss sul punteggio kernel<br>Lᵢ=max(0,1−yᵢf(xᵢ))",
    update:
      "αᵢ ← clip((1−yᵢ(otherᵢ+b))/Kᵢᵢ, 0, C)<br>b ← media(yᵢ−f(xᵢ)) sui support vector liberi",
    explain:
      "Kernel polinomiale. Parametri: C, grado d, γ, coef0. Metriche di classificazione.",
  },
  tree: {
    status:
      "Decision Tree implementato con ricerca di split e criteri Gini/entropia.",
    model: "Nodo: split feature j e soglia t; foglia: classe maggioritaria",
    cost: "Gini = 1−Σₖpₖ²<br>oppure Entropy = −Σₖpₖ log₂pₖ",
    update:
      "Sceglie lo split con impurità pesata minima; ricorsione fino ai limiti impostati.",
    explain:
      "Parametri: profondità massima, min campioni split/foglia, criterio. Metriche: classificazione.",
  },
  forest: {
    status:
      "Random Forest implementata con bootstrap e selezione casuale delle feature.",
    model: "ŷ = moda{T₁(x),…,Tₙ(x)}",
    cost: "Impurità Gini/Entropy aggregata sugli split degli alberi",
    update:
      "Addestra alberi su campioni bootstrap e sottoinsiemi casuali di feature; aggrega per voto.",
    explain:
      "Parametri: numero alberi, profondità, max feature, bootstrap, min samples leaf. Metriche: classificazione e OOB se disponibile.",
  },
  boost: {
    status: "Gradient Boosting implementato per regressione e classificazione.",
    model: "Fₘ(x)=Fₘ₋₁(x)+ηhₘ(x)",
    cost: "L(y,F(x)) (es. errore quadratico o log-loss)",
    update:
      "Addestra hₘ sui residui/gradienti negativi correnti e aggiunge ηhₘ alla previsione.",
    explain:
      "Parametri: numero stimatori, learning rate, profondità/alberi, subsampling. Metriche: loss, accuracy, F1.",
  },
  naive: {
    status:
      "Naive Bayes implementato con varianti gaussiana, multinomiale e Bernoulli.",
    model: "P(y|x) ∝ P(y) ΠⱼP(xⱼ|y)",
    cost: "Negative log-likelihood dei parametri stimati dai dati",
    update:
      "Stima prior e distribuzioni condizionate dai campioni; non usa gradient descent standard.",
    explain:
      "Parametri: smoothing α, modello gaussiano/multinomiale/bernoulli. Metriche: classificazione.",
  },
  mlp: {
    status:
      "Rete neurale implementata con backpropagation e ottimizzatore Adam predefinito.",
    model: "a⁽ˡ⁾=φ(W⁽ˡ⁾a⁽ˡ⁻¹⁾+b⁽ˡ⁾)",
    cost: "Cross-entropy per classificazione; MSE/2 per regressione",
    update:
      "mₜ=β₁mₜ₋₁+(1−β₁)gₜ<br>vₜ=β₂vₜ₋₁+(1−β₂)gₜ²<br>W⁽ˡ⁾, b⁽ˡ⁾ ← Adam(W⁽ˡ⁾, b⁽ˡ⁾, gₜ)",
    explain:
      "La backpropagation calcola i gradienti di pesi e bias; Adam è l’ottimizzatore predefinito. Metriche: loss, accuracy o MSE e confusion matrix.",
  },
  sgd: {
    status: "SGD implementato come ottimizzatore della rete neurale.",
    model: "gₜ=∇θJᵢ(θₜ)",
    cost: "Jᵢ: loss del campione o mini-batch selezionato",
    update: "θₜ₊₁=θₜ−αgₜ",
    explain:
      "Parametri: learning rate, batch size, shuffle. Ottimizza il modello neurale selezionato.",
  },
  momentum: {
    status: "Momentum implementato come ottimizzatore della rete neurale.",
    model: "gₜ=∇θJ(θₜ)",
    cost: "J(θ): loss del batch corrente",
    update: "vₜ=μvₜ₋₁+gₜ<br>θₜ₊₁=θₜ−αvₜ",
    explain: "Parametri: learning rate α, momentum μ, batch size.",
  },
  nesterov: {
    status: "Nesterov implementato con gradiente nella posizione anticipata.",
    model: "gₜ=∇θJ(θₜ+vₜ₋₁)",
    cost: "J(θ): loss del batch corrente",
    update: "vₜ=μvₜ₋₁−αgₜ<br>θₜ₊₁=θₜ+vₜ",
    explain:
      "Valuta il gradiente al look-ahead θ+v e aggiorna la velocità. Parametri: α, μ.",
  },
  adagrad: {
    status: "AdaGrad implementato come ottimizzatore della rete neurale.",
    model: "gₜ=∇θJ(θₜ)",
    cost: "J(θ): loss del batch corrente",
    update: "rₜ=rₜ₋₁+gₜ⊙gₜ<br>θₜ₊₁=θₜ−αgₜ/(√rₜ+ε)",
    explain: "Learning rate adattivo per parametro. Parametri: α, epsilon.",
  },
  rmsprop: {
    status: "RMSProp implementato come ottimizzatore della rete neurale.",
    model: "gₜ=∇θJ(θₜ)",
    cost: "J(θ): loss del batch corrente",
    update: "rₜ=ρrₜ₋₁+(1−ρ)gₜ²<br>θₜ₊₁=θₜ−αgₜ/(√rₜ+ε)",
    explain: "Parametri: α, decay ρ, epsilon.",
  },
  adam: {
    status: "Adam implementato come ottimizzatore della rete neurale.",
    model: "gₜ=∇θJ(θₜ)",
    cost: "J(θ): loss del batch corrente",
    update: "mₜ=β₁mₜ₋₁+(1−β₁)gₜ<br>vₜ=β₂vₜ₋₁+(1−β₂)gₜ²<br>θₜ₊₁=θₜ−αm̂ₜ/(√v̂ₜ+ε)",
    explain:
      "Parametri: α, β₁, β₂, epsilon. m̂ e v̂ sono corretti per il bias iniziale.",
  },
  adamw: {
    status: "AdamW implementato con weight decay disaccoppiato.",
    model: "gₜ=∇θJ(θₜ)",
    cost: "J(θ): loss del batch corrente; il weight decay è separato",
    update: "Adam: θ′=θ−αm̂ₜ/(√v̂ₜ+ε)<br>θₜ₊₁=θ′−αλθₜ",
    explain: "Weight decay disaccoppiato. Parametri: α, β₁, β₂, epsilon, λ.",
  },
  kmeans: {
    status: "Implementato nel motore base come clustering.",
    model: "assegna xᵢ al centro μₖ più vicino",
    cost: "Inertia = Σᵢ minₖ ||xᵢ−μₖ||²",
    update: "Assegnazione: cᵢ=argminₖ||xᵢ−μₖ||²<br>Centro: μₖ=media{xᵢ:cᵢ=k}",
    explain:
      "Parametri: numero cluster k, inizializzazione, iterazioni. Metriche: inertia e silhouette (se calcolata).",
  },
  dbscan: {
    status: "DBSCAN implementato con espansione dei cluster per densità.",
    model: "Core point se |Nε(x)| ≥ minPts; cluster tramite densità",
    cost: "Non minimizza una funzione di costo globale standard",
    update:
      "Conta i vicini entro ε; espande i punti raggiungibili e identifica il rumore.",
    explain:
      "Parametri: epsilon, minPts, metrica. Metriche: silhouette sui punti assegnati, numero cluster e rumore.",
  },
  gmm: {
    status: "Gaussian Mixture Model implementato con algoritmo EM.",
    model: "p(x)=Σₖπₖ N(x|μₖ,Σₖ)",
    cost: "Negative log-likelihood: −Σᵢ log ΣₖπₖN(xᵢ|μₖ,Σₖ)",
    update: "EM: E-step calcola responsabilità; M-step aggiorna πₖ, μₖ, Σₖ.",
    explain:
      "Parametri: componenti, covarianza, regolarizzazione, iterazioni. Metriche: log-likelihood, AIC/BIC.",
  },
};
const METHOD_SYMBOLS = {
  logistic: [
    ["x, xᵢ", "Vettore delle feature di un campione; è l’input del modello."],
    [
      "y, yᵢ",
      "Indice della classe corretta (o target one-hot) usato nel calcolo della cross-entropy.",
    ],
    ["W, wₖ", "Matrice dei pesi: una riga di coefficienti per ogni classe."],
    ["b, bₖ", "Bias per classe, sommato al punteggio lineare."],
    ["zₖ", "Punteggio lineare wₖᵀx+bₖ prima della normalizzazione."],
    [
      "softmax",
      "Trasforma i punteggi in probabilità che sommano a 1; con due classi equivale a una sigmoide sulla differenza dei punteggi.",
    ],
    ["p(y=k|x)", "Probabilità stimata della classe k."],
    ["J", "Cross-entropy media, con eventuale penalità L2 impostata."],
    ["m", "Numero di esempi mediati nel batch."],
    ["α", "Learning rate: ampiezza dei passi di aggiornamento."],
    ["i, j, k", "Indici di esempio, feature e classe."],
    ["log", "Logaritmo naturale nella cross-entropy."],
    [
      "∇, ∂",
      "Gradiente rispetto a W e b; indica come aggiornare i parametri per ridurre la loss.",
    ],
  ],
  softmax: [
    ["x", "Vettore delle feature in ingresso."],
    [
      "yᵢₖ",
      "Etichetta one-hot: vale 1 per la classe corretta k del campione i, altrimenti 0.",
    ],
    ["p(y=k|x), pᵢₖ", "Probabilità stimata che x appartenga alla classe k."],
    ["W, wₖ", "Pesi del modello: una riga/vettore per ciascuna classe."],
    ["b, bₖ", "Bias della classe k; regola il suo punteggio di base."],
    [
      "exp",
      "Esponenziale; trasforma i punteggi in valori positivi prima della normalizzazione.",
    ],
    [
      "Σⱼ",
      "Somma sui punteggi di tutte le classi; normalizza le probabilità affinché sommino a 1.",
    ],
    ["J", "Cross-entropy media tra classi vere e probabilità predette."],
    [
      "m, i, k, j",
      "Numero di campioni; indici di campione, classe valutata e classe sommata.",
    ],
    ["α", "Learning rate dell’aggiornamento."],
    ["∇", "Gradiente della loss rispetto ai pesi o ai bias."],
  ],
  perceptron: [
    ["x, xᵢ", "Vettore delle feature del campione."],
    ["yᵢ", "Indice della classe vera."],
    ["wₖ, bₖ", "Pesi e bias del punteggio della classe k."],
    ["zₖ", "Punteggio lineare wₖᵀx+bₖ."],
    ["ŷ", "Classe con il punteggio più alto."],
    [
      "α",
      "Learning rate usato nell’aggiornamento della classe vera e di quella predetta.",
    ],
    ["i, k", "Indici di campione e classe."],
  ],
  linear: [
    ["x", "Feature originali del campione."],
    [
      "φ(x)",
      "Trasformazione delle feature (per esempio termini quadratici); consente di modellare relazioni non lineari nelle feature.",
    ],
    ["w", "Coefficienti appresi, uno per feature trasformata."],
    ["b", "Bias/intercetta della previsione."],
    ["ŷ", "Valore continuo previsto dal modello."],
    ["yᵢ", "Target osservato del campione i."],
    [
      "MSE",
      "Errore quadratico medio: penalizza gli scarti tra previsione e target.",
    ],
    ["m, i", "Numero di campioni e indice del campione nella media."],
    ["α", "Learning rate per l’aggiornamento dei coefficienti."],
    [
      "∇, ∂",
      "Derivata/gradiente della MSE; indica la direzione di riduzione dell’errore.",
    ],
  ],
  ridge: [
    [
      "x, φ(x)",
      "Feature in ingresso e loro eventuale trasformazione polinomiale.",
    ],
    ["w", "Coefficienti del modello; la penalità L2 ne limita i valori."],
    ["b", "Intercetta, non penalizzata in questa formula."],
    ["ŷ, yᵢ", "Target previsto e osservato; la differenza misura l’errore."],
    ["MSE", "Errore quadratico medio sui dati."],
    [
      "λ",
      "Intensità della regolarizzazione: valori maggiori riducono maggiormente i pesi.",
    ],
    ["‖w‖₂²", "Somma dei quadrati dei coefficienti; penalizza pesi grandi."],
    ["α", "Learning rate."],
    ["∇, ∂", "Gradiente della loss rispetto a pesi e bias."],
  ],
  lasso: [
    ["x, φ(x)", "Feature in ingresso e loro trasformazione."],
    [
      "w",
      "Coefficienti da apprendere; la penalità può portarne alcuni esattamente a zero.",
    ],
    ["b", "Intercetta, normalmente non penalizzata."],
    ["ŷ, yᵢ", "Previsione continua e target osservato."],
    ["MSE", "Errore quadratico medio."],
    [
      "λ",
      "Intensità della penalità di sparsità; controlla la selezione delle feature.",
    ],
    [
      "‖w‖₁",
      "Somma dei valori assoluti dei coefficienti; incoraggia coefficienti nulli.",
    ],
    [
      "proximal-gradient",
      "Passo di ottimizzazione che combina il gradiente con la soglia L1, capace di azzerare pesi.",
    ],
    ["α, ∇", "Learning rate e gradiente dell’errore."],
  ],
  elastic: [
    ["x, φ(x)", "Feature di ingresso, eventualmente trasformate."],
    ["w, b", "Coefficienti e intercetta del modello."],
    ["ŷ, yᵢ, MSE", "Previsione, target osservato ed errore quadratico medio."],
    ["λ", "Intensità complessiva della regolarizzazione."],
    ["ρ", "Rapporto tra penalità L1 e L2: 1 usa solo L1, 0 solo L2."],
    ["‖w‖₁", "Penalità L1: favorisce coefficienti nulli e seleziona feature."],
    [
      "‖w‖₂²",
      "Penalità L2: riduce i coefficienti senza azzerarli necessariamente.",
    ],
    [
      "sign(w)",
      "Segno di ciascun peso, usato nella derivata della penalità L1.",
    ],
    ["α, ∇", "Learning rate e gradiente dell’errore."],
  ],
  knn: [
    ["x", "Nuovo campione da classificare."],
    ["xᵢ, yᵢ", "Feature e classe di ciascun esempio già memorizzato."],
    ["Nₖ(x)", "Insieme dei k esempi più vicini a x secondo la metrica scelta."],
    ["k", "Numero di vicini che partecipano al voto."],
    ["ŷ(x)", "Classe assegnata al nuovo campione."],
    [
      "moda",
      "Classe più frequente tra i vicini (o con maggior peso se si selezionano pesi per distanza).",
    ],
  ],
  svmrbf: [
    [
      "K(x,z)",
      "Kernel RBF: misura la somiglianza tra x e z e permette un confine non lineare.",
    ],
    ["x, z", "Due vettori di feature confrontati dal kernel."],
    [
      "γ",
      "Gamma: controlla quanto rapidamente diminuisce la somiglianza con la distanza; influenza la curvatura del confine.",
    ],
    ["‖x−z‖²", "Distanza euclidea al quadrato tra i due campioni."],
    [
      "w, b",
      "Pesi e bias nello spazio trasformato dal kernel; definiscono il confine.",
    ],
    ["C", "Compromesso tra margine ampio e penalizzazione degli errori."],
    [
      "ξᵢ",
      "Variabile di slack: misura la violazione del margine per il campione i.",
    ],
    [
      "yᵢ",
      "Etichetta della classe, nella formula SVM codificata come −1 o +1.",
    ],
    [
      "φ(x)",
      "Rappresentazione implicita nello spazio in cui si separano i dati.",
    ],
    [
      "αᵢ",
      "Moltiplicatore duale associato al campione; identifica e pesa i support vector.",
    ],
    ["f(xᵢ)", "Punteggio decisionale del modello sul campione."],
    ["η", "Passo dell’aggiornamento iterativo duale."],
    [
      "clip[·,0,C]",
      "Limita il moltiplicatore duale all’intervallo consentito.",
    ],
  ],
  svmpoly: [
    [
      "K(x,z)",
      "Kernel polinomiale: calcola una somiglianza polinomiale tra due campioni.",
    ],
    ["x, z", "Vettori di feature confrontati."],
    ["γ", "Fattore che scala il prodotto scalare."],
    [
      "r (coef0)",
      "Costante del kernel; influenza i termini di grado inferiore.",
    ],
    ["d", "Grado del polinomio; controlla la complessità delle interazioni."],
    [
      "w, b, φ(x)",
      "Pesi, bias e rappresentazione nello spazio indotto dal kernel.",
    ],
    ["C", "Compromesso tra margine e penalizzazione delle violazioni."],
    ["ξᵢ", "Violazione del margine consentita per il campione i."],
    ["yᵢ", "Etichetta binaria codificata come −1 o +1 nella formula SVM."],
    [
      "αᵢ",
      "Moltiplicatore duale del campione; pesi non nulli indicano support vector.",
    ],
    ["0≤αᵢ≤C", "Vincolo che limita ogni moltiplicatore duale."],
  ],
  tree: [
    [
      "j, t",
      "Feature scelta per lo split e soglia: separano i dati in due rami.",
    ],
    ["pₖ", "Frazione di campioni della classe k nel nodo."],
    ["k", "Indice della classe considerata."],
    [
      "Gini",
      "Impurità del nodo: è zero se contiene una sola classe; lo split cerca di ridurla.",
    ],
    [
      "Entropy",
      "Impurità basata sulla distribuzione delle classi; misura l’incertezza del nodo.",
    ],
    ["Σₖ", "Somma sui gruppi/classi presenti nel nodo."],
    ["log₂", "Logaritmo in base 2 usato per calcolare l’entropia."],
    ["ŷ", "Classe maggioritaria nella foglia raggiunta dal campione."],
  ],
  forest: [
    ["x", "Campione da classificare."],
    ["T₁,…,Tₙ", "Alberi decisionali addestrati su campioni e feature casuali."],
    ["n", "Numero di alberi nell’ensemble."],
    ["Tᵢ(x)", "Classe predetta dall’albero i sul campione x."],
    [
      "moda",
      "Voto di maggioranza: seleziona la classe predetta dal maggior numero di alberi.",
    ],
    [
      "Gini/Entropy",
      "Misure d’impurità usate per scegliere gli split nei singoli alberi.",
    ],
    [
      "bootstrap",
      "Campionamento con reinserimento usato per creare dataset diversi per gli alberi.",
    ],
  ],
  boost: [
    ["Fₘ(x)", "Predizione complessiva dopo m stadi di boosting."],
    [
      "Fₘ₋₁(x)",
      "Predizione dell’ensemble prima di aggiungere il nuovo stadio.",
    ],
    ["η", "Learning rate: ridimensiona il contributo di ogni nuovo albero."],
    [
      "hₘ(x)",
      "Albero/stimatore aggiunto allo stadio m per correggere gli errori correnti.",
    ],
    [
      "L(y,F(x))",
      "Funzione loss che misura lo scarto tra target y e predizione F(x).",
    ],
    ["y", "Classe o target osservato, secondo il tipo di problema."],
    ["m", "Indice dello stadio/numero di alberi aggiunti."],
  ],
  naive: [
    [
      "P(·)",
      "Probabilità; quantifica quanto è plausibile un evento o una classe.",
    ],
    ["y", "Classe candidata."],
    ["xⱼ", "Valore della feature j del campione."],
    ["P(y|x)", "Probabilità della classe y date tutte le feature x."],
    ["P(y)", "Probabilità iniziale/prior della classe, stimata dai dati."],
    [
      "P(xⱼ|y)",
      "Probabilità della feature j condizionata alla classe; Naive Bayes assume indipendenza condizionata.",
    ],
    [
      "∝",
      "Proporzionale a: si omette il fattore comune a tutte le classi, che non cambia la classe vincente.",
    ],
    ["Πⱼ", "Prodotto delle probabilità delle feature, indice j."],
  ],
  mlp: [
    ["a⁽ˡ⁾", "Attivazioni/uscite dei neuroni nello strato l."],
    ["a⁽ˡ⁻¹⁾", "Attivazioni provenienti dallo strato precedente."],
    ["W⁽ˡ⁾", "Matrice dei pesi tra lo strato l−1 e lo strato l."],
    ["b⁽ˡ⁾", "Vettore dei bias dello strato l."],
    ["φ", "Funzione di attivazione; introduce non linearità nella rete."],
    [
      "z, ŷ",
      "Valore prima dell’attivazione e output/probabilità finale della rete.",
    ],
    [
      "yᵢ, yᵢₖ",
      "Target continuo nella regressione o etichetta one-hot nella classificazione.",
    ],
    ["J", "Cross-entropy media in classificazione; MSE/2 in regressione."],
    ["m, i, k", "Numero di esempi e indici di campione e classe."],
    ["α", "Learning rate."],
    [
      "θ, θ₀",
      "Vettore dei pesi e bias; θ₀ è il primo parametro mostrato nella traccia.",
    ],
    ["gₜ", "Gradiente dei parametri al passo t."],
    [
      "∂J/∂W, ∂J/∂b",
      "Gradienti della loss rispetto a pesi e bias, calcolati con backpropagation.",
    ],
  ],
  sgd: [
    ["θₜ", "Tutti i parametri del modello al passo t."],
    ["Lᵢ", "Loss del campione i (o del mini-batch selezionato)."],
    [
      "gₜ",
      "Gradiente della loss: direzione di aumento più rapido, quindi si sottrae per ridurla.",
    ],
    ["∇θ", "Derivata rispetto a tutti i parametri θ."],
    ["α", "Learning rate: dimensione del passo."],
    ["t", "Indice del passo/aggiornamento."],
  ],
  momentum: [
    ["θₜ", "Parametri del modello al passo t."],
    ["J(θₜ)", "Funzione obiettivo valutata con i parametri correnti."],
    ["gₜ", "Gradiente della loss al passo corrente."],
    [
      "vₜ",
      "Velocità/memoria degli aggiornamenti precedenti, che smorza oscillazioni e accelera direzioni coerenti.",
    ],
    ["μ", "Coefficiente momentum: quanto della velocità precedente mantenere."],
    ["α", "Learning rate."],
    ["t", "Indice del passo."],
  ],
  nesterov: [
    ["θₜ", "Parametri correnti del modello."],
    ["J", "Funzione obiettivo da minimizzare."],
    ["gₜ", "Gradiente calcolato in anticipo alla posizione θₜ−αμvₜ₋₁."],
    ["vₜ", "Velocità accumulata dagli aggiornamenti precedenti."],
    ["μ", "Coefficiente momentum."],
    ["α", "Learning rate e ampiezza del passo anticipato."],
    ["t", "Indice dell’aggiornamento."],
  ],
  adagrad: [
    ["θₜ", "Parametri del modello al passo t."],
    ["J(θ)", "Funzione obiettivo."],
    ["gₜ", "Gradiente corrente."],
    [
      "rₜ",
      "Somma cumulativa dei quadrati dei gradienti; riduce il passo per parametri aggiornati spesso.",
    ],
    ["⊙", "Prodotto elemento per elemento tra vettori."],
    ["√rₜ", "Scala adattiva calcolata separatamente per ciascun parametro."],
    ["α", "Learning rate iniziale."],
    ["ε", "Piccola costante per evitare divisioni per zero."],
    ["t", "Indice del passo."],
  ],
  rmsprop: [
    ["θₜ", "Parametri del modello."],
    ["J(θ)", "Funzione obiettivo."],
    ["gₜ", "Gradiente corrente."],
    [
      "rₜ",
      "Media mobile dei gradienti al quadrato; adatta il passo alla scala recente del gradiente.",
    ],
    ["ρ", "Fattore di decadimento della media mobile."],
    ["α", "Learning rate."],
    ["ε", "Piccola costante numerica per stabilizzare la divisione."],
    ["t", "Indice del passo."],
  ],
  adam: [
    ["θₜ", "Parametri del modello al passo t."],
    ["J(θ)", "Funzione obiettivo."],
    ["gₜ", "Gradiente corrente."],
    ["mₜ", "Media mobile dei gradienti: stima il primo momento/direzione."],
    [
      "vₜ",
      "Media mobile dei gradienti al quadrato: stima il secondo momento/scala.",
    ],
    ["β₁, β₂", "Fattori di decadimento delle due medie mobili."],
    ["m̂ₜ, v̂ₜ", "Stime corrette per il bias iniziale delle medie mobili."],
    ["α", "Learning rate."],
    ["ε", "Piccola costante che evita divisioni instabili."],
    ["t", "Indice del passo, usato anche per la correzione del bias."],
  ],
  adamw: [
    ["θₜ", "Parametri prima dell’aggiornamento."],
    [
      "θ′",
      "Parametri dopo l’aggiornamento Adam, prima del decadimento dei pesi.",
    ],
    ["J(θ), gₜ", "Funzione obiettivo e relativo gradiente corrente."],
    [
      "m̂ₜ, v̂ₜ",
      "Momenti Adam corretti per il bias iniziale; stimano direzione e scala del gradiente.",
    ],
    ["β₁, β₂", "Fattori di decadimento dei momenti Adam."],
    ["α", "Learning rate."],
    ["ε", "Piccola costante di stabilità numerica."],
    [
      "λ",
      "Weight decay: riduce direttamente i pesi, separatamente dal gradiente della loss.",
    ],
    ["t", "Indice del passo."],
  ],
  kmeans: [
    ["xᵢ", "Vettore delle feature del campione i."],
    [
      "μₖ",
      "Centroide del cluster k, cioè il centro rappresentativo dei suoi punti.",
    ],
    ["k", "Indice/numero del cluster candidato."],
    ["cᵢ", "Cluster assegnato al campione i."],
    ["argminₖ", "Sceglie il cluster con distanza minima dal campione."],
    ["‖xᵢ−μₖ‖²", "Distanza euclidea quadratica dal campione al centroide."],
    [
      "Inertia",
      "Somma delle distanze al quadrato dai centroidi assegnati; misura la compattezza dei cluster.",
    ],
    ["Σᵢ", "Somma su tutti i campioni."],
  ],
  dbscan: [
    ["x", "Punto da esaminare."],
    ["Nε(x)", "Insieme dei punti entro la distanza ε da x."],
    [
      "ε (epsilon)",
      "Raggio del vicinato che determina quali punti sono vicini.",
    ],
    [
      "minPts",
      "Numero minimo di punti nel vicinato per considerare x un punto core.",
    ],
    ["|Nε(x)|", "Numero di punti nel vicinato di x."],
    [
      "≥",
      "La condizione stabilisce se la densità locale è sufficiente per espandere un cluster.",
    ],
  ],
  gmm: [
    ["x, xᵢ", "Campione osservato e sue feature."],
    ["p(x)", "Probabilità/densità del campione secondo la miscela."],
    ["k, K", "Indice e numero totale delle componenti gaussiane."],
    ["πₖ", "Peso/probabilità a priori della componente k; i pesi sommano a 1."],
    ["N(x|μₖ,Σₖ)", "Densità gaussiana della componente k valutata in x."],
    ["μₖ", "Vettore media, cioè il centro della componente k."],
    [
      "Σₖ",
      "Matrice di covarianza; descrive dispersione e correlazioni delle feature.",
    ],
    [
      "log",
      "Logaritmo naturale della probabilità; rende additivi i contributi e penalizza assegnazioni improbabili.",
    ],
    ["−Σᵢ log p(xᵢ)", "Negative log-likelihood totale da minimizzare."],
    [
      "E-step / responsabilità",
      "Stima quanto ciascuna componente spiega ogni campione.",
    ],
    ["M-step", "Aggiorna pesi, medie e covarianze usando le responsabilità."],
  ],
};
METHOD_SYMBOLS.logistic.push(
  [
    "wᵀx",
    "Prodotto scalare tra pesi e feature; combina gli input in un solo punteggio.",
  ],
  [
    "Σ",
    "Somma dei contributi dei campioni per ottenere la loss/il gradiente medio.",
  ],
);
METHOD_SYMBOLS.softmax.push(
  [
    "wₖᵀx",
    "Prodotto scalare tra i pesi della classe k e le feature; produce il punteggio di quella classe.",
  ],
  [
    "Σ",
    "Somma di normalizzazione sulle classi o somma delle loss sui campioni.",
  ],
);
METHOD_SYMBOLS.perceptron.push([
  "wᵀx",
  "Prodotto scalare che misura il punteggio del campione rispetto al confine.",
]);
METHOD_SYMBOLS.linear.push(
  [
    "wᵀφ(x)",
    "Prodotto scalare dei coefficienti per le feature trasformate; la trasposizione indica il prodotto scalare.",
  ],
  ["Σ", "Somma degli errori quadratici sui campioni."],
);
METHOD_SYMBOLS.ridge.push(
  ["‖w‖₂", "Norma L2: misura la grandezza complessiva dei coefficienti."],
  ["Σ", "Somma usata per l’errore medio e la penalità sui pesi."],
);
METHOD_SYMBOLS.lasso.push(
  ["‖w‖₁", "Norma L1: somma i valori assoluti dei coefficienti."],
  ["Σ", "Somma degli errori sui campioni."],
);
METHOD_SYMBOLS.elastic.push(
  [
    "‖w‖₁, ‖w‖₂",
    "Norme L1 e L2 dei pesi; generano rispettivamente selezione e riduzione dei coefficienti.",
  ],
  ["Σ", "Somma degli errori e dei contributi di regolarizzazione."],
);
METHOD_SYMBOLS.svmrbf.push(
  ["exp", "Esponenziale; trasforma la distanza in una somiglianza tra 0 e 1."],
  ["Σᵢ", "Somma sui campioni delle penalità di margine."],
  ["wᵀφ(x)", "Prodotto scalare nello spazio trasformato dal kernel."],
);
METHOD_SYMBOLS.svmpoly.push(
  ["xᵀz", "Prodotto scalare tra i due campioni."],
  ["Σᵢ", "Somma delle penalità di margine sui campioni."],
);
METHOD_SYMBOLS.tree.push(
  [
    "1−, −",
    "Indicano rispettivamente il complemento della somma Gini e il segno negativo della somma nell’entropia.",
  ],
  ["²", "Quadrato delle proporzioni di classe nel Gini."],
);
METHOD_SYMBOLS.forest.push(["n", "Numero totale di alberi che votano."]);
METHOD_SYMBOLS.boost.push([
  "−∇",
  "Gradiente negativo della loss: indica la correzione che il nuovo stimatore cerca di apprendere.",
]);
METHOD_SYMBOLS.naive.push(
  ["Π", "Prodotto delle probabilità condizionate delle feature."],
  [
    "α",
    "Smoothing additivo: evita probabilità nulle per valori di feature non osservati nella classe.",
  ],
);
METHOD_SYMBOLS.mlp.push(
  ["l", "Indice dello strato della rete; l−1 indica lo strato precedente."],
  ["ΣᵢΣₖ", "Somma della cross-entropy su esempi e classi."],
);
METHOD_SYMBOLS.mlp.push(
  [
    "mₜ, vₜ",
    "Medie mobili del gradiente e del gradiente al quadrato usate da Adam.",
  ],
  ["β₁, β₂", "Tassi di decadimento delle medie mobili di Adam."],
  ["ε", "Piccola costante per evitare divisioni per zero nell’aggiornamento."],
);
METHOD_SYMBOLS.sgd.push([
  "∇",
  "Operatore gradiente: raccoglie le derivate della loss rispetto ai parametri.",
]);
METHOD_SYMBOLS.momentum.push([
  "∇",
  "Operatore gradiente della funzione obiettivo.",
]);
METHOD_SYMBOLS.nesterov.push([
  "∇",
  "Operatore gradiente calcolato nella posizione anticipata.",
]);
METHOD_SYMBOLS.adagrad.push(
  ["∇", "Operatore gradiente della funzione obiettivo."],
  ["²", "Quadrato elemento per elemento del gradiente."],
);
METHOD_SYMBOLS.rmsprop.push(
  ["∇", "Operatore gradiente della funzione obiettivo."],
  ["²", "Quadrato elemento per elemento del gradiente."],
);
METHOD_SYMBOLS.adam.push(
  ["∇", "Operatore gradiente della funzione obiettivo."],
  ["²", "Quadrato elemento per elemento del gradiente."],
  [
    "√",
    "Radice quadrata della stima del secondo momento; normalizza l’aggiornamento.",
  ],
);
METHOD_SYMBOLS.adamw.push(
  ["∇", "Operatore gradiente della funzione obiettivo."],
  ["²", "Quadrato elemento per elemento del gradiente."],
  ["√", "Radice quadrata della stima del secondo momento."],
);
METHOD_SYMBOLS.kmeans.push(
  ["minₖ", "Valore della distanza più piccola tra tutti i centroidi."],
  ["Σ", "Somma delle distanze al quadrato di tutti i campioni."],
);
METHOD_SYMBOLS.gmm.find(([symbol]) => symbol === "Σₖ")[1] =
  "Nella sommatoria indica l’addizione sulle componenti; come pedice della covarianza in N(x|μₖ,Σₖ), indica la matrice di covarianza della componente k.";
METHOD_SYMBOLS.gmm.push([
  "|",
  "Dato che: N(x|μ,Σ) indica una gaussiana con media μ e covarianza Σ.",
]);
METHOD_SYMBOLS.softmax.push([
  "log",
  "Logaritmo naturale nella cross-entropy; penalizza probabilità basse assegnate alla classe corretta.",
]);
METHOD_SYMBOLS.mlp.push([
  "log",
  "Logaritmo naturale nella cross-entropy; misura la penalità per la probabilità assegnata alla classe vera.",
]);
METHOD_SYMBOLS.kmeans.push([
  "media{xᵢ:cᵢ=k}",
  "Media delle feature dei punti assegnati al cluster; diventa il nuovo centroide.",
]);
METHOD_SYMBOLS.tree.push([
  "y, ŷ",
  "Classe corretta e classe predetta dalla foglia dell’albero.",
]);
// Parameter tuples are [key, visible label, default value, input type].
// A select type encodes its choices after "select:", separated by vertical bars.
const PARAMS = {
  logistic: [
    ["learningRate", "Learning rate α", "0.08", "number"],
    ["epochs", "Epoche", "120", "number"],
    [
      "batchSize",
      "Batch size",
      "Full batch",
      "select:Full batch|Mini-batch|SGD",
    ],
    ["lambda", "Regolarizzazione λ", "0", "number"],
    ["threshold", "Soglia", "0.5", "number"],
  ],
  softmax: [
    ["learningRate", "Learning rate α", "0.05", "number"],
    ["epochs", "Epoche", "150", "number"],
    [
      "batchSize",
      "Batch size",
      "Full batch",
      "select:Full batch|Mini-batch|SGD",
    ],
    ["lambda", "Regolarizzazione λ", "0", "number"],
  ],
  perceptron: [
    ["learningRate", "Learning rate α", "0.1", "number"],
    ["epochs", "Epoche", "100", "number"],
    ["shuffle", "Shuffle", "true", "select:true|false"],
  ],
  linear: [
    ["featureDegree", "Grado feature (auto)", "auto", "select:auto|1|2|3|4|5"],
    ["learningRate", "Learning rate α", "0.08", "number"],
    ["epochs", "Epoche", "120", "number"],
    [
      "batchSize",
      "Batch size",
      "Full batch",
      "select:Full batch|Mini-batch|SGD",
    ],
  ],
  ridge: [
    ["featureDegree", "Grado feature (auto)", "auto", "select:auto|1|2|3|4|5"],
    ["learningRate", "Learning rate", "0.01", "number"],
    ["lambda", "λ L2", "0.1", "number"],
    ["epochs", "Epoche", "500", "number"],
  ],
  lasso: [
    ["featureDegree", "Grado feature (auto)", "auto", "select:auto|1|2|3|4|5"],
    ["learningRate", "Learning rate", "0.01", "number"],
    ["lambda", "λ L1", "0.1", "number"],
    ["epochs", "Epoche", "500", "number"],
  ],
  elastic: [
    ["featureDegree", "Grado feature (auto)", "auto", "select:auto|1|2|3|4|5"],
    ["learningRate", "Learning rate", "0.01", "number"],
    ["lambda", "λ", "0.1", "number"],
    ["l1Ratio", "L1 ratio ρ", "0.5", "number"],
    ["epochs", "Epoche", "500", "number"],
  ],
  knn: [
    ["k", "Numero vicini k", "5", "number"],
    ["metric", "Distanza", "euclidean", "select:euclidean|manhattan"],
    ["weights", "Pesi vicini", "uniform", "select:uniform|distance"],
  ],
  svmrbf: [
    ["C", "C", "1", "number"],
    ["gamma", "Gamma", "0.5", "number"],
    ["tolerance", "Tolleranza", "0.001", "number"],
    ["maxIter", "Iterazioni max", "100", "number"],
  ],
  svmpoly: [
    ["C", "C", "1", "number"],
    ["degree", "Grado polinomiale", "3", "select:1|2|3|4|5|6|7|8"],
    ["gamma", "Gamma", "0.5", "number"],
    ["coef0", "Coef0", "1", "number"],
    ["tolerance", "Tolleranza", "0.001", "number"],
    ["maxIter", "Iterazioni max", "100", "number"],
  ],
  tree: [
    ["criterion", "Criterio", "gini", "select:gini|entropy|log_loss"],
    ["maxDepth", "Profondità max", "5", "number"],
    ["minSplit", "Min campioni split", "2", "number"],
    ["minLeaf", "Min campioni foglia", "1", "number"],
  ],
  forest: [
    ["nEstimators", "Numero alberi", "100", "number"],
    ["maxDepth", "Profondità max", "8", "number"],
    ["maxFeatures", "Feature per split", "sqrt", "select:sqrt|log2|all"],
    ["criterion", "Criterio", "gini|entropy|log_loss"],
    ["minLeaf", "Min campioni foglia", "1", "number"],
    ["bootstrap", "Bootstrap", "true", "select:true|false"],
  ],
  boost: [
    ["nEstimators", "Numero stimatori", "100", "number"],
    ["learningRate", "Learning rate", "0.1", "number"],
    ["maxDepth", "Profondità base learner", "3", "number"],
    ["subsample", "Subsample", "1", "number"],
  ],
  naive: [
    [
      "variant",
      "Variante",
      "gaussian",
      "select:gaussian|multinomial|bernoulli",
    ],
    ["smoothing", "Smoothing α", "1", "number"],
  ],
  mlp: [
    ["learningRate", "Learning rate", "0.08", "number"],
    ["epochs", "Epoche", "100", "number"],
    [
      "batchSize",
      "Batch size",
      "Full batch",
      "select:Full batch|Mini-batch|SGD",
    ],
    ["hidden", "Neuroni hidden", "5", "number"],
  ],
  sgd: [
    ["learningRate", "Learning rate", "0.01", "number"],
    ["batchSize", "Batch size", "1", "number"],
    ["shuffle", "Shuffle", "true", "select:true|false"],
  ],
  momentum: [
    ["learningRate", "Learning rate", "0.01", "number"],
    ["momentum", "Momentum μ", "0.9", "number"],
    ["batchSize", "Batch size", "32", "number"],
  ],
  nesterov: [
    ["learningRate", "Learning rate", "0.01", "number"],
    ["momentum", "Momentum μ", "0.9", "number"],
  ],
  adagrad: [
    ["learningRate", "Learning rate", "0.01", "number"],
    ["epsilon", "Epsilon", "1e-8", "number"],
  ],
  rmsprop: [
    ["learningRate", "Learning rate", "0.001", "number"],
    ["rho", "Decay ρ", "0.9", "number"],
    ["epsilon", "Epsilon", "1e-8", "number"],
  ],
  adam: [
    ["learningRate", "Learning rate", "0.001", "number"],
    ["beta1", "Beta 1", "0.9", "number"],
    ["beta2", "Beta 2", "0.999", "number"],
    ["epsilon", "Epsilon", "1e-8", "number"],
  ],
  adamw: [
    ["learningRate", "Learning rate", "0.001", "number"],
    ["beta1", "Beta 1", "0.9", "number"],
    ["beta2", "Beta 2", "0.999", "number"],
    ["epsilon", "Epsilon", "1e-8", "number"],
    ["weightDecay", "Weight decay λ", "0.01", "number"],
  ],
  kmeans: [
    ["clusters", "Numero cluster k", "3", "number"],
    ["init", "Inizializzazione", "kmeans++", "select:kmeans++|random"],
    ["maxIter", "Iterazioni max", "100", "number"],
  ],
  dbscan: [
    ["eps", "Epsilon", "0.5", "number"],
    ["minPts", "MinPts", "5", "number"],
    ["metric", "Distanza", "euclidean", "select:euclidean|manhattan"],
  ],
  gmm: [
    ["components", "Componenti", "3", "number"],
    ["covariance", "Covarianza", "full", "select:full|diag|spherical"],
    ["maxIter", "Iterazioni max", "100", "number"],
    ["regCovar", "Regolarizzazione covarianza", "0.000001", "number"],
  ],
};
PARAMS.forest[3] = [
  "criterion",
  "Criterio",
  "gini",
  "select:gini|entropy|log_loss",
];
const COMMON_METHOD_SYMBOLS = [
  [
    "x, xᵢ, xⱼ",
    "Feature/input del campione; gli indici identificano campione o feature.",
  ],
  ["y, yᵢ", "Target osservato o classe corretta."],
  ["ŷ", "Previsione prodotta dal modello."],
  ["W, w, wₖ", "Pesi/coefficienti appresi dal modello."],
  ["b, bₖ", "Bias/intercetta appresa dal modello."],
  [
    "J, L, Lᵢ",
    "Funzione di costo (loss), sul batch o sul singolo campione indicato.",
  ],
  ["θ, θₜ", "Insieme dei parametri del modello al passo t."],
  ["gₜ", "Gradiente della loss rispetto ai parametri al passo t."],
  ["α", "Learning rate, cioè ampiezza dell’aggiornamento."],
  ["λ", "Intensità della regolarizzazione o del weight decay."],
  ["ρ", "Fattore di decadimento o proporzione L1, secondo il metodo."],
  ["μ", "Momentum o media/centro, secondo il metodo."],
  ["β₁, β₂", "Fattori di decadimento delle medie mobili di Adam."],
  ["ε", "Piccola costante per stabilizzare divisioni e radici."],
  [
    "γ",
    "Parametro di scala del kernel o fattore di decadimento, secondo il metodo.",
  ],
  ["C", "Penalità delle violazioni del margine nella SVM."],
  ["η", "Learning rate del boosting."],
  [
    "i, j, k, m, n",
    "Indici di campioni, feature, classi o componenti; m è il numero di campioni nel batch.",
  ],
  [
    "Σ, Σᵢ, Σⱼ, Σₖ",
    "Somma sui campioni, feature, classi o componenti indicati.",
  ],
  ["Π, Πⱼ", "Prodotto dei termini sulle feature indicate."],
  ["∇, ∂", "Gradiente o derivata rispetto alla variabile indicata."],
  ["exp", "Funzione esponenziale."],
  ["log, log₂", "Logaritmo naturale o in base 2."],
  ["argmax, argmin", "Indice che massimizza o minimizza l’espressione."],
  ["max, min", "Massimo o minimo tra i valori indicati."],
  [
    "‖·‖, ‖w‖₂, ‖w‖₁",
    "Norma: misura la grandezza del vettore; L1 usa valori assoluti, L2 quadrati.",
  ],
  ["⊙", "Prodotto elemento per elemento tra vettori."],
  ["sign", "Segno del valore, usato nella soglia prossimale."],
  ["clip", "Limita un valore all’intervallo specificato."],
  ["φ", "Trasformazione delle feature o attivazione, secondo il metodo."],
  ["p(·), pᵢₖ", "Probabilità stimata dal modello."],
  ["Tᵢ, Tₙ", "Albero decisionale i-esimo o numero di alberi dell’ensemble."],
  ["K(x,z), Kᵢᵢ", "Kernel/somiglianza tra campioni."],
  ["Nε(x)", "Insieme dei vicini entro epsilon dal punto x."],
  ["cᵢ", "Cluster assegnato al campione i."],
  ["μₖ", "Centroide o media della componente k."],
  ["πₖ", "Peso/probabilità della componente gaussiana k."],
  ["Σₖ", "Matrice di covarianza della componente gaussiana k."],
  [
    "Fₘ(x), Fₘ₋₁(x)",
    "Predizione del boosting dopo m stadi o prima dello stadio corrente.",
  ],
  ["hₘ(x)", "Albero/stimatore aggiunto allo stadio m del boosting."],
  [
    "r, rₜ",
    "Coefficiente del kernel o accumulatore adattivo del gradiente, secondo il metodo.",
  ],
  [
    "vₜ",
    "Velocità del momentum o media mobile dei gradienti quadratici, secondo l’ottimizzatore.",
  ],
];
const METHOD_FUNCTION_SYMBOLS = [
  [
    "softmax",
    "Trasforma punteggi (logit) in probabilità positive che sommano a 1: softmax(z)ₖ = exp(zₖ)/Σⱼexp(zⱼ).",
  ],
  [
    "sigmoid",
    "Funzione σ(z)=1/(1+exp(−z)); comprime un punteggio in una probabilità tra 0 e 1.",
  ],
  [
    "tanh",
    "Funzione di attivazione che trasforma z in un valore tra −1 e 1, permettendo alla rete di apprendere relazioni non lineari.",
  ],
  [
    "ReLU",
    "Funzione di attivazione max(0,z): azzera i valori negativi e lascia invariati quelli positivi.",
  ],
  [
    "one-hot",
    "Codifica una classe come vettore di zeri con 1 nella posizione della classe corretta, per esempio classe 2 → [0,0,1].",
  ],
  [
    "cross-entropy",
    "Loss di classificazione: −Σₖ yₖ log(pₖ); penalizza soprattutto le probabilità basse assegnate alla classe corretta.",
  ],
  [
    "MSE",
    "Errore quadratico medio: media di (previsione − target)²; penalizza maggiormente gli errori grandi.",
  ],
  ["RMSE", "Radice della MSE; esprime l’errore nella stessa unità del target."],
  ["MAE", "Errore assoluto medio: media di |previsione − target|."],
  [
    "R²",
    "Quota di variabilità del target spiegata dalla regressione; 1 è perfetto, 0 equivale alla media, può essere negativo.",
  ],
  [
    "hinge loss",
    "Loss SVM max(0,1−y·f(x)); è zero quando il campione è classificato correttamente oltre il margine.",
  ],
  [
    "log-likelihood",
    "Logaritmo della probabilità/densità assegnata dal modello ai dati osservati; valori maggiori indicano un adattamento migliore.",
  ],
  [
    "negative log-likelihood",
    "Opposto della log-likelihood; viene minimizzato e penalizza le distribuzioni che assegnano bassa probabilità ai dati.",
  ],
  [
    "Gini",
    "Misura l’impurità delle classi in un nodo: 1−Σₖpₖ²; vale zero quando tutti i campioni appartengono a una classe.",
  ],
  [
    "Entropy",
    "Misura l’incertezza delle classi: −Σₖpₖlog₂pₖ; è zero per un nodo con una sola classe.",
  ],
  [
    "inertia",
    "Somma delle distanze quadratiche dei punti dai centroidi assegnati; misura la compattezza dei cluster.",
  ],
  [
    "kernel",
    "Funzione che misura la somiglianza tra campioni e permette alla SVM di costruire confini non lineari.",
  ],
  [
    "residuo",
    "Differenza tra target osservato e previsione; nei metodi ad alberi indica l’errore che un nuovo stadio cerca di correggere.",
  ],
  [
    "regularization",
    "Penalità aggiunta alla loss per limitare la complessità o la grandezza dei parametri e ridurre l’overfitting.",
  ],
  [
    "mini-batch",
    "Sottoinsieme di campioni usato per stimare il gradiente e aggiornare i parametri a ogni passo.",
  ],
  [
    "backpropagation",
    "Algoritmo che propaga all’indietro l’errore della loss per calcolare i gradienti dei pesi di ogni strato.",
  ],
  [
    "responsabilità",
    "Probabilità stimata che un campione appartenga a ciascuna componente di un GMM; nell’EM sono calcolate nell’E-step.",
  ],
];
COMMON_METHOD_SYMBOLS.push(...METHOD_FUNCTION_SYMBOLS);

export { METHOD_INFO, PARAMS, METHOD_SYMBOLS, COMMON_METHOD_SYMBOLS };
export const defaultParameters = (id) =>
  Object.fromEntries((PARAMS[id] || []).map(([key, , value]) => [key, value]));
