export const learningRows = `
Loss|Loss|Потери|Pérdida|损失
Hinge loss sul punteggio kernel|Hinge loss on the kernel score|Шарнирные потери по оценке ядра|Pérdida hinge sobre la puntuación del kernel|核得分上的合页损失
Aggiorna solo se|Update only if|Обновлять только если|Actualizar solo si|仅在以下情况下更新
Nessuna funzione di costo ottimizzata: KNN è un metodo instance-based.|No optimized cost function: KNN is instance-based.|Нет оптимизируемой функции потерь: KNN основан на примерах.|Sin función de coste optimizada: KNN se basa en instancias.|没有需要优化的损失函数：KNN 是基于实例的方法。
Nessun aggiornamento dei pesi. Si memorizzano gli esempi; la previsione usa i k vicini.|No weight updates. Stores samples and predicts using k neighbors.|Веса не обновляются. Сохраняет объекты и прогнозирует по k соседям.|No actualiza pesos. Guarda muestras y predice usando k vecinos.|不更新权重。存储样本，并用 k 个近邻进行预测。
Sceglie lo split con impurità pesata minima; ricorsione fino ai limiti impostati.|Chooses the split with minimum weighted impurity; recurses to the configured limits.|Выбирает разбиение с минимальной взвешенной нечистотой; рекурсия до заданных пределов.|Elige la división con impureza ponderada mínima; recursión hasta los límites configurados.|选择加权不纯度最小的分裂，递归至设定限制。
Negative log-likelihood dei parametri stimati dai dati|Negative log-likelihood of parameters estimated from data|Отрицательный логарифм правдоподобия параметров, оценённых по данным|Log-verosimilitud negativa de parámetros estimados de los datos|由数据估计的参数的负对数似然
loss del campione o mini-batch selezionato|loss of the selected sample or mini-batch|потери выбранного объекта или мини-пакета|pérdida de la muestra o minilote elegido|所选样本或小批次的损失
loss del batch corrente|loss of the current batch|потери текущего пакета|pérdida del lote actual|当前批次的损失
il weight decay è separato|weight decay is separate|затухание весов отделено|el decaimiento de pesos está separado|权重衰减单独处理
assegna xᵢ al centro μₖ più vicino|assign xᵢ to the nearest center μₖ|назначить xᵢ ближайшему центру μₖ|asigna xᵢ al centro μₖ más cercano|将 xᵢ 分配给最近的中心 μₖ
Non minimizza una funzione di costo globale standard|Does not minimize a standard global cost function|Не минимизирует стандартную глобальную функцию потерь|No minimiza una función de coste global estándar|不最小化标准的全局损失函数
I pesi sono ancora nulli: dopo un aggiornamento sarà definita una distanza perpendicolare alla retta.|Weights are still zero: one update will define a perpendicular distance to the line.|Веса пока нулевые: после обновления появится перпендикулярное расстояние до прямой.|Los pesos aún son cero: tras una actualización se definirá una distancia perpendicular a la recta.|权重仍为零：更新后才能定义到直线的垂直距离。
Le distanze firmate sono perpendicolari alla retta e positive sul lato della classe 1.|Signed distances are perpendicular to the line and positive on class 1's side.|Знаковые расстояния перпендикулярны прямой и положительны со стороны класса 1.|Las distancias con signo son perpendiculares a la recta y positivas del lado de la clase 1.|有符号距离垂直于直线，在类别 1 一侧为正。
Attivazioni della rete sul campione selezionato|Network activations for the selected sample|Активации сети для выбранного объекта|Activaciones de la red para la muestra elegida|所选样本的网络激活值
Punteggio e probabilità per classe|Score and probability by class|Оценка и вероятность по классам|Puntuación y probabilidad por clase|各类别得分与概率
Probabilità per classe|Probability by class|Вероятности классов|Probabilidad por clase|各类别概率
Voti degli alberi per il campione selezionato|Tree votes for the selected sample|Голоса деревьев для выбранного объекта|Votos de árboles para la muestra elegida|各棵树对所选样本的投票
Percorso del campione nell’albero|Sample path through the tree|Путь объекта в дереве|Recorrido de la muestra por el árbol|样本在树中的路径
Sfondo = risposta prevista dal modello; punti = dati osservati. Seleziona un campione dal controllo o dalla tabella per ispezionarlo.|Background = model prediction; points = observed data. Select a sample using the control or table to inspect it.|Фон = прогноз модели; точки = наблюдаемые данные. Выберите объект через поле или таблицу для анализа.|Fondo = predicción del modelo; puntos = datos observados. Selecciona una muestra con el control o la tabla para inspeccionarla.|背景表示模型预测，点表示观测数据。通过控件或表格选择样本以查看详情。
Servono almeno 4 campioni, due feature numeriche e un target finale.|At least 4 samples, two numeric features, and a final target are required.|Нужны минимум 4 объекта, два числовых признака и целевая переменная в конце.|Se requieren al menos 4 muestras, dos características numéricas y un objetivo final.|至少需要 4 个样本、两个数值特征及最后一列目标值。
Il CSV contiene valori mancanti, non numerici o righe di lunghezza diversa.|The CSV contains missing or nonnumeric values, or rows with different lengths.|CSV содержит пропуски, нечисловые значения или строки разной длины.|El CSV contiene valores ausentes, no numéricos o filas de distinta longitud.|CSV 含有缺失值、非数值或长度不一致的行。
moda|mode|мода|moda|众数
media|mean|среднее|media|均值
Centro|Center|Центр|Centro|中心
Assegnazione|Assignment|Назначение|Asignación|分配
Parametri|Parameters|Параметры|Parámetros|参数
Errore|Error|Ошибка|Error|误差
Probabilità|Probability|Вероятность|Probabilidad|概率
Punteggi|Scores|Оценки|Puntuaciones|得分
Coefficienti|Coefficients|Коэффициенты|Coeficientes|系数
Pesi|Weights|Веса|Pesos|权重
Medie del training|Training means|Средние обучающей выборки|Medias de entrenamiento|训练集均值
Scale del training|Training scales|Масштабы обучающей выборки|Escalas de entrenamiento|训练集尺度
grezzo|raw|исходный|original|原始
standardizzato|standardized|стандартизованный|estandarizado|标准化
residuo|residual|остаток|residuo|残差
classe reale|true class|истинный класс|clase real|真实类别
classe|class|класс|clase|类别
previsione corrente|current prediction|текущий прогноз|predicción actual|当前预测
Previsione corrente|Current prediction|Текущий прогноз|Predicción actual|当前预测
Aggiornamento reale|Actual update|Фактическое обновление|Actualización real|实际更新
Matrice W completa|Full W matrix|Полная матрица W|Matriz W completa|完整 W 矩阵
Bias completo|Full bias|Полный вектор смещений|Sesgo completo|完整偏置
prime righe|first rows|первые строки|primeras filas|前几行
somma|sum|сумма|suma|求和
metrica|metric|метрика|métrica|度量
centroide|centroid|центроид|centroide|质心
indici|indices|индексы|índices|索引
non ancora assegnato|not assigned yet|пока не назначен|aún sin asignar|尚未分配
non trovato|not found|не найден|no encontrado|未找到
non soddisfatto|not satisfied|не выполнено|no cumplido|未满足
soddisfatto|satisfied|выполнено|cumplido|已满足
Componenti|Components|Компоненты|Componentes|分量
Regolarizzazione covarianza|Covariance regularization|Регуляризация ковариации|Regularización de covarianza|协方差正则化
Shuffle|Shuffle|Перемешивание|Mezclar|打乱顺序
Momentum μ|Momentum μ|Импульс μ|Momento μ|动量 μ
Weight decay λ|Weight decay λ|Затухание весов λ|Decaimiento de pesos λ|权重衰减 λ
Smoothing α|Smoothing α|Сглаживание α|Suavizado α|平滑系数 α
Decay ρ|Decay ρ|Затухание ρ|Decaimiento ρ|衰减系数 ρ
Subsample|Subsample|Доля подвыборки|Submuestreo|子采样比例
euclidean|Euclidean|Евклидово|Euclídea|欧氏距离
manhattan|Manhattan|Манхэттенское|Manhattan|曼哈顿距离
uniform|Uniform|Равномерные|Uniformes|均匀
distance|Distance|По расстоянию|Por distancia|按距离
entropy|Entropy|Энтропия|Entropía|熵
all|All|Все|Todas|全部
random|Random|Случайная|Aleatoria|随机
true|Yes|Да|Sí|是
false|No|Нет|No|否
auto|Auto|Авто|Auto|自动
gaussian|Gaussian|Гауссовский|Gaussiana|高斯
multinomial|Multinomial|Мультиномиальный|Multinomial|多项式
bernoulli|Bernoulli|Бернулли|Bernoulli|伯努利
full|Full|Полная|Completa|完整
diag|Diagonal|Диагональная|Diagonal|对角
spherical|Spherical|Сферическая|Esférica|球形
Implementato per classificazione binaria; per più classi usa Softmax.|Implemented for binary classification; use Softmax for more classes.|Реализовано для двух классов; для нескольких используйте Softmax.|Implementado para clasificación binaria; usa Softmax para más clases.|支持二分类；多类别请使用 Softmax。
Stima la probabilità di una delle due classi con la funzione sigmoide. Per classificare più di due classi seleziona Softmax. Metriche: accuracy, precision, recall, F1 e matrice di confusione.|Estimates the probability of either class with a sigmoid. Select Softmax for more than two classes. Metrics: accuracy, precision, recall, F1, and confusion matrix.|Оценивает вероятность одного из двух классов с помощью сигмоиды. Для более чем двух классов выберите Softmax. Метрики: доля верных ответов, точность, полнота, F1 и матрица ошибок.|Estima la probabilidad de una de dos clases con la sigmoide. Para más de dos clases, selecciona Softmax. Métricas: exactitud, precisión, exhaustividad, F1 y matriz de confusión.|通过 Sigmoid 函数估计两个类别的概率。超过两个类别时请选择 Softmax。指标包括准确率、精确率、召回率、F1 和混淆矩阵。
Implementato nel motore come classificatore lineare multiclasse.|Implemented as a multiclass linear classifier.|Реализован линейный многоклассовый классификатор.|Implementado como clasificador lineal multiclase.|已实现线性多分类器。
Un vettore di probabilità per ogni classe. Metriche: accuracy, macro/micro precision, recall, F1 e confusion matrix.|Produces a vector of class probabilities. Metrics: accuracy, macro/micro precision, recall, F1, and confusion matrix.|Формирует вектор вероятностей классов. Метрики: доля верных ответов, макро/микро точность, полнота, F1 и матрица ошибок.|Produce un vector de probabilidades por clase. Métricas: exactitud, precisión macro/micro, exhaustividad, F1 y matriz de confusión.|输出各类别的概率向量。指标包括准确率、宏/微平均精确率、召回率、F1 和混淆矩阵。
Implementato nel motore per classificazione binaria e multiclasse.|Implemented for binary and multiclass classification.|Реализован для бинарной и многоклассовой классификации.|Implementado para clasificación binaria y multiclase.|支持二分类与多分类。
Aggiorna i parametri quando un esempio è classificato male. Metriche: accuracy, precision, recall, F1.|Updates parameters when a sample is misclassified. Metrics: accuracy, precision, recall, F1.|Обновляет параметры при ошибочной классификации. Метрики: доля верных ответов, точность, полнота, F1.|Actualiza los parámetros cuando una muestra se clasifica mal. Métricas: exactitud, precisión, exhaustividad y F1.|样本分类错误时更新参数。指标包括准确率、精确率、召回率和 F1。
Regressione lineare con feature originali o trasformate.|Linear regression with original or transformed features.|Линейная регрессия с исходными или преобразованными признаками.|Regresión lineal con características originales o transformadas.|使用原始或变换特征的线性回归。
Il grado automatico usa una base lineare, quadratica o polinomiale in base al dataset; puoi impostarlo manualmente. Metriche: MSE, RMSE, MAE e R².|Automatic degree selects a linear, quadratic, or polynomial basis for the dataset; you can override it. Metrics: MSE, RMSE, MAE, and R².|Автоматическая степень выбирает линейный, квадратичный или полиномиальный базис по данным; её можно задать вручную. Метрики: MSE, RMSE, MAE и R².|El grado automático elige una base lineal, cuadrática o polinómica según los datos; puedes cambiarlo. Métricas: MSE, RMSE, MAE y R².|自动阶数根据数据集选择线性、二次或多项式基函数，也可手动设置。指标包括 MSE、RMSE、MAE 和 R²。
Ridge Regression implementata con feature originali o trasformate.|Ridge regression with original or transformed features.|Ridge-регрессия с исходными или преобразованными признаками.|Regresión Ridge con características originales o transformadas.|使用原始或变换特征的岭回归。
Regolarizzazione L2 sulla base selezionata: riduce i coefficienti senza azzerarli in generale. Puoi scegliere il grado delle feature.|L2 regularization on the selected basis shrinks coefficients without generally setting them to zero. You can choose the feature degree.|L2-регуляризация выбранного базиса уменьшает коэффициенты, обычно не обнуляя их. Степень признаков можно выбрать.|La regularización L2 reduce los coeficientes de la base elegida sin anularlos en general. Puedes elegir el grado de las características.|L2 正则化收缩所选基函数的系数，通常不会使其变为零。可选择特征阶数。
Lasso implementato con aggiornamento prossimale e feature trasformate.|Lasso with proximal updates and transformed features.|Lasso с проксимальными обновлениями и преобразованными признаками.|Lasso con actualizaciones proximales y características transformadas.|使用近端更新与变换特征的 Lasso。
Regolarizzazione L1 sulla base selezionata: può azzerare coefficienti e selezionare termini polinomiali. Puoi scegliere il grado delle feature.|L1 regularization on the selected basis can zero coefficients and select polynomial terms. You can choose the feature degree.|L1-регуляризация может обнулять коэффициенты и отбирать полиномиальные члены. Степень признаков можно выбрать.|La regularización L1 puede anular coeficientes y seleccionar términos polinómicos. Puedes elegir el grado de las características.|L1 正则化可将系数置零并选择多项式项。可选择特征阶数。
Elastic Net implementato con penalità L1 e L2.|Elastic Net with L1 and L2 penalties.|Elastic Net со штрафами L1 и L2.|Elastic Net con penalizaciones L1 y L2.|使用 L1 和 L2 惩罚的 Elastic Net。
Combina selezione delle feature (L1) e shrinkage (L2) sulla base scelta. Il grado automatico si adatta ai dataset sintetici; puoi modificarlo.|Combines feature selection (L1) and shrinkage (L2) on the selected basis. Automatic degree adapts to synthetic datasets and can be changed.|Сочетает отбор признаков (L1) и сжатие коэффициентов (L2). Автоматическая степень адаптируется к синтетическим данным и может быть изменена.|Combina selección de características (L1) y contracción (L2). El grado automático se adapta a los datos sintéticos y se puede cambiar.|结合特征选择（L1）和系数收缩（L2）。自动阶数适应合成数据集，也可修改。
Implementato nel motore base; classificazione multiclasse e distanza sulle feature supportate.|Supports multiclass classification and distances over features.|Поддерживает многоклассовую классификацию и расстояния по признакам.|Admite clasificación multiclase y distancias entre características.|支持多分类与特征距离计算。
Metriche: accuracy, precision, recall, F1, confusion matrix. Parametri: k, metrica e pesatura dei vicini.|Metrics: accuracy, precision, recall, F1, confusion matrix. Parameters: k, distance metric, neighbor weighting.|Метрики: доля верных ответов, точность, полнота, F1, матрица ошибок. Параметры: k, метрика расстояния, веса соседей.|Métricas: exactitud, precisión, exhaustividad, F1 y matriz de confusión. Parámetros: k, métrica y pesos de vecinos.|指标包括准确率、精确率、召回率、F1 和混淆矩阵。参数：k、距离度量和近邻权重。
SVM con kernel RBF implementata tramite aggiornamento duale semplificato.|RBF-kernel SVM with simplified dual updates.|SVM с RBF-ядром и упрощёнными двойственными обновлениями.|SVM con kernel RBF y actualizaciones duales simplificadas.|使用简化对偶更新的 RBF 核 SVM。
Kernel RBF produce frontiere non lineari. Parametri: C, γ, tolleranza, criterio multiclasse. Metriche di classificazione.|The RBF kernel produces nonlinear boundaries. Parameters: C, γ, tolerance, multiclass criterion. Classification metrics.|RBF-ядро создаёт нелинейные границы. Параметры: C, γ, допуск, многоклассовый критерий. Метрики классификации.|El kernel RBF produce fronteras no lineales. Parámetros: C, γ, tolerancia y criterio multiclase. Métricas de clasificación.|RBF 核产生非线性边界。参数：C、γ、容差与多分类准则。使用分类指标。
SVM con kernel polinomiale implementata tramite aggiornamento duale semplificato.|Polynomial-kernel SVM with simplified dual updates.|SVM с полиномиальным ядром и упрощёнными двойственными обновлениями.|SVM con kernel polinómico y actualizaciones duales simplificadas.|使用简化对偶更新的多项式核 SVM。
Kernel polinomiale. Parametri: C, grado d, γ, coef0. Metriche di classificazione.|Polynomial kernel. Parameters: C, degree d, γ, coef0. Classification metrics.|Полиномиальное ядро. Параметры: C, степень d, γ, coef0. Метрики классификации.|Kernel polinómico. Parámetros: C, grado d, γ y coef0. Métricas de clasificación.|多项式核。参数：C、阶数 d、γ、coef0。使用分类指标。
Decision Tree implementato con ricerca di split e criteri Gini/entropia.|Decision tree with split search and Gini/entropy criteria.|Дерево решений с поиском разбиений по Джини/энтропии.|Árbol de decisión con búsqueda de divisiones y criterios Gini/entropía.|使用分裂搜索及基尼/熵准则的决策树。
Parametri: profondità massima, min campioni split/foglia, criterio. Metriche: classificazione.|Parameters: maximum depth, minimum split/leaf samples, criterion. Classification metrics.|Параметры: максимальная глубина, минимум объектов для разбиения/листа, критерий. Метрики классификации.|Parámetros: profundidad máxima, muestras mínimas para división/hoja y criterio. Métricas de clasificación.|参数：最大深度、分裂/叶节点最少样本数、准则。使用分类指标。
Random Forest implementata con bootstrap e selezione casuale delle feature.|Random forest with bootstrap sampling and random feature selection.|Случайный лес с бутстрэпом и случайным выбором признаков.|Bosque aleatorio con bootstrap y selección aleatoria de características.|使用自助采样和随机特征选择的随机森林。
Parametri: numero alberi, profondità, max feature, bootstrap, min samples leaf. Metriche: classificazione e OOB se disponibile.|Parameters: trees, depth, maximum features, bootstrap, minimum leaf samples. Metrics: classification and OOB when available.|Параметры: число деревьев, глубина, максимум признаков, бутстрэп, минимум объектов в листе. Метрики классификации и OOB, если доступна.|Parámetros: árboles, profundidad, características máximas, bootstrap y muestras mínimas por hoja. Métricas de clasificación y OOB si está disponible.|参数：树数、深度、最大特征数、自助采样、叶节点最少样本数。使用分类指标及可用时的袋外评估。
Gradient Boosting implementato per regressione e classificazione.|Gradient boosting for regression and classification.|Градиентный бустинг для регрессии и классификации.|Gradient boosting para regresión y clasificación.|用于回归与分类的梯度提升。
Parametri: numero stimatori, learning rate, profondità/alberi, subsampling. Metriche: loss, accuracy, F1.|Parameters: estimators, learning rate, tree depth, subsampling. Metrics: loss, accuracy, F1.|Параметры: число оценщиков, скорость обучения, глубина деревьев, подвыборка. Метрики: потери, доля верных ответов, F1.|Parámetros: estimadores, tasa de aprendizaje, profundidad y submuestreo. Métricas: pérdida, exactitud y F1.|参数：估计器数量、学习率、树深度、子采样。指标：损失、准确率和 F1。
Naive Bayes implementato con varianti gaussiana, multinomiale e Bernoulli.|Naive Bayes with Gaussian, multinomial, and Bernoulli variants.|Наивный Байес: гауссовский, мультиномиальный и Бернулли.|Naive Bayes con variantes gaussiana, multinomial y Bernoulli.|支持高斯、多项式及伯努利变体的朴素贝叶斯。
Parametri: smoothing α, modello gaussiano/multinomiale/bernoulli. Metriche: classificazione.|Parameters: smoothing α, Gaussian/multinomial/Bernoulli model. Classification metrics.|Параметры: сглаживание α, гауссовская/мультиномиальная/Бернулли модель. Метрики классификации.|Parámetros: suavizado α y modelo gaussiano/multinomial/Bernoulli. Métricas de clasificación.|参数：平滑系数 α、高斯/多项式/伯努利模型。使用分类指标。
Rete neurale implementata con backpropagation e ottimizzatore Adam predefinito.|Neural network with backpropagation and Adam as the default optimizer.|Нейронная сеть с обратным распространением и Adam по умолчанию.|Red neuronal con retropropagación y Adam como optimizador predeterminado.|使用反向传播、默认 Adam 优化器的神经网络。
La backpropagation calcola i gradienti di pesi e bias; Adam è l’ottimizzatore predefinito. Metriche: loss, accuracy o MSE e confusion matrix.|Backpropagation computes weight and bias gradients; Adam is the default optimizer. Metrics: loss, accuracy or MSE, and confusion matrix.|Обратное распространение вычисляет градиенты весов и смещений; по умолчанию используется Adam. Метрики: потери, доля верных ответов или MSE, матрица ошибок.|La retropropagación calcula gradientes de pesos y sesgos; Adam es el optimizador predeterminado. Métricas: pérdida, exactitud o MSE y matriz de confusión.|反向传播计算权重和偏置的梯度，默认使用 Adam。指标：损失、准确率或 MSE、混淆矩阵。
implementato come ottimizzatore della rete neurale.|implemented as a neural network optimizer.|реализован как оптимизатор нейронной сети.|implementado como optimizador de la red neuronal.|作为神经网络优化器实现。
Parametri: learning rate, batch size, shuffle. Ottimizza il modello neurale selezionato.|Parameters: learning rate, batch size, shuffle. Optimizes the selected neural model.|Параметры: скорость обучения, размер пакета, перемешивание. Оптимизирует выбранную нейронную модель.|Parámetros: tasa de aprendizaje, tamaño de lote y mezcla. Optimiza el modelo neuronal seleccionado.|参数：学习率、批次大小、打乱顺序。优化所选神经网络模型。
Parametri: learning rate α, momentum μ, batch size.|Parameters: learning rate α, momentum μ, batch size.|Параметры: скорость обучения α, импульс μ, размер пакета.|Parámetros: tasa de aprendizaje α, momento μ y tamaño de lote.|参数：学习率 α、动量 μ、批次大小。
Nesterov implementato con gradiente nella posizione anticipata.|Nesterov with a gradient at the look-ahead position.|Нестеров с градиентом в упреждающей позиции.|Nesterov con gradiente en la posición anticipada.|在前瞻位置计算梯度的 Nesterov。
Valuta il gradiente al look-ahead θ+v e aggiorna la velocità. Parametri: α, μ.|Evaluates the gradient at θ+v and updates velocity. Parameters: α, μ.|Вычисляет градиент в θ+v и обновляет скорость. Параметры: α, μ.|Evalúa el gradiente en θ+v y actualiza la velocidad. Parámetros: α y μ.|在前瞻位置 θ+v 计算梯度并更新速度。参数：α、μ。
Learning rate adattivo per parametro. Parametri: α, epsilon.|Adaptive learning rate per parameter. Parameters: α, epsilon.|Адаптивная скорость обучения для каждого параметра. Параметры: α, epsilon.|Tasa de aprendizaje adaptativa por parámetro. Parámetros: α y epsilon.|逐参数自适应学习率。参数：α、epsilon。
Parametri: α, decay ρ, epsilon.|Parameters: α, decay ρ, epsilon.|Параметры: α, затухание ρ, epsilon.|Parámetros: α, decaimiento ρ y epsilon.|参数：α、衰减系数 ρ、epsilon。
Parametri: α, β₁, β₂, epsilon. m̂ e v̂ sono corretti per il bias iniziale.|Parameters: α, β₁, β₂, epsilon. m̂ and v̂ are corrected for initial bias.|Параметры: α, β₁, β₂, epsilon. m̂ и v̂ скорректированы по начальному смещению.|Parámetros: α, β₁, β₂ y epsilon. m̂ y v̂ se corrigen por el sesgo inicial.|参数：α、β₁、β₂、epsilon。m̂ 和 v̂ 经过初始偏差校正。
AdamW implementato con weight decay disaccoppiato.|AdamW with decoupled weight decay.|AdamW с отделённым затуханием весов.|AdamW con decaimiento de pesos desacoplado.|使用解耦权重衰减的 AdamW。
Weight decay disaccoppiato. Parametri: α, β₁, β₂, epsilon, λ.|Decoupled weight decay. Parameters: α, β₁, β₂, epsilon, λ.|Отделённое затухание весов. Параметры: α, β₁, β₂, epsilon, λ.|Decaimiento de pesos desacoplado. Parámetros: α, β₁, β₂, epsilon y λ.|解耦权重衰减。参数：α、β₁、β₂、epsilon、λ。
Implementato nel motore base come clustering.|Implemented as a clustering method.|Реализован как метод кластеризации.|Implementado como método de agrupamiento.|已实现的聚类方法。
Parametri: numero cluster k, inizializzazione, iterazioni. Metriche: inertia e silhouette (se calcolata).|Parameters: clusters k, initialization, iterations. Metrics: inertia and silhouette when computed.|Параметры: число кластеров k, инициализация, итерации. Метрики: инерция и силуэт, если вычисляется.|Parámetros: grupos k, inicialización e iteraciones. Métricas: inercia y silueta si se calcula.|参数：簇数 k、初始化、迭代次数。指标：惯性及可计算时的轮廓系数。
DBSCAN implementato con espansione dei cluster per densità.|DBSCAN with density-based cluster expansion.|DBSCAN с расширением кластеров по плотности.|DBSCAN con expansión de grupos por densidad.|基于密度扩展簇的 DBSCAN。
Parametri: epsilon, minPts, metrica. Metriche: silhouette sui punti assegnati, numero cluster e rumore.|Parameters: epsilon, minPts, distance metric. Metrics: silhouette of assigned points, cluster count, noise.|Параметры: epsilon, minPts, метрика. Метрики: силуэт назначенных точек, число кластеров и шум.|Parámetros: epsilon, minPts y métrica. Métricas: silueta de puntos asignados, número de grupos y ruido.|参数：epsilon、minPts、距离度量。指标：已分配点的轮廓系数、簇数和噪声。
Gaussian Mixture Model implementato con algoritmo EM.|Gaussian mixture model using EM.|Гауссова смесь с алгоритмом EM.|Mezcla gaussiana con el algoritmo EM.|使用 EM 算法的高斯混合模型。
Parametri: componenti, covarianza, regolarizzazione, iterazioni. Metriche: log-likelihood, AIC/BIC.|Parameters: components, covariance, regularization, iterations. Metrics: log-likelihood, AIC/BIC.|Параметры: компоненты, ковариация, регуляризация, итерации. Метрики: логарифм правдоподобия, AIC/BIC.|Parámetros: componentes, covarianza, regularización e iteraciones. Métricas: log-verosimilitud y AIC/BIC.|参数：分量、协方差、正则化、迭代次数。指标：对数似然、AIC/BIC。
Campione e parametri|Sample and parameters|Объект и параметры|Muestra y parámetros|样本与参数
Costruzione del vettore di feature|Constructing the feature vector|Построение вектора признаков|Construcción del vector de características|构建特征向量
Prodotto pesi × feature e previsione|Weights × features and prediction|Веса × признаки и прогноз|Pesos × características y predicción|权重 × 特征与预测
Loss, regolarizzazione e gradiente|Loss, regularization, and gradient|Потери, регуляризация и градиент|Pérdida, regularización y gradiente|损失、正则化与梯度
Distanza dalla frontiera|Distance to the boundary|Расстояние до границы|Distancia a la frontera|到边界的距离
Prodotto scalare per ogni classe|Dot product for each class|Скалярное произведение для каждого класса|Producto escalar por clase|各类别的点积
Aggiornamento Perceptron|Perceptron update|Обновление перцептрона|Actualización del perceptrón|感知机更新
Matrice dei pesi × input|Weight matrix × input|Матрица весов × вход|Matriz de pesos × entrada|权重矩阵 × 输入
Errore e gradiente|Error and gradient|Ошибка и градиент|Error y gradiente|误差与梯度
Distanze dal campione|Distances from the sample|Расстояния от объекта|Distancias a la muestra|到样本的距离
Voto dei k vicini|Vote of k neighbors|Голосование k соседей|Voto de los k vecinos|k 个近邻投票
Kernel tra support vector e campione|Kernel between support vectors and sample|Ядро между опорными векторами и объектом|Kernel entre vectores de soporte y muestra|支持向量与样本间的核值
Somma dei contributi e decisione|Sum of contributions and decision|Сумма вкладов и решение|Suma de contribuciones y decisión|贡献之和与决策
Margine e perdita|Margin and loss|Отступ и потери|Margen y pérdida|间隔与损失
Percorso degli split|Split path|Путь разбиений|Ruta de divisiones|分裂路径
Foglia e previsione|Leaf and prediction|Лист и прогноз|Hoja y predicción|叶节点与预测
Predizione dei singoli alberi|Individual tree predictions|Прогнозы отдельных деревьев|Predicciones de cada árbol|各棵树的预测
Voto aggregato|Aggregated vote|Итоговое голосование|Voto agregado|汇总投票
Valore base e correzioni|Base value and corrections|Базовое значение и поправки|Valor base y correcciones|基础值与修正
Somma progressiva|Running sum|Накопленная сумма|Suma progresiva|累积和
Punteggi per classe|Class scores|Оценки классов|Puntuaciones por clase|各类别得分
Normalizzazione e classe|Normalization and class|Нормализация и класс|Normalización y clase|归一化与类别
Statistiche per classe|Class statistics|Статистика классов|Estadísticas por clase|各类别统计
Log-probabilità e decisione|Log-probability and decision|Логарифм вероятности и решение|Log-probabilidad y decisión|对数概率与决策
Primo strato|First layer|Первый слой|Primera capa|第一层
Strato di output|Output layer|Выходной слой|Capa de salida|输出层
Backpropagation e ottimizzatore|Backpropagation and optimizer|Обратное распространение и оптимизатор|Retropropagación y optimizador|反向传播与优化器
Distanza da ogni centroide|Distance to each centroid|Расстояние до каждого центроида|Distancia a cada centroide|到各质心的距离
Assegnazione e aggiornamento centroide|Assignment and centroid update|Назначение и обновление центроида|Asignación y actualización del centroide|分配与质心更新
Regione ε e densità|ε-neighborhood and density|ε-окрестность и плотность|Vecindad ε y densidad|ε 邻域与密度
Nucleo, bordo o rumore|Core, border, or noise|Ядро, граница или шум|Núcleo, borde o ruido|核心点、边界点或噪声
Densità di ogni componente|Density of each component|Плотность каждой компоненты|Densidad de cada componente|各分量密度
E-step: responsabilità|E-step: responsibilities|E-шаг: ответственности|Paso E: responsabilidades|E 步：责任度
M-step e obiettivo|M-step and objective|M-шаг и целевая функция|Paso M y objetivo|M 步与目标
Aggiornamento realmente applicato|Actual applied update|Фактически применённое обновление|Actualización aplicada realmente|实际应用的更新
Valutazione sul set di validazione|Validation set evaluation|Оценка на проверочной выборке|Evaluación del conjunto de validación|验证集评估
Genera o carica un dataset per vedere i calcoli.|Generate or upload a dataset to see calculations.|Создайте или загрузите данные для просмотра вычислений.|Genera o carga datos para ver los cálculos.|生成或上传数据集以查看计算过程。
regolarizzazione|regularization|регуляризация|regularización|正则化
validazione|validation|проверка|validación|验证
previsione|prediction|прогноз|predicción|预测
parametri selezionati|selected parameters|выбранные параметры|parámetros elegidos|所选参数
Campione di training reale|Actual training sample|Реальный обучающий объект|Muestra real de entrenamiento|实际训练样本
nessun iperparametro aggiuntivo|no additional hyperparameters|нет дополнительных гиперпараметров|sin hiperparámetros adicionales|无额外超参数
`;
