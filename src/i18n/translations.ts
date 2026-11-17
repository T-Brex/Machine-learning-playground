import type { Language } from "./Language.tsx";
import { learningRows } from "./learningTranslations.ts";
import { glossaryRows } from "./glossaryTranslations.ts";

// Source text stays Italian. Each row supplies English, Russian, Spanish, Chinese.
const rows = `
Lingua|Language|Язык|Idioma|语言
Sperimenta · Addestra · Visualizza|Experiment · Train · Visualize|Эксперименты · Обучение · Графики|Experimenta · Entrena · Visualiza|实验 · 训练 · 可视化
Laboratorio interattivo di Machine Learning|Interactive machine learning laboratory|Интерактивная лаборатория машинного обучения|Laboratorio interactivo de aprendizaje automático|交互式机器学习实验室
Navigazione principale|Main navigation|Основная навигация|Navegación principal|主导航
Panoramica|Overview|Обзор|Resumen|概览
Apprendimento|Training|Обучение|Entrenamiento|训练
Visualizzazioni|Visualizations|Визуализации|Visualizaciones|可视化
Formule e metriche|Formulas and metrics|Формулы и метрики|Fórmulas y métricas|公式与指标
Guida didattica|Learning guide|Учебное руководство|Guía didáctica|学习指南
Come iniziare|Getting started|Начало работы|Cómo empezar|开始使用
Scegli i dati, configura un algoritmo e avvia l’apprendimento. Osserva pesi, bias e grafici.|Choose data, configure an algorithm, and start training. Inspect weights, biases, and charts.|Выберите данные, настройте алгоритм и начните обучение. Изучайте веса, смещения и графики.|Elige los datos, configura un algoritmo e inicia el entrenamiento. Observa pesos, sesgos y gráficos.|选择数据，配置算法并开始训练。观察权重、偏置和图表。
Il modello modifica i parametri per ridurre una funzione di costo.|The model adjusts its parameters to minimize a cost function.|Модель изменяет параметры, чтобы уменьшить функцию потерь.|El modelo ajusta sus parámetros para minimizar una función de coste.|模型通过调整参数来减小损失函数。
Tipo di problema|Problem type|Тип задачи|Tipo de problema|问题类型
Classificazione|Classification|Классификация|Clasificación|分类
Regressione|Regression|Регрессия|Regresión|回归
Clustering|Clustering|Кластеризация|Agrupamiento|聚类
Ottimizzatori|Optimizers|Оптимизаторы|Optimizadores|优化器
ML Playground · laboratorio didattico locale. Gli algoritmi funzionano nel browser: nessun dato viene inviato a un server.|ML Playground · local learning laboratory. Algorithms run in your browser; no data is sent to a server.|ML Playground · локальная учебная лаборатория. Алгоритмы работают в браузере; данные не отправляются на сервер.|ML Playground · laboratorio didáctico local. Los algoritmos funcionan en tu navegador; no se envían datos a un servidor.|ML Playground · 本地学习实验室。算法在浏览器中运行，数据不会发送至服务器。
Colori|Colors|Цвета|Colores|颜色
Personalizza colori|Customize colors|Настроить цвета|Personalizar colores|自定义颜色
Primario|Primary|Основной|Primario|主色
Evidenziazione|Highlight|Акцент|Resaltado|强调色
Colore primario|Primary color|Основной цвет|Color primario|主色
Colore evidenziazione|Highlight color|Акцентный цвет|Color de resaltado|强调色
Ripristina colori|Reset colors|Сбросить цвета|Restablecer colores|重置颜色
Controlli simulazione|Simulation controls|Управление симуляцией|Controles de simulación|模拟控制
Avvia / Continua|Start / Resume|Начать / Продолжить|Iniciar / Continuar|开始 / 继续
Riavvia simulazione|Restart simulation|Перезапустить симуляцию|Reiniciar simulación|重新开始模拟
Un passo alla volta|Single step|Один шаг|Un paso|单步运行
Pausa|Pause|Пауза|Pausa|暂停
Preparazione solver…|Preparing solver…|Подготовка решателя…|Preparando el motor…|正在准备求解器…
Training in corso|Training in progress|Идёт обучение|Entrenamiento en curso|训练中
Simulazione completata|Simulation complete|Симуляция завершена|Simulación completada|模拟完成
In pausa|Paused|Приостановлено|En pausa|已暂停
Pronto|Ready|Готово|Listo|就绪
passo|step|шаг|paso|步
Passo|Step|Шаг|Paso|步骤
Dataset|Dataset|Данные|Datos|数据集
Dataset di esempio|Example dataset|Пример набора данных|Datos de ejemplo|示例数据集
Numero di classi|Number of classes|Число классов|Número de clases|类别数量
CSV importato · ultima colonna = target|Imported CSV · last column = target|CSV загружен · последний столбец = целевая переменная|CSV importado · última columna = objetivo|已导入 CSV · 最后一列为目标值
Campioni|Samples|Объекты|Muestras|样本
campioni|samples|объектов|muestras|样本
Campione|Sample|Объект|Muestra|样本
Feature|Features|Признаки|Características|特征
feature|features|признаков|características|特征
classi|classes|классов|clases|类别
I modelli usano tutte le feature. La vista spaziale proietta le prime due o tre.|Models use all features. The spatial view projects the first two or three.|Модели используют все признаки. Пространственный график отображает первые два или три.|Los modelos usan todas las características. La vista espacial proyecta las primeras dos o tres.|模型使用所有特征。空间视图展示前两个或前三个特征。
Rumore dei dati|Data noise|Шум данных|Ruido de los datos|数据噪声
Rigenera dataset|Regenerate dataset|Создать данные заново|Regenerar datos|重新生成数据集
Carica CSV (opzionale)|Upload CSV (optional)|Загрузить CSV (необязательно)|Cargar CSV (opcional)|上传 CSV（可选）
Almeno due feature + target numerico. Intestazione facoltativa; separatori: virgola, punto e virgola, tab.|At least two features and a numeric target. Optional header; separators: comma, semicolon, or tab.|Не менее двух признаков и числовая целевая переменная. Заголовок необязателен; разделители: запятая, точка с запятой или табуляция.|Al menos dos características y un objetivo numérico. Cabecera opcional; separadores: coma, punto y coma o tabulador.|至少两个特征和一个数值目标。表头可选；支持逗号、分号或制表符分隔。
Classificazione binaria lineare|Linear binary classification|Линейная бинарная классификация|Clasificación binaria lineal|线性二分类
Settori lineari · multiclasse|Linear sectors · multiclass|Линейные секторы · несколько классов|Sectores lineales · multiclase|线性分区 · 多分类
Cerchi concentrici|Concentric circles|Концентрические окружности|Círculos concéntricos|同心圆
Due lune|Two moons|Две луны|Dos lunas|双月
Spirali intrecciate|Intertwined spirals|Переплетённые спирали|Espirales entrelazadas|交错螺旋
Due gruppi gaussiani|Two Gaussian clusters|Два гауссовых кластера|Dos grupos gaussianos|两个高斯簇
Gruppi gaussiani multiclasse|Multiclass Gaussian clusters|Гауссовы кластеры нескольких классов|Grupos gaussianos multiclase|多类别高斯簇
Una classe · nuvola singola|One class · single cloud|Один класс · одно облако|Una clase · una nube|单类别 · 单点云
Lineare|Linear|Линейная|Lineal|线性
Quadratica|Quadratic|Квадратичная|Cuadrática|二次
Sinusoidale|Sinusoidal|Синусоидальная|Sinusoidal|正弦
Con interazione|With interactions|С взаимодействиями|Con interacción|含交互项
Configurazione modello|Model configuration|Настройка модели|Configuración del modelo|模型配置
Algoritmo|Algorithm|Алгоритм|Algoritmo|算法
Regressione logistica binaria|Binary logistic regression|Бинарная логистическая регрессия|Regresión logística binaria|二元逻辑回归
Regressione softmax|Softmax regression|Softmax-регрессия|Regresión softmax|Softmax 回归
Regressione lineare|Linear regression|Линейная регрессия|Regresión lineal|线性回归
Learning rate α|Learning rate α|Скорость обучения α|Tasa de aprendizaje α|学习率 α
Learning rate|Learning rate|Скорость обучения|Tasa de aprendizaje|学习率
Epoche|Epochs|Эпохи|Épocas|轮数
Batch size|Batch size|Размер пакета|Tamaño de lote|批次大小
Full batch|Full batch|Полный пакет|Lote completo|全批次
Mini-batch|Mini-batch|Мини-пакет|Minilote|小批次
Regolarizzazione λ|Regularization λ|Регуляризация λ|Regularización λ|正则化 λ
Soglia|Threshold|Порог|Umbral|阈值
Split training / validazione|Training / validation split|Разбиение обучение / проверка|División entrenamiento / validación|训练 / 验证划分
Reset parametri|Reset parameters|Сбросить параметры|Restablecer parámetros|重置参数
Output layer|Output layer|Выходной слой|Capa de salida|输出层
Automatico dal dataset|Automatic from dataset|Автоматически по данным|Automático según los datos|根据数据集自动选择
Sigmoid · binario|Sigmoid · binary|Сигмоида · два класса|Sigmoide · binario|Sigmoid · 二分类
Softmax · multiclasse|Softmax · multiclass|Softmax · несколько классов|Softmax · multiclase|Softmax · 多分类
Grado feature (auto)|Feature degree (auto)|Степень признаков (авто)|Grado de características (auto)|特征阶数（自动）
Numero vicini k|Number of neighbors k|Число соседей k|Número de vecinos k|近邻数 k
Distanza|Distance|Расстояние|Distancia|距离
Pesi vicini|Neighbor weights|Веса соседей|Pesos de vecinos|近邻权重
Tolleranza|Tolerance|Допуск|Tolerancia|容差
Iterazioni max|Maximum iterations|Максимум итераций|Iteraciones máximas|最大迭代次数
Grado polinomiale|Polynomial degree|Степень полинома|Grado polinómico|多项式阶数
Criterio|Criterion|Критерий|Criterio|准则
Profondità max|Maximum depth|Максимальная глубина|Profundidad máxima|最大深度
Min campioni split|Minimum split samples|Минимум объектов для разделения|Muestras mínimas para dividir|分裂所需最少样本
Min campioni foglia|Minimum leaf samples|Минимум объектов в листе|Muestras mínimas por hoja|叶节点最少样本
Numero alberi|Number of trees|Число деревьев|Número de árboles|树的数量
Feature per split|Features per split|Признаков на разделение|Características por división|每次分裂的特征数
Numero stimatori|Number of estimators|Число оценщиков|Número de estimadores|估计器数量
Profondità base learner|Base learner depth|Глубина базовой модели|Profundidad del modelo base|基学习器深度
Variante|Variant|Вариант|Variante|变体
Neuroni hidden|Hidden neurons|Скрытые нейроны|Neuronas ocultas|隐藏神经元
Numero cluster k|Number of clusters k|Число кластеров k|Número de grupos k|簇数 k
Iterazioni|Iterations|Итерации|Iteraciones|迭代次数
Inizializzazione|Initialization|Инициализация|Inicialización|初始化
Numero componenti|Number of components|Число компонент|Número de componentes|分量数
Covarianza|Covariance|Ковариация|Covarianza|协方差
Min campioni|Minimum samples|Минимум объектов|Muestras mínimas|最少样本
Dati e previsione|Data and predictions|Данные и прогнозы|Datos y predicciones|数据与预测
Dati e previsione del modello|Model data and predictions|Данные и прогнозы модели|Datos y predicciones del modelo|模型数据与预测
Confini e opacità|Boundaries and opacity|Границы и прозрачность|Fronteras y opacidad|边界与不透明度
Opacità|Opacity|Непрозрачность|Opacidad|不透明度
Visibilità dei confini|Boundary visibility|Видимость границ|Visibilidad de fronteras|边界可见性
Confine|Boundary|Граница|Frontera|边界
Punti del dataset|Dataset points|Точки данных|Puntos del conjunto|数据点
Mostra tutti i punti|Show all points|Показать все точки|Mostrar todos los puntos|显示全部点
Mostra punti|Show points|Показать точки|Mostrar puntos|显示点
Visibilità dei gruppi di punti|Point group visibility|Видимость групп точек|Visibilidad de grupos de puntos|点组可见性
Spazio classificato|Classified space|Классифицированное пространство|Espacio clasificado|分类空间
Evidenzia spazio classificato|Highlight classified space|Выделить классифицированное пространство|Resaltar espacio clasificado|突出分类空间
Evidenzia lo spazio previsto per classe|Highlight predicted class regions|Выделить области предсказанных классов|Resaltar regiones de clases predichas|突出显示预测类别区域
Opacità dello spazio|Region opacity|Непрозрачность областей|Opacidad de regiones|区域不透明度
Opacità spazio classificato|Classified space opacity|Непрозрачность классифицированного пространства|Opacidad del espacio clasificado|分类空间不透明度
Classe|Class|Класс|Clase|类别
Rumore|Noise|Шум|Ruido|噪声
rumore|noise|шум|ruido|噪声
Mostra spazio classe|Show class region|Показать область класса|Mostrar región de clase|显示类别区域
Mostra punti classe|Show class points|Показать точки класса|Mostrar puntos de clase|显示类别点
Visibilità delle regioni classificate|Classified region visibility|Видимость классифицированных областей|Visibilidad de regiones clasificadas|分类区域可见性
Calcolo delle regioni…|Computing regions…|Вычисление областей…|Calculando regiones…|正在计算区域…
Lo spazio dei modelli non lineari è mostrato con un’approssimazione campionata.|Nonlinear model regions use a sampled approximation.|Области нелинейных моделей отображаются с помощью выборочной аппроксимации.|Las regiones de modelos no lineales usan una aproximación por muestreo.|非线性模型区域采用采样近似显示。
Ripristina vista|Reset view|Сбросить вид|Restablecer vista|重置视图
Riduci zoom|Zoom out|Уменьшить|Alejar|缩小
Aumenta zoom|Zoom in|Увеличить|Acercar|放大
Ctrl + rotellina / pinch: zoom · trascina|Ctrl + scroll / pinch: zoom · drag|Ctrl + прокрутка / щипок: масштаб · перетаскивание|Ctrl + rueda / pellizco: zoom · arrastra|Ctrl + 滚轮 / 双指缩放 · 拖动
per ruotare · Shift: sposta|to rotate · Shift: pan|для вращения · Shift: перемещение|para girar · Mayús: desplazar|旋转 · Shift：平移
: sposta|: pan|: перемещение|: desplazar|：平移
Punti = esempi · sfondo = regioni previste dal modello.|Points = samples · background = predicted model regions.|Точки = объекты · фон = предсказанные области модели.|Puntos = muestras · fondo = regiones predichas por el modelo.|点代表样本，背景代表模型预测区域。
Punti = target osservati · superficie = previsione al target mediano. Assi = prime tre feature; le altre sono fissate alle medie del dataset.|Points = observed targets · surface = prediction at the median target. Axes show the first three features; others are fixed at their dataset means.|Точки = наблюдаемые значения · поверхность = прогноз на медианном уровне. Оси — первые три признака; остальные равны средним по данным.|Puntos = objetivos observados · superficie = predicción al objetivo mediano. Los ejes muestran las primeras tres características; las demás se fijan en sus medias.|点表示观测目标值，曲面表示目标中位数处的预测。坐标轴为前三个特征，其余固定为均值。
Colori = cluster assegnati · croci = centroidi. Assi = prime tre feature.|Colors = assigned clusters · crosses = centroids. Axes show the first three features.|Цвета = назначенные кластеры · кресты = центроиды. Оси — первые три признака.|Colores = grupos asignados · cruces = centroides. Los ejes muestran las primeras tres características.|颜色表示所属簇，十字表示质心。坐标轴为前三个特征。
Superfici = confini di decisione. Lo spazio evidenziato mostra la classe prevista. Assi = prime tre feature; le altre sono fissate alle medie del dataset.|Surfaces = decision boundaries. Highlighted space shows the predicted class. Axes show the first three features; others are fixed at their dataset means.|Поверхности = границы решений. Выделенное пространство показывает предсказанный класс. Оси — первые три признака; остальные равны средним по данным.|Superficies = fronteras de decisión. El espacio resaltado muestra la clase predicha. Los ejes son las primeras tres características; las demás se fijan en sus medias.|曲面表示决策边界，高亮区域表示预测类别。坐标轴为前三个特征，其余固定为均值。
Superficie di previsione|Prediction surface|Поверхность прогноза|Superficie de predicción|预测曲面
Loss per epoca|Loss by epoch|Потери по эпохам|Pérdida por época|每轮损失
Loss e prestazione per epoca|Loss and performance by epoch|Потери и качество по эпохам|Pérdida y rendimiento por época|每轮损失与性能
Loss finale|Final loss|Итоговые потери|Pérdida final|最终损失
Passi|Steps|Шаги|Pasos|步数
Distribuzione della loss|Loss distribution|Распределение потерь|Distribución de la pérdida|损失分布
Training loss|Training loss|Потери обучения|Pérdida de entrenamiento|训练损失
Training|Training|Обучение|Entrenamiento|训练
Validazione|Validation|Проверка|Validación|验证
train / validazione|training / validation|обучение / проверка|entrenamiento / validación|训练 / 验证
Purezza|Purity|Чистота|Pureza|纯度
Accuracy|Accuracy|Точность|Exactitud|准确率
Epoca|Epoch|Эпоха|Época|轮次
Numero di epoche|Number of epochs|Число эпох|Número de épocas|轮数
Avvia il training per vedere le curve|Start training to see the curves|Начните обучение, чтобы увидеть кривые|Inicia el entrenamiento para ver las curvas|开始训练以查看曲线
R² può essere negativo quando il modello predice peggio della media.|R² can be negative when the model predicts worse than the mean.|R² может быть отрицательным, если модель предсказывает хуже среднего.|R² puede ser negativo si el modelo predice peor que la media.|当模型预测不如均值基线时，R² 可以为负值。
Le etichette sono usate solo per valutare la purezza. Con una sola classe etichettata, questa metrica non è informativa.|Labels are used only to evaluate purity. With only one labeled class, this metric is not informative.|Метки используются только для оценки чистоты. При одном размеченном классе эта метрика неинформативна.|Las etiquetas solo se usan para evaluar la pureza. Con una sola clase etiquetada, esta métrica no es informativa.|标签仅用于评估纯度。只有一个标注类别时，该指标不具参考价值。
Percentuale di previsioni corrette sui set separati di training e validazione.|Percentage of correct predictions on separate training and validation sets.|Доля правильных прогнозов на отдельных обучающей и проверочной выборках.|Porcentaje de predicciones correctas en conjuntos separados de entrenamiento y validación.|独立训练集和验证集上的正确预测比例。
Parametri del modello|Model parameters|Параметры модели|Parámetros del modelo|模型参数
Parametro|Parameter|Параметр|Parámetro|参数
Valore|Value|Значение|Valor|值
Avvia il training per apprendere i parametri.|Start training to learn the parameters.|Начните обучение, чтобы определить параметры.|Inicia el entrenamiento para aprender los parámetros.|开始训练以学习参数。
Valori correnti · fino a 200 parametri mostrati.|Current values · up to 200 parameters shown.|Текущие значения · показано до 200 параметров.|Valores actuales · se muestran hasta 200 parámetros.|当前值 · 最多显示 200 个参数。
Campioni del dataset|Dataset samples|Объекты набора данных|Muestras del conjunto|数据集样本
Target y|Target y|Целевая переменная y|Objetivo y|目标 y
Predizione|Prediction|Прогноз|Predicción|预测
Primi 12 campioni. La predizione usa tutte le feature.|First 12 samples. Predictions use all features.|Первые 12 объектов. Прогноз использует все признаки.|Primeras 12 muestras. La predicción usa todas las características.|前 12 个样本。预测使用所有特征。
Prova un input|Try an input|Проверить входные данные|Probar una entrada|尝试输入
Modello non addestrato|Model not trained|Модель не обучена|Modelo sin entrenar|模型尚未训练
· previsione aggiornata in tempo reale usando tutte le|· prediction updated live using all|· прогноз обновляется в реальном времени с использованием всех|· predicción actualizada en tiempo real usando las|· 实时预测，使用全部
feature.|features.|признаков.|características.|特征。
Etichetta originale:|Original label:|Исходная метка:|Etiqueta original:|原始标签：
Rete neurale · attivazioni e backpropagation|Neural network · activations and backpropagation|Нейронная сеть · активации и обратное распространение|Red neuronal · activaciones y retropropagación|神经网络 · 激活与反向传播
Struttura della rete neurale e pesi|Neural network structure and weights|Структура нейронной сети и веса|Estructura de la red neuronal y pesos|神经网络结构与权重
Blu: pesi positivi · arancione: pesi negativi · spessore proporzionale al valore assoluto. Ogni nodo mostra l’attivazione.|Blue: positive weights · orange: negative weights · thickness is proportional to magnitude. Each node shows its activation.|Синий: положительные веса · оранжевый: отрицательные · толщина пропорциональна модулю. Узлы показывают активацию.|Azul: pesos positivos · naranja: negativos · grosor proporcional a la magnitud. Cada nodo muestra su activación.|蓝色为正权重，橙色为负权重，线宽与绝对值成正比。节点显示激活值。
La superficie loss e i gradienti dettagliati si aggiornano in pausa o al termine del training|The loss surface and detailed gradients update when paused or when training finishes|Поверхность потерь и подробные градиенты обновляются при паузе или после обучения|La superficie de pérdida y los gradientes detallados se actualizan en pausa o al finalizar el entrenamiento|损失曲面和详细梯度在暂停或训练结束时更新
ultimo passo|last step|последний шаг|último paso|上一步
Superficie loss · due pesi della rete|Loss surface · two network weights|Поверхность потерь · два веса сети|Superficie de pérdida · dos pesos de la red|损失曲面 · 两个网络权重
Superficie loss attorno ai due primi pesi|Loss surface around the first two weights|Поверхность потерь около первых двух весов|Superficie de pérdida alrededor de los dos primeros pesos|前两个权重附近的损失曲面
Sezione della loss sul training, con gli altri parametri fissi. Bianco = pesi correnti. Blu = loss minore, arancione = loss maggiore.|Training loss slice with other parameters fixed. White = current weights. Blue = lower loss, orange = higher loss.|Срез обучающих потерь при фиксированных остальных параметрах. Белый = текущие веса. Синий = меньшие потери, оранжевый = большие.|Sección de la pérdida de entrenamiento con los demás parámetros fijos. Blanco = pesos actuales. Azul = pérdida menor, naranja = mayor.|固定其他参数后的训练损失切片。白色为当前权重，蓝色损失较低，橙色较高。
Pesi e bias · gradienti|Weights and biases · gradients|Веса и смещения · градиенты|Pesos y sesgos · gradientes|权重与偏置 · 梯度
Primo peso W1[0,0]|First weight W1[0,0]|Первый вес W1[0,0]|Primer peso W1[0,0]|第一个权重 W1[0,0]
Deuxième poids W1[0,1]|Second weight W1[0,1]|Второй вес W1[0,1]|Segundo peso W1[0,1]|第二个权重 W1[0,1]
Metodo di apprendimento · formule e valutazioni|Learning method · formulas and evaluation|Метод обучения · формулы и оценка|Método de aprendizaje · fórmulas y evaluación|学习方法 · 公式与评估
Formula della previsione / modello|Prediction / model formula|Формула прогноза / модели|Fórmula de predicción / modelo|预测 / 模型公式
Funzione obiettivo / costo|Objective / cost function|Целевая функция / функция потерь|Función objetivo / coste|目标 / 损失函数
Aggiornamento dei parametri|Parameter update|Обновление параметров|Actualización de parámetros|参数更新
Legenda delle formule · simboli e funzione|Formula glossary · symbols and meaning|Обозначения формул · символы и значения|Leyenda de fórmulas · símbolos y significado|公式说明 · 符号与含义
Metriche sul set di validazione|Validation set metrics|Метрики проверочной выборки|Métricas del conjunto de validación|验证集指标
Avvia il training per calcolare le metriche sul set separato di validazione.|Start training to calculate metrics on the held-out validation set.|Начните обучение для расчёта метрик на отдельной проверочной выборке.|Inicia el entrenamiento para calcular métricas en el conjunto de validación independiente.|开始训练以计算独立验证集上的指标。
Guida didattica · percorso e calcoli passo passo|Learning guide · walkthrough and step-by-step calculations|Учебное руководство · пошаговые вычисления|Guía didáctica · recorrido y cálculos paso a paso|学习指南 · 流程与逐步计算
Calcolo completo|Full calculation|Полный расчёт|Cálculo completo|完整计算
← Indietro|← Back|← Назад|← Atrás|← 上一步
Avanti →|Next →|Далее →|Siguiente →|下一步 →
Esegui un passo reale nel playground|Run one training step|Выполнить один шаг обучения|Ejecutar un paso de entrenamiento|运行一个训练步骤
Cerca un campione per numero|Find a sample by number|Найти объект по номеру|Buscar una muestra por número|按编号查找样本
Scegli un campione valido.|Choose a valid sample.|Выберите допустимый объект.|Elige una muestra válida.|请选择有效样本。
Distanza / residuo prima|Distance / residual before|Расстояние / остаток до|Distancia / residuo antes|更新前距离 / 残差
Distanza / residuo dopo|Distance / residual after|Расстояние / остаток после|Distancia / residuo después|更新后距离 / 残差
Variazione|Change|Изменение|Cambio|变化
Calcolo numerico completo|Full numerical calculation|Полный численный расчёт|Cálculo numérico completo|完整数值计算
CNN · MNIST|CNN · MNIST|CNN · MNIST|CNN · MNIST|CNN · MNIST
CNN · laboratorio MNIST|CNN · MNIST lab|CNN · лаборатория MNIST|CNN · laboratorio MNIST|CNN · MNIST 实验室
Caricamento laboratorio CNN…|Loading CNN lab…|Загрузка лаборатории CNN…|Cargando el laboratorio CNN…|正在加载 CNN 实验室…
Esplora una CNN reale: immagine 28×28, convoluzioni, ReLU, max-pooling, appiattimento e classificazione. I filtri iniziano casuali e cambiano con la backpropagation.|Explore a real CNN: 28×28 image, convolutions, ReLU, max-pooling, flattening, and classification. Filters start randomly and change through backpropagation.|Изучите настоящую CNN: изображение 28×28, свёртки, ReLU, max-pooling, развёртывание и классификация. Фильтры инициализируются случайно и меняются при обратном распространении ошибки.|Explora una CNN real: imagen 28×28, convoluciones, ReLU, max-pooling, aplanamiento y clasificación. Los filtros comienzan aleatorios y cambian con la retropropagación.|探索真实 CNN：28×28 图像、卷积、ReLU、最大池化、展平和分类。滤波器随机初始化并通过反向传播更新。
Carica MNIST|Load MNIST|Загрузить MNIST|Cargar MNIST|加载 MNIST
Ricarica MNIST|Reload MNIST|Перезагрузить MNIST|Recargar MNIST|重新加载 MNIST
Download MNIST in corso…|Downloading MNIST…|Загрузка MNIST…|Descargando MNIST…|正在下载 MNIST…
Scarica il training set MNIST ufficiale (circa 10 MB, solo download).|Download the official MNIST training set (about 10 MB; download only).|Загрузить официальный обучающий набор MNIST (около 10 МБ; только загрузка).|Descarga el conjunto de entrenamiento MNIST oficial (unos 10 MB; solo descarga).|下载官方 MNIST 训练集（约 10 MB，仅下载）。
Download e decompressione dei file MNIST…|Downloading and decompressing MNIST files…|Загрузка и распаковка файлов MNIST…|Descargando y descomprimiendo los archivos MNIST…|正在下载并解压 MNIST 文件…
Fonte dati: MNIST training set, Yann LeCun et al., mirror compresso IDX nel repository GitHub fgnt/mnist. Il browser scarica i dati; nessuna immagine viene caricata o inviata.|Data source: MNIST training set by Yann LeCun et al., compressed IDX mirror in the fgnt/mnist GitHub repository. The browser downloads the data; no images are uploaded or sent.|Источник данных: обучающий набор MNIST Яна Лекуна и соавторов, сжатое зеркало IDX в репозитории GitHub fgnt/mnist. Браузер загружает данные; изображения никуда не отправляются.|Fuente de datos: conjunto de entrenamiento MNIST de Yann LeCun et al., espejo IDX comprimido en el repositorio GitHub fgnt/mnist. El navegador descarga los datos; no se suben ni envían imágenes.|数据来源：Yann LeCun 等人的 MNIST 训练集，来自 GitHub 仓库 fgnt/mnist 的压缩 IDX 镜像。浏览器下载数据；不会上传或发送图像。
immagini bilanciate dal training set MNIST|balanced images from the MNIST training set|сбалансированных изображений из обучающего набора MNIST|imágenes equilibradas del conjunto de entrenamiento MNIST|张来自 MNIST 训练集的均衡图像
Controlli CNN|CNN controls|Управление CNN|Controles de CNN|CNN 控制
Avvia training|Start training|Начать обучение|Iniciar entrenamiento|开始训练
Ricomincia training|Restart training|Перезапустить обучение|Reiniciar entrenamiento|重新开始训练
Un mini-batch|One mini-batch|Один мини-пакет|Un minilote|一个小批次
Reinizializza filtri|Reinitialize filters|Переинициализировать фильтры|Reinicializar filtros|重新初始化滤波器
Learning rate|Learning rate|Скорость обучения|Tasa de aprendizaje|学习率
mini-batch|mini-batch|мини-пакет|minilote|小批次
loss|loss|потери|pérdida|损失
accuracy|accuracy|точность|exactitud|准确率
Immagine da ispezionare|Image to inspect|Изображение для анализа|Imagen para inspeccionar|要检查的图像
etichetta reale|true label|истинная метка|etiqueta real|真实标签
Etichetta MNIST|MNIST label|Метка MNIST|Etiqueta MNIST|MNIST 标签
1 · Immagine di input|1 · Input image|1 · Входное изображение|1 · Imagen de entrada|1 · 输入图像
28×28 pixel in scala di grigi · intensità normalizzata tra 0 e 1.|28×28 grayscale pixels · intensity normalized between 0 and 1.|28×28 пикселей в оттенках серого · интенсивность нормализована от 0 до 1.|28×28 píxeles en escala de grises · intensidad normalizada entre 0 y 1.|28×28 灰度像素 · 强度归一化至 0 到 1。
2 · Filtri creati dalla rete|2 · Filters created by the network|2 · Фильтры, созданные сетью|2 · Filtros creados por la red|2 · 网络创建的滤波器
Quattro kernel 3×3 inizializzati casualmente; i valori si aggiornano con il gradiente.|Four randomly initialized 3×3 kernels; their values update with the gradient.|Четыре случайно инициализированных ядра 3×3; их значения обновляются градиентом.|Cuatro kernels 3×3 inicializados aleatoriamente; sus valores se actualizan con el gradiente.|四个随机初始化的 3×3 卷积核；其数值随梯度更新。
3 · Convoluzione|3 · Convolution|3 · Свёртка|3 · Convolución|3 · 卷积
Il kernel scorre sull’immagine: prodotto elemento per elemento e somma, con stride 1. Risultato: 26×26 per filtro.|The kernel slides over the image: element-wise products are summed with stride 1. Result: 26×26 per filter.|Ядро скользит по изображению: сумма поэлементных произведений с шагом 1. Результат: 26×26 для каждого фильтра.|El kernel se desplaza por la imagen: suma de productos elemento a elemento con stride 1. Resultado: 26×26 por filtro.|卷积核以步幅 1 在图像上滑动：逐元素相乘后求和。每个滤波器得到 26×26 结果。
4 · ReLU|4 · ReLU|4 · ReLU|4 · ReLU|4 · ReLU
Attivazione elemento per elemento: max(0, x). I valori negativi diventano zero.|Element-wise activation: max(0, x). Negative values become zero.|Поэлементная активация: max(0, x). Отрицательные значения становятся нулём.|Activación elemento a elemento: max(0, x). Los valores negativos pasan a cero.|逐元素激活：max(0, x)。负值变为零。
5 · Max-pooling|5 · Max-pooling|5 · Max-pooling|5 · Max-pooling|5 · 最大池化
Massimo in ogni finestra 2×2, stride 2: riduce ogni mappa da 26×26 a 13×13.|Maximum in each 2×2 window, stride 2: reduces each map from 26×26 to 13×13.|Максимум в каждом окне 2×2 с шагом 2: уменьшает карту с 26×26 до 13×13.|Máximo en cada ventana 2×2, stride 2: reduce cada mapa de 26×26 a 13×13.|每个 2×2 窗口取最大值，步幅为 2：将每张特征图从 26×26 缩小到 13×13。
6 · Flatten e classificazione|6 · Flatten and classification|6 · Развёртывание и классификация|6 · Aplanamiento y clasificación|6 · 展平与分类
Le 4 mappe 13×13 diventano 676 valori. Un livello denso produce 10 logits, trasformati in probabilità dalla softmax.|The four 13×13 maps become 676 values. A dense layer produces 10 logits, converted to probabilities by softmax.|Четыре карты 13×13 преобразуются в 676 значений. Полносвязный слой формирует 10 логитов, которые softmax преобразует в вероятности.|Los cuatro mapas de 13×13 se convierten en 676 valores. Una capa densa genera 10 logits, que softmax convierte en probabilidades.|四张 13×13 特征图展平为 676 个值。全连接层生成 10 个 logits，softmax 将其转换为概率。
Feature map|Feature map|Карта признаков|Mapa de características|特征图
Predizione|Prediction|Предсказание|Predicción|预测
Logit|Logit|Логит|Logit|Logit
Calcolo dettagliato del pixel selezionato|Detailed calculation for the selected pixel|Подробный расчёт выбранного пикселя|Cálculo detallado del píxel seleccionado|所选像素的详细计算
Clicca un pixel in una mappa, oppure usa i controlli per selezionare filtro e coordinate. La selezione è collegata tra gli stadi.|Click a pixel in a map, or use the controls to select the filter and coordinates. The selection is linked across stages.|Нажмите на пиксель карты или выберите фильтр и координаты с помощью элементов управления. Выбор связан между этапами.|Haz clic en un píxel del mapa o usa los controles para elegir el filtro y las coordenadas. La selección se mantiene entre etapas.|点击特征图中的像素，或使用控件选择滤波器和坐标。所选像素会在各阶段关联显示。
Mappa da ispezionare|Map to inspect|Карта для анализа|Mapa para inspeccionar|要检查的特征图
Convoluzione|Convolution|Свёртка|Convolución|卷积
Riga|Row|Строка|Fila|行
Colonna|Column|Столбец|Columna|列
Coordinate selezionate|Selected coordinates|Выбранные координаты|Coordenadas seleccionadas|所选坐标
indici da 0|zero-based indices|индексация с нуля|índices desde 0|从 0 开始的索引
Pixel convoluzione collegato|Linked convolution pixel|Связанный пиксель свёртки|Píxel de convolución vinculado|关联的卷积像素
Patch immagine normalizzata (3×3)|Normalized image patch (3×3)|Нормализованный фрагмент изображения (3×3)|Parche de imagen normalizado (3×3)|归一化图像块（3×3）
Pesi del filtro (3×3)|Filter weights (3×3)|Веса фильтра (3×3)|Pesos del filtro (3×3)|滤波器权重（3×3）
Prodotti elemento per elemento|Element-wise products|Поэлементные произведения|Productos elemento a elemento|逐元素乘积
Somma dei 9 prodotti|Sum of the 9 products|Сумма 9 произведений|Suma de los 9 productos|9 个乘积之和
Bias del filtro|Filter bias|Смещение фильтра|Sesgo del filtro|滤波器偏置
Convoluzione: somma + bias|Convolution: sum + bias|Свёртка: сумма + смещение|Convolución: suma + sesgo|卷积：总和 + 偏置
ReLU: max(0, convoluzione)|ReLU: max(0, convolution)|ReLU: max(0, свёртка)|ReLU: max(0, convolución)|ReLU：max(0, 卷积值)
finestra|window|окно|ventana|窗口
coordinate|coordinates|координаты|coordenadas|坐标
Valore max-pooling|Max-pooling value|Значение max-pooling|Valor de max-pooling|max-pooling 值
massimo selezionato|selected maximum|выбранный максимум|máximo seleccionado|选中的最大值
Esplora tutti gli stadi di una CNN su MNIST: due blocchi convoluzionali, ReLU, pooling, flatten, neuroni dense e softmax. I filtri e i pesi si aggiornano con la backpropagation.|Explore every stage of a CNN on MNIST: two convolution blocks, ReLU, pooling, flattening, dense neurons, and softmax. Filters and weights update through backpropagation.|Изучите все этапы CNN на MNIST: два блока свёртки, ReLU, pooling, развёртывание, полносвязные нейроны и softmax. Фильтры и веса обновляются обратным распространением ошибки.|Explora todas las etapas de una CNN en MNIST: dos bloques convolucionales, ReLU, pooling, aplanamiento, neuronas densas y softmax. Los filtros y pesos se actualizan mediante retropropagación.|探索 MNIST CNN 的所有阶段：两个卷积块、ReLU、池化、展平、全连接神经元和 softmax。滤波器和权重通过反向传播更新。
2 · Convolution layer 1|2 · Convolution layer 1|2 · Слой свёртки 1|2 · Capa convolucional 1|2 · 卷积层 1
3 · ReLU layer 1|3 · ReLU layer 1|3 · Слой ReLU 1|3 · ReLU 1|3 · ReLU 层 1
4 · Pooling layer 1|4 · Pooling layer 1|4 · Слой pooling 1|4 · Capa de pooling 1|4 · 池化层 1
5 · Repetition of feature extraction|5 · Repetition of feature extraction|5 · Повторное извлечение признаков|5 · Repetición de extracción de características|5 · 重复特征提取
Convolution layer 1|Convolution layer 1|Слой свёртки 1|Capa convolucional 1|卷积层 1
Convolution layer 2|Convolution layer 2|Слой свёртки 2|Capa convolucional 2|卷积层 2
ReLU layer 2|ReLU layer 2|Слой ReLU 2|ReLU 2|ReLU 层 2
Pooling layer 2|Pooling layer 2|Слой pooling 2|Capa de pooling 2|池化层 2
6 · Flattening layer|6 · Flattening layer|6 · Слой развёртывания|6 · Capa de aplanamiento|6 · 展平层
7 · Fully connected (dense) layer|7 · Fully connected (dense) layer|7 · Полносвязный слой|7 · Capa totalmente conectada (densa)|7 · 全连接层
Ogni pixel in uscita è la somma dei 9 prodotti tra patch 3×3 e kernel, più il bias.|Each output pixel is the sum of the 9 products between a 3×3 patch and kernel, plus the bias.|Каждый выходной пиксель — сумма 9 произведений значений фрагмента 3×3 и ядра плюс смещение.|Cada píxel de salida es la suma de los 9 productos entre un parche 3×3 y el kernel, más el sesgo.|每个输出像素都是 3×3 图像块与卷积核之间 9 个乘积之和，再加偏置。
Il secondo blocco ripete convoluzione, ReLU e pooling sulle feature del primo blocco.|The second block repeats convolution, ReLU, and pooling on the first block's features.|Второй блок повторяет свёртку, ReLU и pooling признаков первого блока.|El segundo bloque repite la convolución, ReLU y pooling sobre las características del primer bloque.|第二个块对第一个块的特征重复卷积、ReLU 和池化。
Quattro filtri di uscita; ciascuno combina le quattro mappe del primo pooling con 4 kernel 3×3, producendo mappe 11×11.|Four output filters; each combines the four maps from the first pooling layer with four 3×3 kernels, producing 11×11 maps.|Четыре выходных фильтра; каждый объединяет четыре карты первого pooling четырьмя ядрами 3×3, создавая карты 11×11.|Cuatro filtros de salida; cada uno combina los cuatro mapas del primer pooling con cuatro kernels 3×3 y genera mapas de 11×11.|四个输出滤波器；每个滤波器用四个 3×3 卷积核组合第一个池化层的四张特征图，生成 11×11 特征图。
Finestre 2×2 valide con stride 2: mappe 11×11 → 5×5; l’ultima riga e colonna non entrano in una finestra.|Valid 2×2 windows with stride 2: maps 11×11 → 5×5; the last row and column do not fit in a window.|Допустимые окна 2×2 с шагом 2: карты 11×11 → 5×5; последняя строка и столбец не попадают в окно.|Ventanas válidas de 2×2 con stride 2: mapas 11×11 → 5×5; la última fila y columna quedan fuera de las ventanas.|步幅为 2 的有效 2×2 窗口：特征图从 11×11 变为 5×5；最后一行和一列不在窗口内。
Media delle finestre 2×2 valide con stride 2: mappe 11×11 → 5×5; l’ultima riga e colonna non entrano in una finestra.|Average of valid 2×2 windows with stride 2: maps 11×11 → 5×5; the last row and column do not fit in a window.|Среднее допустимых окон 2×2 с шагом 2: карты 11×11 → 5×5; последняя строка и столбец не попадают в окно.|Promedio de ventanas válidas de 2×2 con stride 2: mapas 11×11 → 5×5; la última fila y columna quedan fuera de las ventanas.|步幅为 2 的有效 2×2 窗口取平均：特征图从 11×11 变为 5×5；最后一行和一列不在窗口内。
Le 4 mappe 5×5 sono appiattite in un vettore ordinato di 100 valori, pronto per i livelli densi.|The four 5×5 maps are flattened into an ordered vector of 100 values, ready for the dense layers.|Четыре карты 5×5 преобразуются в упорядоченный вектор из 100 значений для полносвязных слоёв.|Los cuatro mapas de 5×5 se aplanan en un vector ordenado de 100 valores para las capas densas.|四张 5×5 特征图展平为包含 100 个值的有序向量，供全连接层使用。
Dimensione vettore|Vector size|Размер вектора|Tamaño del vector|向量大小
Ogni neurone calcola la somma pesata dei 100 ingressi più il bias, poi applica ReLU.|Each neuron calculates the weighted sum of 100 inputs plus the bias, then applies ReLU.|Каждый нейрон вычисляет взвешенную сумму 100 входов со смещением, затем применяет ReLU.|Cada neurona calcula la suma ponderada de 100 entradas más el sesgo y aplica ReLU.|每个神经元计算 100 个输入的加权和加偏置，然后应用 ReLU。
Neurone hidden da ispezionare|Hidden neuron to inspect|Скрытый нейрон для анализа|Neurona oculta para inspeccionar|要检查的隐藏神经元
Mostra i 100 prodotti, la somma e il bias di questo neurone|Show this neuron's 100 products, sum, and bias|Показать 100 произведений, сумму и смещение этого нейрона|Mostrar los 100 productos, la suma y el sesgo de esta neurona|显示该神经元的 100 个乘积、总和和偏置
Output layer · Softmax|Output layer · Softmax|Выходной слой · Softmax|Capa de salida · Softmax|输出层 · Softmax
8 · Output layer · Softmax|8 · Output layer · Softmax|8 · Выходной слой · Softmax|8 · Capa de salida · Softmax|8 · 输出层 · Softmax
Dieci neuroni calcolano i logits dai 16 valori dense; softmax li converte in probabilità che sommano a 1.|Ten neurons calculate logits from the 16 dense values; softmax converts them into probabilities that sum to 1.|Десять нейронов вычисляют логиты по 16 значениям плотного слоя; softmax преобразует их в вероятности с суммой 1.|Diez neuronas calculan los logits a partir de los 16 valores densos; softmax los convierte en probabilidades que suman 1.|十个神经元根据 16 个全连接值计算 logits；softmax 将它们转换为总和为 1 的概率。
Classe di output da ispezionare|Output class to inspect|Класс выходного слоя для анализа|Clase de salida para inspeccionar|要检查的输出类别
Mostra i 16 prodotti, la somma e il bias per il logit|Show the 16 products, sum, and bias for the logit|Показать 16 произведений, сумму и смещение логита|Mostrar los 16 productos, la suma y el sesgo del logit|显示 logit 的 16 个乘积、总和和偏置
Softmax|Softmax|Softmax|Softmax|Softmax
somma|sum|сумма|suma|总和
Denominatore softmax|Softmax denominator|Знаменатель softmax|Denominador de softmax|Softmax 分母
Classe|Class|Класс|Clase|类别
Esponenziale normalizzato|Shifted exponential|Экспонента со сдвигом|Exponencial desplazada|平移后的指数值
Probabilità|Probability|Вероятность|Probabilidad|概率
Tipo di pooling|Pooling type|Тип pooling|Tipo de pooling|池化类型
Max pooling|Max pooling|Max pooling|Max pooling|最大池化
Average pooling|Average pooling|Average pooling|Average pooling|平均池化
Pooling|Pooling|Pooling|Pooling|池化
Mappa max-pooling|Max-pooling map|Карта max-pooling|Mapa de max-pooling|max-pooling 特征图
Mappa average pooling|Average-pooling map|Карта average pooling|Mapa de average pooling|平均池化特征图
Mappa di convoluzione layer 2|Convolution map layer 2|Карта свёртки слоя 2|Mapa de convolución de la capa 2|卷积层 2 特征图
Attivazioni ReLU layer 2|ReLU activations layer 2|Активации ReLU слоя 2|Activaciones ReLU de la capa 2|ReLU 层 2 激活
Filtro, kernel 3 per 3|Filter, 3 by 3 kernel|Фильтр, ядро 3 на 3|Filtro, kernel de 3 por 3|滤波器，3×3 卷积核
Canale ingresso|Input channel|Входной канал|Canal de entrada|输入通道
Patch canale|Channel patch|Фрагмент канала|Parche del canal|通道图像块
Layer|Layer|Слой|Capa|层
Pesi del filtro (3×3)|Filter weights (3×3)|Веса фильтра (3×3)|Pesos del filtro (3×3)|滤波器权重（3×3）
Mappa di convoluzione|Convolution map|Карта свёртки|Mapa de convolución|卷积特征图
Attivazioni ReLU|ReLU activations|Активации ReLU|Activaciones ReLU|ReLU 激活
Vettore flatten, 100 valori|Flatten vector, 100 values|Вектор развёртывания, 100 значений|Vector aplanado, 100 valores|展平向量，共 100 个值
Il cambio del tipo di pooling reinizializza i pesi.|Changing the pooling type reinitializes the weights.|При смене типа pooling веса инициализируются заново.|Al cambiar el tipo de pooling se reinicializan los pesos.|更改池化类型会重新初始化权重。
Pooling layer 1|Pooling layer 1|Слой pooling 1|Capa de pooling 1|池化层 1
Media dei 4 valori in ogni finestra 2×2, stride 2: riduce ogni mappa da 26×26 a 13×13.|Average of the 4 values in each 2×2 window, stride 2: reduces each map from 26×26 to 13×13.|Среднее 4 значений в каждом окне 2×2 с шагом 2: уменьшает карту с 26×26 до 13×13.|Promedio de los 4 valores en cada ventana 2×2, stride 2: reduce cada mapa de 26×26 a 13×13.|每个 2×2 窗口取 4 个值的平均值，步幅为 2：将特征图从 26×26 缩小到 13×13。
Questo pixel sul bordo non rientra in una finestra pooling 2×2 valida.|This edge pixel is not covered by a valid 2×2 pooling window.|Этот граничный пиксель не попадает в допустимое окно pooling 2×2.|Este píxel del borde no pertenece a una ventana de pooling 2×2 válida.|此边缘像素不属于有效的 2×2 池化窗口。
Valore pooling|Pooling value|Значение pooling|Valor de pooling|池化值
Somma dei 36 prodotti|Sum of the 36 products|Сумма 36 произведений|Suma de los 36 productos|36 个乘积之和
Attivazione precedente|Previous activation|Предыдущая активация|Activación anterior|上一级激活
bias|bias|смещение|sesgo|偏置
`;

const catalogue = new Map<string, string[]>();
for (const row of `${rows.trim()}\n${learningRows.trim()}\n${glossaryRows.trim()}`.split(
  "\n",
)) {
  const [source, ...values] = row
    .split(/(?<!\\)\|/)
    .map((value) => value.replaceAll("\\|", "|"));
  if (values.length !== 4)
    throw new Error(`Invalid translation row: ${source}`);
  catalogue.set(source, values);
}
const escape = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const phrases = new RegExp(
  `(?<![\\p{L}\\p{N}_])(?:${[...catalogue.keys()]
    .sort((a, b) => b.length - a.length)
    .map(escape)
    .join("|")})(?![\\p{L}\\p{N}_])`,
  "gu",
);
export function translate(text: string, language: Language): string {
  if (language === "it") return text;
  const index = { en: 0, ru: 1, es: 2, zh: 3 }[language];
  const normalized = text.replace(/\s+/g, " ").trim();
  const exact = catalogue.get(normalized);
  if (exact)
    return text.match(/^\s*/)?.[0] + exact[index] + text.match(/\s*$/)?.[0];
  return text.replace(phrases, (source) => catalogue.get(source)![index]);
}
