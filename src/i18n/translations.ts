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
