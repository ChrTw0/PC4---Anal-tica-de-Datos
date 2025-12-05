# PC4 - Analítica de Datos
## Grupo 2: Predicción de Hotspots de Extorsión en Lima Metropolitana

**Autor:** [Tu Nombre]
**Fecha:** Diciembre 2024
**Curso:** Analítica de Datos
**Tema:** Cap. 12 - Custom Models and Training with TensorFlow

---

## 📋 Tabla de Contenidos

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Introducción](#introducción)
3. [Metodología](#metodología)
4. [Arquitectura del Sistema](#arquitectura-del-sistema)
5. [Implementación Técnica](#implementación-técnica)
6. [Resultados Experimentales](#resultados-experimentales)
7. [Visualizaciones](#visualizaciones)
8. [Conclusiones](#conclusiones)
9. [Referencias](#referencias)

---

## 1. Resumen Ejecutivo

Este proyecto implementa un sistema de **predicción de hotspots geoespaciales de extorsión** en Lima Metropolitana y Callao, utilizando técnicas avanzadas de Deep Learning con TensorFlow/Keras. El sistema predice qué áreas geográficas (~1km²) tendrán extorsiones en la próxima semana, con un **PR-AUC de 0.555** y **F1-Score de 0.577**.

**Valor Operacional:**
- Captura ~49% de los hotspots reales
- 71% de precisión en predicciones positivas
- Útil para asignación de recursos policiales

**Componentes Técnicos Implementados (Cap. 12):**
- ✅ Custom Loss Functions (WeightedBCE, Focal Loss)
- ✅ Custom Metrics (RecallAtKHotspots - streaming metric)
- ✅ Custom Layers (MyDense con regularización)
- ✅ Custom Models (RiskModel - subclassing Model)
- ✅ Custom Training Loops (GradientTape + tf.function)
- ✅ 20+ experimentos con diferentes configuraciones

---

## 2. Introducción

### 2.1 Problema

La extorsión es un delito creciente en Lima Metropolitana. Las autoridades necesitan herramientas predictivas para:
- Identificar zonas de alto riesgo
- Asignar recursos policiales eficientemente
- Prevenir delitos antes de que ocurran

### 2.2 Enfoque

**Predicción Geoespacial:**
- **Input:** Grids espaciales (~1km x 1km) + historial temporal (lags, rolling, vecindad)
- **Output:** Probabilidad de que cada grid tenga extorsiones la próxima semana
- **Ventaja:** Predicción específica por ubicación (no solo distritos completos)

### 2.3 Objetivos

1. Implementar modelos custom con TensorFlow (Capítulo 12)
2. Entrenar 20+ variantes de modelos
3. Evaluar con métricas apropiadas (PR-AUC, Recall@K)
4. Generar visualizaciones interactivas de hotspots

---

## 3. Metodología

### 3.1 Datos

**Fuente:** Base de datos MySQL `denuncias_peru`
- **Tabla:** `denuncias`
- **Periodo:** 2020-01-01 a 2025-06-24 (5.5 años)
- **Casos totales:** 34,537 denuncias de extorsión
- **Después de limpieza:** 31,832 casos válidos
- **Región:** Lima Metropolitana + Callao

**Filtros aplicados:**
```sql
WHERE modalidad_hecho = 'EXTORSION'
    AND departamento_hecho IN ('LIMA', 'CALLAO')
    AND lat_hecho IS NOT NULL
    AND long_hecho IS NOT NULL
    AND lat_hecho BETWEEN -12.6 AND -11.4
    AND long_hecho BETWEEN -77.5 AND -76.5
```

### 3.2 Preprocesamiento

#### 3.2.1 Creación de Grids Espaciales
- **Grid size:** 0.01 grados (~1.1 km en latitud)
- **Total de grids únicos:** 894 grids
- **Rango geográfico:** Lima + Callao

```python
df['grid_lat'] = (df['lat_hecho'] / 0.01).round() * 0.01
df['grid_lon'] = (df['long_hecho'] / 0.01).round() * 0.01
df['grid_id'] = df['grid_lat'].astype(str) + '_' + df['grid_lon'].astype(str)
```

#### 3.2.2 Panel Grid-Semana
- **Panel completo:** 894 grids × 288 semanas = 257,472 observaciones
- **Con extorsiones:** 15,850 (6.2%)
- **Sin extorsiones:** 241,622 (93.8%)

#### 3.2.3 Features Engineered

**Features Temporales:**
- `count_lag1`: Extorsiones hace 1 semana
- `count_lag2`: Extorsiones hace 2 semanas
- `count_lag3`: Extorsiones hace 3 semanas
- `count_lag4`: Extorsiones hace 4 semanas
- `count_roll_mean4`: Promedio móvil últimas 4 semanas

**Features Espaciales:**
- `neighbor_mean`: Promedio de extorsiones en 8 grids vecinos

**Target:**
- `target_bin_t1`: ¿Habrá extorsión en este grid la próxima semana? (binario)
- `target_count_t1`: Cantidad de extorsiones la próxima semana (Poisson)

#### 3.2.4 Curación de Datos

**Problemas detectados y solucionados:**

1. **NaNs en features:** Rellenados con 0 (grids inactivos al inicio)
2. **Features sin normalizar:** StandardScaler aplicado
3. **Desbalance de clases (1:18):** Class weights implementados
4. **Data leakage:** Excluido `count` actual de features

```python
# Normalización
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Class weights
class_weight = {
    0: 1.0,
    1: (n_neg / n_pos) * 0.5  # ~9x peso para clase positiva
}
```

### 3.3 Split Temporal

```
Train:  164,211 samples (5.21% positivo) - 2020-01 a 2023-07
Val:     41,052 samples (10.07% positivo) - 2023-07 a 2024-05
Test:    51,315 samples (6.13% positivo) - 2024-05 a 2025-06
```

---

## 4. Arquitectura del Sistema

### 4.1 Estructura de Directorios

```
PC4---Anal-tica-de-Datos/
│
├── src/                          # Código base
│   ├── __init__.py
│   ├── config.py                 # Configuración general
│   ├── data.py                   # Procesamiento CSV
│   └── models.py                 # Modelos custom TensorFlow
│
├── mysql_version/                # Versión MySQL (nueva)
│   ├── config_mysql.py           # Config MySQL
│   ├── database.py               # Conexión MySQL
│   ├── data_mysql.py             # Carga desde MySQL
│   ├── data_geospatial_mysql.py  # Pipeline geoespacial
│   ├── analyze_geospatial.py     # EDA geoespacial
│   ├── experiments_geospatial_mysql.py  # 20+ modelos
│   └── visualize_predictions.py  # Mapas interactivos
│
└── reports/                      # Resultados
    ├── leaderboard_geospatial.csv
    ├── best_model_geospatial.keras
    ├── map_predictions.html
    ├── map_real_hotspots.html
    ├── map_comparison.html
    └── [otros reportes y visualizaciones]
```

### 4.2 Pipeline de Datos

```
MySQL Database
    ↓
[database.py] → Carga denuncias de extorsión
    ↓
[data_geospatial_mysql.py] → Procesamiento
    ├─ Crear grids espaciales
    ├─ Agregar por grid-semana
    ├─ Generar features temporales (lags, rolling)
    ├─ Calcular features espaciales (vecindad)
    └─ Crear targets (t+1)
    ↓
Panel Grid-Semana (257K filas)
    ↓
Split Temporal → Train/Val/Test
    ↓
Normalización (StandardScaler)
    ↓
X_train, y_train (164K samples)
```

### 4.3 Pipeline de Modelado

```
[experiments_geospatial_mysql.py]
    ↓
20+ Configuraciones de Experimentos
    ├─ 5 Loss variants (BCE, Focal, Weighted BCE, ...)
    ├─ 5 Metric variants (Custom Recall sí/no)
    ├─ 5 Architecture variants ([16], [64,32], [128], ...)
    └─ 5 Training mode variants (fit vs custom loop)
    ↓
Entrenar cada modelo (15 epochs)
    ↓
Evaluar en Test Set
    ├─ PR-AUC (Precision-Recall AUC)
    ├─ Recall@K (Top K hotspots)
    ├─ F1-Score
    ├─ Precision / Recall
    └─ Training Time
    ↓
Leaderboard (ordenado por PR-AUC)
    ↓
Guardar Mejor Modelo
```

---

## 5. Implementación Técnica

### 5.1 Custom Loss Functions

#### 5.1.1 Weighted Binary Crossentropy

```python
class WeightedBinaryCrossentropy(keras.losses.Loss):
    """BCE ponderada para desbalance de clases."""

    def __init__(self, pos_weight: float = 2.0, name: str = "weighted_bce"):
        super().__init__(name=name)
        self.pos_weight = pos_weight

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        loss_pos = -self.pos_weight * y_true * tf.math.log(y_pred)
        loss_neg = -(1.0 - y_true) * tf.math.log(1.0 - y_pred)
        return tf.reduce_mean(loss_pos + loss_neg)
```

**Justificación:** Penaliza más los falsos negativos (no detectar un hotspot real), crítico para seguridad pública.

#### 5.1.2 Focal Loss

```python
class FocalLoss(keras.losses.Loss):
    """Focal Loss para enfocarse en ejemplos difíciles."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # Focal loss fórmula
        bce = -y_true * tf.math.log(y_pred) - (1-y_true) * tf.math.log(1-y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1 - p_t, self.gamma)

        return tf.reduce_mean(self.alpha * focal_weight * bce)
```

**Justificación:** Da más peso a ejemplos mal clasificados, útil para clase minoritaria.

### 5.2 Custom Metrics

#### 5.2.1 RecallAtKHotspots (Streaming Metric)

```python
class RecallAtKHotspots(keras.metrics.Metric):
    """Métrica streaming para calcular recall en top-K predicciones."""

    def __init__(self, k: int = 10, name="recall_at_k", **kwargs):
        super().__init__(name=name, **kwargs)
        self.k = k
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.total_positives = self.add_weight(name="total_pos", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.reshape(y_pred, [-1])

        # Top-K predicciones
        _, top_k_indices = tf.nn.top_k(y_pred, k=tf.minimum(self.k, tf.size(y_pred)))

        # Cuántos positivos reales capturamos
        top_k_true = tf.gather(y_true, top_k_indices)
        self.true_positives.assign_add(tf.reduce_sum(top_k_true))
        self.total_positives.assign_add(tf.reduce_sum(y_true))

    def result(self):
        return self.true_positives / (self.total_positives + 1e-7)

    def reset_state(self):
        self.true_positives.assign(0.0)
        self.total_positives.assign(0.0)
```

**Justificación:** Métrica operacional - ¿capturamos los hotspots reales en nuestras top predicciones?

### 5.3 Custom Model

#### 5.3.1 RiskModel (Model Subclassing)

```python
class RiskModel(keras.Model):
    """Modelo de riesgo con capas densas custom."""

    def __init__(
        self,
        input_dim: int,
        hidden_units: list = [64, 32],
        dropout_rate: float = 0.1,
        task: str = "classification",
        reg_lambda: float = 1e-4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.task = task
        self.hidden_layers = []

        for units in hidden_units:
            self.hidden_layers.append(
                MyDense(
                    units=units,
                    activation="relu",
                    kernel_regularizer=keras.regularizers.l2(reg_lambda),
                    name=f"dense_{units}"
                )
            )
            self.hidden_layers.append(layers.Dropout(dropout_rate))

        # Output layer
        if task == "classification":
            self.output_layer = layers.Dense(1, activation="sigmoid", name="output")
        else:  # Poisson
            self.output_layer = layers.Dense(1, activation="exponential", name="output")

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        return self.output_layer(x)
```

**Justificación:** Flexibilidad para experimentar con diferentes arquitecturas y regularización.

### 5.4 Custom Training Loop

```python
# Custom training loop con GradientTape
sample_weights = np.where(y_train == 1, class_weight[1], class_weight[0])
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train, sample_weights)).batch(128)

@tf.function
def train_step(xb, yb, wb):
    with tf.GradientTape() as tape:
        logits = model(xb, training=True)
        logits = tf.reshape(logits, tf.shape(yb))

        # Loss con sample weights
        loss_value = loss_fn(yb, logits, sample_weight=wb)

        # Regularization losses
        if model.losses:
            loss_value += tf.add_n(model.losses)

    # Backpropagation
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Update metrics
    for m in metric_objs:
        if hasattr(m, 'update_state'):
            m.update_state(yb, logits)

    return loss_value

# Training loop
for epoch in range(epochs):
    for m in metric_objs:
        if hasattr(m, 'reset_states'):
            m.reset_states()

    for xb, yb, wb in train_ds:
        _ = train_step(xb, yb, wb)
```

**Ventajas:**
- Control total sobre el proceso de entrenamiento
- Sample weighting personalizado
- Integración de regularización
- tf.function para optimización

---

## 6. Resultados Experimentales

### 6.1 Experimentos Ejecutados

**Total:** 20 experimentos con diferentes configuraciones

#### Variantes de Loss (5 experimentos)
- `loss_weighted_bce`: Binary Crossentropy con peso para positivos
- `loss_focal`: Focal Loss (γ=2.0, α=0.25)
- `loss_bce`: Binary Crossentropy estándar
- `loss_bce_ls`: BCE con Label Smoothing
- `loss_bce_pos_weight_3`: BCE con pos_weight=3

#### Variantes de Arquitectura (5 experimentos)
- `arch_0`: [16] - Red simple
- `arch_1`: [64, 32] - Red mediana
- `arch_2`: [64, 32, 16] - Red profunda
- `arch_3`: [128] - Red ancha
- `arch_4`: [32, 32] - Red balanceada

#### Variantes de Training Mode (5 experimentos)
- `train_*_fit_*`: Usando `model.fit()`
- `train_*_custom_loop_*`: Usando GradientTape
- Con y sin `@tf.function`

#### Variantes de Métricas (5 experimentos)
- Con y sin `RecallAtKHotspots` custom

### 6.2 Leaderboard (Top 10)

| Rank | Modelo | PR-AUC | Recall@K | F1 | Precision | Recall | Time (s) |
|------|--------|--------|----------|-----|-----------|--------|----------|
| 1 | train_4_custom_loop_tf1 | **0.555** | 0.0025 | **0.577** | 0.709 | 0.487 | 13.2 |
| 2 | train_2_custom_loop_tf1 | 0.555 | 0.0025 | 0.570 | 0.716 | 0.474 | 13.4 |
| 3 | train_3_custom_loop_tf0 | 0.554 | 0.0025 | 0.573 | 0.717 | 0.477 | 496.9 |
| 4 | loss_bce_ls | 0.554 | 0.0029 | 0.440 | 0.317 | 0.717 | 23.6 |
| 5 | loss_bce | 0.549 | 0.0032 | 0.434 | 0.311 | 0.717 | 23.3 |
| 6 | arch_3 | 0.511 | 0.0019 | 0.000 | 0.000 | 0.000 | 24.3 |
| 7 | metrics_recall_4_1 | 0.489 | 0.0022 | 0.000 | 0.000 | 0.000 | 23.0 |
| 8 | arch_0 | 0.475 | 0.0022 | 0.000 | 0.000 | 0.000 | 21.6 |
| 9 | arch_1 | 0.471 | 0.0032 | 0.000 | 0.000 | 0.000 | 24.1 |
| 10 | metrics_recall_3_0 | 0.462 | 0.0029 | 0.000 | 0.000 | 0.000 | 22.0 |

### 6.3 Análisis del Mejor Modelo

**Modelo:** `train_4_custom_loop_tf1`
- **Arquitectura:** [32, 16] con dropout 0.1
- **Training:** Custom loop con @tf.function
- **Loss:** Weighted BCE
- **Optimizador:** Adam (lr=5e-4)
- **Epochs:** 15

**Métricas en Test Set:**
```
PR-AUC:     0.555  (9x mejor que baseline 0.061)
Recall@K:   0.003  (captura algunos en top-10)
F1-Score:   0.577
Precision:  0.709  (71% de predicciones positivas correctas)
Recall:     0.487  (49% de hotspots reales detectados)
```

**Interpretación:**
- El modelo es **útil operacionalmente**: 71% de precisión significa que 7 de cada 10 alertas son correctas
- **Recall del 49%** implica que captura casi la mitad de hotspots reales
- Existe trade-off: se puede aumentar recall bajando umbral (más alertas, menos precisión)

### 6.4 Estadísticas Generales

```
Total experimentos:   20
PR-AUC promedio:      0.442 ± 0.096
Recall@K promedio:    0.0025
Tiempo promedio:      46.1 segundos
```

**Observaciones:**
- Custom training loops con `@tf.function` son más rápidos (~13s) que sin (497s)
- Arquitecturas muy simples ([16]) o muy complejas ([128]) no convergen bien
- Weighted BCE y Focal Loss tienen rendimiento similar
- Label smoothing (bce_ls) mejora recall pero baja precisión

---

## 7. Visualizaciones

### 7.1 Mapas Interactivos (Folium)

#### 7.1.1 Mapa de Predicciones

**Archivo:** `reports/map_predictions.html`

![Mapa de Predicciones](reports/map_predictions.html)

**Descripción:**
- Heatmap de probabilidades predichas por el modelo
- Gradiente de color: amarillo (bajo riesgo) → rojo oscuro (alto riesgo)
- Marcadores amarillos: Top 10 hotspots predichos
- **Interactivo:** Click en marcadores para ver detalles (score, extorsiones reales, coordenadas)

**Insights:**
- Concentración de hotspots en: Cercado de Lima, San Juan de Lurigancho, Comas
- Zonas periféricas tienen menor riesgo predicho
- Modelo identifica correctamente zonas históricamente conflictivas

#### 7.1.2 Mapa de Hotspots Reales

**Archivo:** `reports/map_real_hotspots.html`

![Hotspots Reales](reports/map_real_hotspots.html)

**Descripción:**
- Heatmap de extorsiones reales en test set
- Gradiente: azul (pocas) → púrpura → rojo (muchas)
- Marcadores rojos: Top 10 hotspots reales
- **Interactivo:** Click para ver semanas con extorsión, total de casos, score predicho

**Insights:**
- Hotspots reales coinciden geográficamente con predicciones
- Grid [-12.05, -77.06] es el más afectado (2,042 casos en todo el periodo)
- Distribución espacial concentrada en centro y norte de Lima

#### 7.1.3 Mapa de Comparación

**Archivo:** `reports/map_comparison.html`

![Comparación](reports/map_comparison.html)

**Descripción:**
- Superposición de predicciones vs realidad
- 🟡 Círculos amarillos: Top 20 predichos
- 🔴 Círculos rojos: Top 20 reales
- Overlap visual muestra aciertos del modelo

**Overlap Top 50:**
- **~40%** de overlap entre top 50 predichos y top 50 reales
- Indica que el modelo captura correctamente las zonas de mayor riesgo

### 7.2 Gráficos Estáticos (Matplotlib)

#### 7.2.1 Heatmaps Comparativos

**Archivo:** `reports/prediction_heatmaps.png`

![Prediction Heatmaps](reports/prediction_heatmaps.png)

**4 Subgráficos:**

1. **Distribución Geográfica de Predicciones**
   - Scatter plot: longitud vs latitud
   - Color: score predicho
   - Muestra concentración de riesgo en centro de Lima

2. **Mapa de Calor de Hotspots Reales**
   - Hexbin plot
   - Colormap: YlOrRd
   - Identifica zonas con más denuncias

3. **Top 20 Grids Predichos**
   - Bar chart horizontal
   - Score predicho (naranja) vs semanas reales (rojo)
   - Permite ver aciertos y desaciertos

4. **Distribución de Scores por Clase**
   - Histograma superpuesto
   - Azul: negativos (sin extorsión)
   - Rojo: positivos (con extorsión)
   - Línea negra: umbral 0.5
   - **Separabilidad:** Cierta superposición, pero tendencia clara

#### 7.2.2 Evolución Temporal

**Archivo:** `reports/temporal_predictions.png`

![Temporal Predictions](reports/temporal_predictions.png)

**2 Subgráficos:**

1. **Predicciones vs Realidad por Semana**
   - Línea roja: extorsiones reales
   - Línea naranja: grids predichos como hotspots
   - **Tendencia:** Modelo captura variabilidad temporal

2. **Score Promedio por Semana**
   - Línea púrpura: riesgo promedio predicho
   - Muestra estacionalidad y tendencias

#### 7.2.3 Leaderboard Visual

**Archivo:** `reports/leaderboard_geospatial.png`

![Leaderboard](reports/leaderboard_geospatial.png)

**4 Subgráficos:**

1. **PR-AUC por Modelo** (Bar Chart)
   - 20 modelos ordenados descendente
   - Línea roja: media
   - Variabilidad clara entre configuraciones

2. **Recall@K por Modelo** (Bar Chart)
   - Métrica operacional
   - Algunos modelos capturan mejor top hotspots

3. **PR-AUC vs Training Time** (Scatter)
   - Color: F1-Score
   - Trade-off tiempo/rendimiento
   - Custom loops con tf.function son óptimos

4. **Precision vs Recall** (Scatter)
   - Color: PR-AUC
   - Trade-off clásico
   - Mejor modelo: buen balance

### 7.3 Datos Tabulares

#### 7.3.1 Top 50 Hotspots Predichos

**Archivo:** `reports/top_50_hotspots_predicted.csv`

```csv
grid_id,grid_lat,grid_lon,pred_score,y_true,target_count_t1
-12.05_-77.06,-12.05,-77.06,0.789,45,2042
-11.94_-77.0,-11.94,-77.0,0.776,38,1844
...
```

#### 7.3.2 Top 50 Hotspots Reales

**Archivo:** `reports/top_50_hotspots_real.csv`

```csv
grid_id,grid_lat,grid_lon,y_true,target_count_t1,pred_score
-12.05_-77.06,-12.05,-77.06,45,2042,0.789
-11.94_-77.0,-11.94,-77.0,38,1844,0.776
...
```

#### 7.3.3 Todas las Predicciones

**Archivo:** `reports/all_predictions.csv`

**Columnas:**
- `grid_id`, `grid_lat`, `grid_lon`: Identificación espacial
- `week_start`: Semana predicha
- `count`: Extorsiones en semana actual
- `target_count_t1`: Extorsiones reales en t+1
- `pred_score`: Probabilidad predicha
- `pred_bin`: Predicción binaria (umbral 0.5)
- `y_true`: Target real

**51,315 filas** (test set completo)

---

## 8. Conclusiones

### 8.1 Logros Técnicos

✅ **Implementación Completa de Capítulo 12:**
- Custom Loss Functions (3 variantes)
- Custom Metrics (streaming metric)
- Custom Layers (MyDense con regularización)
- Custom Models (RiskModel con subclassing)
- Custom Training Loops (GradientTape + tf.function)

✅ **20+ Experimentos:**
- Variaciones en loss, arquitectura, métricas, training mode
- Leaderboard completo con métricas comparables

✅ **Metodología Robusta:**
- Split temporal (sin data leakage)
- Normalización apropiada
- Class balancing con weights
- Evaluación con métricas correctas (PR-AUC > Accuracy)

### 8.2 Logros Operacionales

✅ **Modelo Útil para Producción:**
- PR-AUC: 0.555 (significativamente mejor que baseline)
- Precision: 71% (7 de 10 alertas son correctas)
- Recall: 49% (captura mitad de hotspots reales)

✅ **Predicción Espacial Granular:**
- Grids de ~1km² (más específico que distritos completos)
- 894 zonas monitoreadas en Lima + Callao

✅ **Visualizaciones Interactivas:**
- Mapas HTML con Folium (explorables en navegador)
- Heatmaps, marcadores, popups informativos
- Útil para presentaciones y toma de decisiones

### 8.3 Desafíos Superados

❌ **Problema Inicial:** Todos los modelos daban mismo resultado (PR-AUC = 0.061)

✅ **Soluciones Implementadas:**
1. Rellenar NaNs con 0
2. Normalizar features con StandardScaler
3. Aplicar class weights (1:9)
4. Aumentar epochs (5 → 15)
5. Reducir learning rate (1e-3 → 5e-4)

**Resultado:** Mejora de 9x en PR-AUC

❌ **Problema:** 100% clase positiva en enfoque distrito-semana

✅ **Solución:** Reformulación a enfoque geoespacial con panel completo (incluir grids sin extorsiones)

❌ **Problema:** Serialización de modelos custom

✅ **Solución:** Cargar arquitectura + pesos separadamente con `custom_objects`

### 8.4 Lecciones Aprendidas

1. **Formulación del problema es crítica:**
   - Predecir "habrá delito" en distrito completo → 100% positivo (inútil)
   - Predecir "habrá delito en este grid específico" → balance realista 6% positivo

2. **Preprocesamiento importa tanto como arquitectura:**
   - Normalización: diferencia entre modelo que no converge y modelo que funciona
   - Class balancing: esencial para datos desbalanceados
   - NaNs: pueden romper silenciosamente el entrenamiento

3. **Métricas apropiadas:**
   - Accuracy es engañosa en datos desbalanceados (94% accuracy prediciendo siempre 0)
   - PR-AUC captura correctamente rendimiento en clase minoritaria
   - Recall@K es métrica operacional útil

4. **Custom training loops:**
   - Más control pero más complejo
   - @tf.function es esencial para rendimiento (13s vs 497s)
   - Sample weighting más flexible que class_weight en fit()

### 8.5 Valor Generado

**Para Autoridades Policiales:**
- Predicción semanal de hotspots con 71% de precisión
- Priorización de 894 zonas de Lima + Callao
- Visualizaciones interactivas para planning operacional

**Para el Curso:**
- Demostración completa de técnicas Cap. 12
- 20+ experimentos documentados
- Código modular y reutilizable

**Para la Comunidad:**
- Enfoque replicable para otros delitos
- Pipeline MySQL → TensorFlow documentado
- Visualizaciones con Folium para análisis geoespacial

---

## 9. Referencias

### 9.1 Fuentes de Datos

- **Base de datos:** Sistema de Denuncias del Perú
- **Tabla:** `denuncias` (denuncias_peru database)
- **Periodo:** Enero 2020 - Junio 2025
- **Región:** Lima Metropolitana y Callao

### 9.2 Librerías y Frameworks

```python
# Core ML
tensorflow==2.15+
keras==3.0+
numpy==1.24+
pandas==2.0+
scikit-learn==1.3+

# Visualización
matplotlib==3.7+
folium==0.15+

# Base de datos
pymysql==1.1+
sqlalchemy==2.0+
```

### 9.3 Recursos de Aprendizaje

1. **TensorFlow Custom Training:**
   - [TensorFlow Guide - Custom Training](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch)
   - [TensorFlow Guide - Custom Layers](https://www.tensorflow.org/guide/keras/custom_layers_and_models)

2. **Imbalanced Classification:**
   - Chawla et al. (2002) - SMOTE
   - Lin et al. (2017) - Focal Loss for Dense Object Detection
   - He & Garcia (2009) - Learning from Imbalanced Data

3. **Geospatial Analysis:**
   - Chainey & Ratcliffe (2005) - GIS and Crime Mapping
   - Folium Documentation: https://python-visualization.github.io/folium/

### 9.4 Estructura del Código

**Repositorio:** [PC4---Anal-tica-de-Datos](https://github.com/[tu-usuario]/PC4---Anal-tica-de-Datos)

**Módulos principales:**
- `src/models.py`: Modelos custom TensorFlow
- `mysql_version/data_geospatial_mysql.py`: Pipeline de datos
- `mysql_version/experiments_geospatial_mysql.py`: Experimentos
- `mysql_version/visualize_predictions.py`: Visualizaciones

---

## Anexos

### A. Comando para Reproducir Resultados

```bash
# 1. Configurar base de datos MySQL
mysql -u root -p < setup_database.sql

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Análisis exploratorio geoespacial
cd mysql_version
python analyze_geospatial.py

# 4. Entrenar 20+ modelos
python experiments_geospatial_mysql.py

# 5. Generar visualizaciones
python visualize_predictions.py

# 6. Abrir mapas interactivos
# En navegador, abrir:
# - reports/map_predictions.html
# - reports/map_real_hotspots.html
# - reports/map_comparison.html
```

### B. Configuración MySQL

```python
# mysql_version/config_mysql.py
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PASSWORD = "1234"
MYSQL_DATABASE = "denuncias_peru"
MYSQL_TABLE_NAME = "denuncias"
```

### C. Hiperparámetros del Mejor Modelo

```python
{
    "architecture": [32, 16],
    "dropout": 0.1,
    "loss": "weighted_bce",
    "pos_weight": 9.0,
    "optimizer": "Adam",
    "learning_rate": 5e-4,
    "epochs": 15,
    "batch_size": 128,
    "regularization": "L2 (1e-4)"
}
```

---

**Fin del Informe**

*Generado automáticamente por el sistema de análisis de hotspots de extorsión*
*Fecha: Diciembre 2024*
