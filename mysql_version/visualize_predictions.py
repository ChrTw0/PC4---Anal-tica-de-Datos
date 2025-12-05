"""Visualización de predicciones de hotspots geoespaciales.

Este script:
1. Carga el mejor modelo entrenado
2. Hace predicciones sobre el test set
3. Genera mapas de calor comparando predicciones vs realidad
4. Identifica los top hotspots predichos y reales
5. Crea visualizaciones interactivas
"""

import logging
import os
import sys
from pathlib import Path

import folium
from folium.plugins import HeatMap, MarkerCluster
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import config_mysql as config
import data_geospatial_mysql as data_geo
from src import models  # Importar para registrar clases custom

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_best_model(input_dim=6):
    """Carga solo los pesos del mejor modelo.

    Como RiskModel tiene problemas con serialización completa,
    reconstruimos la arquitectura y cargamos solo los pesos.
    """
    model_path = os.path.join(config.REPORTS_DIR, "best_model_geospatial.keras")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado en {model_path}. Ejecuta experiments_geospatial_mysql.py primero.")

    logger.info(f"Cargando pesos desde {model_path}")

    # Leer resumen de experimentos para obtener arquitectura del mejor modelo
    summary_path = os.path.join(config.REPORTS_DIR, "experiments_geospatial_summary.json")

    if os.path.exists(summary_path):
        import json
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        logger.info(f"Mejor modelo: {summary['best_model']['name']}")

    # Leer leaderboard para obtener hiperparámetros exactos
    leaderboard_path = os.path.join(config.REPORTS_DIR, "leaderboard_geospatial.csv")
    import pandas as pd

    if os.path.exists(leaderboard_path):
        lb = pd.read_csv(leaderboard_path).sort_values('pr_auc', ascending=False)
        best_row = lb.iloc[0]

        # Extraer hiperparámetros (están como param_*)
        hidden_units = eval(best_row['param_hidden_units']) if 'param_hidden_units' in lb.columns else [32, 16]
        dropout = best_row['param_dropout'] if 'param_dropout' in lb.columns else 0.1

        logger.info(f"Reconstruyendo modelo: hidden_units={hidden_units}, dropout={dropout}")
    else:
        # Valores por defecto
        hidden_units = [32, 16]
        dropout = 0.1
        logger.warning("Usando hiperparámetros por defecto")

    # Reconstruir modelo con arquitectura del mejor
    model = models.RiskModel(
        input_dim=input_dim,
        hidden_units=hidden_units,
        dropout_rate=dropout,
        task="classification"
    )

    # Build el modelo con un batch de prueba
    model(tf.random.normal((1, input_dim)))

    # Cargar solo los pesos
    try:
        model.load_weights(model_path)
        logger.info("✓ Pesos cargados exitosamente")
    except Exception as e:
        logger.error(f"Error cargando pesos: {e}")
        logger.info("Intentando cargar modelo completo...")

        # Plan B: cargar modelo completo
        custom_objects = {
            'RiskModel': models.RiskModel,
            'WeightedBinaryCrossentropy': models.WeightedBinaryCrossentropy,
            'FocalLoss': models.FocalLoss,
            'RecallAtKHotspots': models.RecallAtKHotspots,
        }
        try:
            full_model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            # Copiar pesos
            model.set_weights(full_model.get_weights())
            logger.info("✓ Pesos copiados desde modelo completo")
        except Exception as e2:
            logger.error(f"Error en plan B: {e2}")
            raise

    return model


def prepare_test_data():
    """Prepara datos de test con información de grids."""
    logger.info("Preparando datos de test...")

    # Crear panel completo
    panel = data_geo.make_panel_grid_week()

    # Split temporal
    train_df, val_df, test_df = data_geo.train_val_test_split_time(panel)

    # Preprocesar
    X_train, y_train, feature_names, scaler = data_geo.get_feature_target_arrays(
        train_df, task="classification", fit_scaler=True
    )
    X_test, y_test, _, _ = data_geo.get_feature_target_arrays(
        test_df, task="classification", scaler=scaler, fit_scaler=False
    )

    # Mantener información de grids para visualización
    test_info = test_df[['grid_id', 'grid_lat', 'grid_lon', 'week_start', 'count', 'target_count_t1', 'target_bin_t1']].copy()

    logger.info(f"Test set: {len(X_test)} samples")
    logger.info(f"Semanas en test: {test_info['week_start'].min()} a {test_info['week_start'].max()}")

    return X_test, y_test, test_info, feature_names


def make_predictions(model, X_test, y_test, test_info):
    """Hace predicciones y las combina con información de grids."""
    logger.info("Generando predicciones...")

    # Predicciones
    y_scores = model.predict(X_test, batch_size=256, verbose=0).reshape(-1)
    y_pred_bin = (y_scores >= 0.5).astype(int)

    # Combinar con info de grids
    results = test_info.copy()
    results['pred_score'] = y_scores
    results['pred_bin'] = y_pred_bin
    results['y_true'] = y_test

    # Calcular métricas
    tp = ((y_pred_bin == 1) & (y_test == 1)).sum()
    fp = ((y_pred_bin == 1) & (y_test == 0)).sum()
    fn = ((y_pred_bin == 0) & (y_test == 1)).sum()
    tn = ((y_pred_bin == 0) & (y_test == 0)).sum()

    logger.info(f"\nMétricas en test set:")
    logger.info(f"  TP: {tp:,}  FP: {fp:,}")
    logger.info(f"  FN: {fn:,}  TN: {tn:,}")
    logger.info(f"  Precision: {tp/(tp+fp):.3f}")
    logger.info(f"  Recall:    {tp/(tp+fn):.3f}")
    logger.info(f"  F1:        {2*tp/(2*tp+fp+fn):.3f}")

    return results


def create_folium_maps(results):
    """Crea mapas interactivos con Folium."""
    logger.info("Generando mapas interactivos con Folium...")

    # Agregar por grid (promedio de scores a lo largo del tiempo)
    grid_agg = results.groupby(['grid_id', 'grid_lat', 'grid_lon']).agg({
        'pred_score': 'mean',
        'y_true': 'sum',
        'target_count_t1': 'sum',
        'count': 'sum'
    }).reset_index()

    # === MAPA 1: Heatmap de Predicciones ===
    logger.info("  Creando mapa de predicciones...")

    # Centro de Lima: Cercado de Lima
    lima_center = [-12.05, -77.03]

    map_pred = folium.Map(
        location=lima_center,
        zoom_start=11,
        tiles='OpenStreetMap'
    )

    # Agregar título
    title_html = '''
    <div style="position: fixed;
                top: 10px; left: 50px; width: 400px; height: 50px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:16px; padding: 10px">
    <b>Mapa de Predicciones - Hotspots de Extorsión Lima</b>
    </div>
    '''
    map_pred.get_root().html.add_child(folium.Element(title_html))

    # Preparar datos para heatmap (lat, lon, weight)
    heat_data_pred = [
        [row['grid_lat'], row['grid_lon'], row['pred_score']]
        for _, row in grid_agg.iterrows()
    ]

    # Agregar heatmap de predicciones
    HeatMap(
        heat_data_pred,
        min_opacity=0.3,
        max_opacity=0.8,
        radius=15,
        blur=20,
        gradient={0.4: 'yellow', 0.65: 'orange', 0.8: 'red', 1: 'darkred'}
    ).add_to(map_pred)

    # Agregar marcadores para top 10 hotspots predichos
    top_10_pred = grid_agg.nlargest(10, 'pred_score')
    for idx, row in top_10_pred.iterrows():
        folium.CircleMarker(
            location=[row['grid_lat'], row['grid_lon']],
            radius=8,
            popup=f"""
                <b>Top Hotspot Predicho</b><br>
                Score: {row['pred_score']:.3f}<br>
                Extorsiones reales: {int(row['y_true'])}<br>
                Coord: ({row['grid_lat']:.4f}, {row['grid_lon']:.4f})
            """,
            color='black',
            fill=True,
            fillColor='yellow',
            fillOpacity=0.8,
            weight=2
        ).add_to(map_pred)

    # Guardar
    output_pred = os.path.join(config.REPORTS_DIR, 'map_predictions.html')
    map_pred.save(output_pred)
    logger.info(f"    ✓ Mapa de predicciones guardado: {output_pred}")

    # === MAPA 2: Heatmap de Realidad ===
    logger.info("  Creando mapa de hotspots reales...")

    map_real = folium.Map(
        location=lima_center,
        zoom_start=11,
        tiles='OpenStreetMap'
    )

    # Título
    title_html_real = '''
    <div style="position: fixed;
                top: 10px; left: 50px; width: 400px; height: 50px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:16px; padding: 10px">
    <b>Hotspots Reales - Extorsión Lima (Test Set)</b>
    </div>
    '''
    map_real.get_root().html.add_child(folium.Element(title_html_real))

    # Preparar datos para heatmap de realidad
    heat_data_real = [
        [row['grid_lat'], row['grid_lon'], row['y_true']]
        for _, row in grid_agg.iterrows()
        if row['y_true'] > 0  # Solo grids con extorsiones
    ]

    # Agregar heatmap
    HeatMap(
        heat_data_real,
        min_opacity=0.4,
        max_opacity=0.9,
        radius=15,
        blur=20,
        gradient={0.4: 'blue', 0.65: 'purple', 0.8: 'red', 1: 'darkred'}
    ).add_to(map_real)

    # Agregar marcadores para top 10 hotspots reales
    top_10_real = grid_agg.nlargest(10, 'y_true')
    for idx, row in top_10_real.iterrows():
        folium.CircleMarker(
            location=[row['grid_lat'], row['grid_lon']],
            radius=8,
            popup=f"""
                <b>Top Hotspot Real</b><br>
                Semanas con extorsión: {int(row['y_true'])}<br>
                Total extorsiones: {int(row['target_count_t1'])}<br>
                Score predicho: {row['pred_score']:.3f}<br>
                Coord: ({row['grid_lat']:.4f}, {row['grid_lon']:.4f})
            """,
            color='black',
            fill=True,
            fillColor='red',
            fillOpacity=0.8,
            weight=2
        ).add_to(map_real)

    # Guardar
    output_real = os.path.join(config.REPORTS_DIR, 'map_real_hotspots.html')
    map_real.save(output_real)
    logger.info(f"    ✓ Mapa de realidad guardado: {output_real}")

    # === MAPA 3: Comparación (Predicciones vs Realidad) ===
    logger.info("  Creando mapa de comparación...")

    map_compare = folium.Map(
        location=lima_center,
        zoom_start=11,
        tiles='OpenStreetMap'
    )

    # Título
    title_html_comp = '''
    <div style="position: fixed;
                top: 10px; left: 50px; width: 450px; height: 70px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:14px; padding: 10px">
    <b>Comparación: Predicciones vs Realidad</b><br>
    🟡 Amarillo = Predicho alto | 🔴 Rojo = Hotspot real
    </div>
    '''
    map_compare.get_root().html.add_child(folium.Element(title_html_comp))

    # Top 20 predichos (amarillo)
    top_20_pred = grid_agg.nlargest(20, 'pred_score')
    for idx, row in top_20_pred.iterrows():
        folium.CircleMarker(
            location=[row['grid_lat'], row['grid_lon']],
            radius=6,
            popup=f"""
                <b>Predicho como Hotspot</b><br>
                Score: {row['pred_score']:.3f}<br>
                Realidad: {int(row['y_true'])} semanas con extorsión
            """,
            color='orange',
            fill=True,
            fillColor='yellow',
            fillOpacity=0.7,
            weight=2
        ).add_to(map_compare)

    # Top 20 reales (rojo)
    top_20_real = grid_agg.nlargest(20, 'y_true')
    for idx, row in top_20_real.iterrows():
        folium.CircleMarker(
            location=[row['grid_lat'], row['grid_lon']],
            radius=6,
            popup=f"""
                <b>Hotspot Real</b><br>
                Semanas: {int(row['y_true'])}<br>
                Score predicho: {row['pred_score']:.3f}
            """,
            color='darkred',
            fill=True,
            fillColor='red',
            fillOpacity=0.7,
            weight=2
        ).add_to(map_compare)

    # Guardar
    output_comp = os.path.join(config.REPORTS_DIR, 'map_comparison.html')
    map_compare.save(output_comp)
    logger.info(f"    ✓ Mapa de comparación guardado: {output_comp}")

    return grid_agg


def visualize_heatmaps(results):
    """Crea mapas de calor comparando predicciones vs realidad."""
    logger.info("Generando mapas de calor...")

    # Agregar por grid (promedio de scores a lo largo del tiempo)
    grid_agg = results.groupby(['grid_id', 'grid_lat', 'grid_lon']).agg({
        'pred_score': 'mean',
        'y_true': 'sum',
        'target_count_t1': 'sum',
        'count': 'sum'
    }).reset_index()

    # Crear figura
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # 1. Mapa de scores predichos
    ax = axes[0, 0]
    scatter = ax.scatter(
        grid_agg['grid_lon'],
        grid_agg['grid_lat'],
        c=grid_agg['pred_score'],
        s=50,
        cmap='YlOrRd',
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )
    ax.set_xlabel('Longitud')
    ax.set_ylabel('Latitud')
    ax.set_title('Mapa de Predicciones (Score Promedio por Grid)')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Probabilidad de Hotspot')

    # 2. Mapa de hotspots reales
    ax = axes[0, 1]
    scatter = ax.scatter(
        grid_agg['grid_lon'],
        grid_agg['grid_lat'],
        c=grid_agg['y_true'],
        s=50,
        cmap='Reds',
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )
    ax.set_xlabel('Longitud')
    ax.set_ylabel('Latitud')
    ax.set_title('Hotspots Reales (Semanas con Extorsión)')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Semanas con Extorsión')

    # 3. Comparación: Top 20 grids predichos
    ax = axes[1, 0]
    top_pred = grid_agg.nlargest(20, 'pred_score')
    y_pos = np.arange(len(top_pred))
    ax.barh(y_pos, top_pred['pred_score'], alpha=0.7, color='orange', label='Score Predicho')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{lat:.3f},{lon:.3f}" for lat, lon in zip(top_pred['grid_lat'], top_pred['grid_lon'])], fontsize=8)
    ax.set_xlabel('Probabilidad Predicha')
    ax.set_title('Top 20 Grids Predichos como Hotspots')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Añadir barras de extorsiones reales
    ax2 = ax.twiny()
    ax2.barh(y_pos, top_pred['y_true'], alpha=0.5, color='red', label='Semanas Reales')
    ax2.set_xlabel('Semanas con Extorsión (Real)', color='red')
    ax2.tick_params(axis='x', labelcolor='red')

    # 4. Distribución de scores
    ax = axes[1, 1]

    # Separar por clase real
    scores_pos = results[results['y_true'] == 1]['pred_score']
    scores_neg = results[results['y_true'] == 0]['pred_score']

    ax.hist(scores_neg, bins=50, alpha=0.5, label='Negativos (sin extorsión)', color='blue', density=True)
    ax.hist(scores_pos, bins=50, alpha=0.5, label='Positivos (con extorsión)', color='red', density=True)
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Umbral (0.5)')
    ax.set_xlabel('Score Predicho')
    ax.set_ylabel('Densidad')
    ax.set_title('Distribución de Scores por Clase Real')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(config.REPORTS_DIR, 'prediction_heatmaps.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Mapas guardados en: {output_path}")
    plt.close()


def analyze_temporal_predictions(results):
    """Analiza predicciones a lo largo del tiempo."""
    logger.info("Analizando predicciones temporales...")

    # Agregar por semana
    weekly = results.groupby('week_start').agg({
        'pred_score': 'mean',
        'y_true': 'sum',
        'target_count_t1': 'sum',
        'pred_bin': 'sum'
    }).reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 1. Evolución temporal de extorsiones predichas vs reales
    ax = axes[0]
    ax.plot(weekly['week_start'], weekly['target_count_t1'],
            label='Extorsiones Reales', color='red', linewidth=2, marker='o', markersize=3)
    ax.plot(weekly['week_start'], weekly['pred_bin'],
            label='Grids Predichos como Hotspots', color='orange', linewidth=2, marker='s', markersize=3)
    ax.set_xlabel('Semana')
    ax.set_ylabel('Cantidad')
    ax.set_title('Evolución Temporal: Predicciones vs Realidad')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Score promedio por semana
    ax = axes[1]
    ax.plot(weekly['week_start'], weekly['pred_score'],
            color='purple', linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Semana')
    ax.set_ylabel('Score Promedio')
    ax.set_title('Score de Riesgo Promedio por Semana')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(config.REPORTS_DIR, 'temporal_predictions.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Análisis temporal guardado en: {output_path}")
    plt.close()


def create_hotspot_ranking(results):
    """Crea ranking de hotspots predichos vs reales."""
    logger.info("Creando ranking de hotspots...")

    # Top hotspots por score predicho
    grid_agg = results.groupby(['grid_id', 'grid_lat', 'grid_lon']).agg({
        'pred_score': 'mean',
        'y_true': 'sum',
        'target_count_t1': 'sum',
    }).reset_index()

    # Top 50 predichos
    top_pred = grid_agg.nlargest(50, 'pred_score')

    # Top 50 reales
    top_real = grid_agg.nlargest(50, 'y_true')

    # Guardar rankings
    output_pred = os.path.join(config.REPORTS_DIR, 'top_50_hotspots_predicted.csv')
    output_real = os.path.join(config.REPORTS_DIR, 'top_50_hotspots_real.csv')

    top_pred.to_csv(output_pred, index=False)
    top_real.to_csv(output_real, index=False)

    logger.info(f"Rankings guardados en:")
    logger.info(f"  Predichos: {output_pred}")
    logger.info(f"  Reales:    {output_real}")

    # Calcular overlap
    pred_ids = set(top_pred['grid_id'])
    real_ids = set(top_real['grid_id'])
    overlap = len(pred_ids & real_ids)

    logger.info(f"\nOverlap en Top 50: {overlap}/50 ({overlap/50*100:.1f}%)")

    return top_pred, top_real


def main():
    logger.info("="*80)
    logger.info("VISUALIZACIÓN DE PREDICCIONES DE HOTSPOTS")
    logger.info("="*80)

    # 1. Cargar modelo
    model = load_best_model()

    # 2. Preparar datos de test
    X_test, y_test, test_info, feature_names = prepare_test_data()

    # 3. Hacer predicciones
    results = make_predictions(model, X_test, y_test, test_info)

    # 4. Crear mapas interactivos con Folium
    grid_agg = create_folium_maps(results)

    # 5. Visualizar mapas de calor (Matplotlib - estático)
    visualize_heatmaps(results)

    # 6. Análisis temporal
    analyze_temporal_predictions(results)

    # 6. Ranking de hotspots
    top_pred, top_real = create_hotspot_ranking(results)

    # 7. Guardar predicciones completas
    output_path = os.path.join(config.REPORTS_DIR, 'all_predictions.csv')
    results.to_csv(output_path, index=False)
    logger.info(f"\nPredicciones completas guardadas en: {output_path}")

    logger.info("\n" + "="*80)
    logger.info("VISUALIZACIÓN COMPLETADA")
    logger.info("="*80)
    logger.info("\nArchivos generados:")
    logger.info("\nMapas Interactivos (HTML - abre en navegador):")
    logger.info("  1. map_predictions.html - Mapa de predicciones con heatmap")
    logger.info("  2. map_real_hotspots.html - Mapa de hotspots reales")
    logger.info("  3. map_comparison.html - Comparación predicciones vs realidad")
    logger.info("\nGráficos Estáticos (PNG):")
    logger.info("  4. prediction_heatmaps.png - Mapas de calor matplotlib")
    logger.info("  5. temporal_predictions.png - Evolución temporal")
    logger.info("\nDatos (CSV):")
    logger.info("  6. top_50_hotspots_predicted.csv - Top 50 hotspots predichos")
    logger.info("  7. top_50_hotspots_real.csv - Top 50 hotspots reales")
    logger.info("  8. all_predictions.csv - Todas las predicciones")
    logger.info("="*80)
    logger.info("\n💡 Abre los archivos .html en tu navegador para explorar los mapas interactivos!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
