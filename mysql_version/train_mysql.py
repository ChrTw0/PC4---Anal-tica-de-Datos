"""Script de entrenamiento usando datos desde MySQL.

Este script es equivalente a src/train.py pero usa data_mysql en lugar de data.py.
Entrena un modelo de riesgo de delincuencia usando datos de la base MySQL.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Agregar directorios al path
sys.path.insert(0, str(Path(__file__).parent.parent))  # Para src
sys.path.insert(0, str(Path(__file__).parent))  # Para módulos mysql

import config_mysql as config
import data_mysql as data
from src import models

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento con datos MySQL")
    parser.add_argument(
        "--task",
        type=str,
        default=config.DEFAULT_TASK,
        choices=["classification", "poisson"],
        help="Tipo de tarea"
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="weighted_bce",
        choices=["weighted_bce", "focal", "bce", "bce_ls", "huber", "poisson"],
        help="Variante de loss function"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.DEFAULT_EPOCHS,
        help="Número de epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.DEFAULT_BATCH_SIZE,
        help="Tamaño de batch"
    )
    parser.add_argument(
        "--hidden_units",
        type=int,
        nargs="+",
        default=config.DEFAULT_HIDDEN_UNITS,
        help="Unidades ocultas por capa (ej: --hidden_units 64 32 16)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=config.DEFAULT_DROPOUT,
        help="Tasa de dropout"
    )
    parser.add_argument(
        "--crime_type",
        type=str,
        default=config.MYSQL_CRIME_TYPE_FILTER,
        help="Tipo de delito a analizar"
    )
    return parser.parse_args()


def ensure_reports_dir():
    """Crea el directorio de reportes si no existe."""
    os.makedirs(config.REPORTS_DIR, exist_ok=True)


def plot_training_history(history, task: str, output_path: str):
    """Genera gráficos de la historia de entrenamiento."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history['loss'], label='Train Loss')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Métrica principal
    if task == "classification":
        metric_key = 'pr_auc'
        metric_label = 'PR-AUC'
    else:
        metric_key = 'mae'
        metric_label = 'MAE'

    if metric_key in history:
        axes[1].plot(history[metric_key], label=f'Train {metric_label}')
    if f'val_{metric_key}' in history:
        axes[1].plot(history[f'val_{metric_key}'], label=f'Val {metric_label}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel(metric_label)
    axes[1].set_title(f'Training and Validation {metric_label}')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Gráficos guardados en {output_path}")


def evaluate_model(model, X_test, y_test, task: str):
    """Evalúa el modelo en el conjunto de test."""
    logger.info("\n" + "="*60)
    logger.info("EVALUACIÓN EN TEST SET")
    logger.info("="*60)

    # Predicciones
    y_pred = model.predict(X_test, batch_size=256, verbose=0).reshape(-1)

    # Métricas básicas
    if task == "classification":
        # PR-AUC
        pr_auc_metric = keras.metrics.AUC(curve="PR")
        pr_auc_metric.update_state(y_test, y_pred)
        pr_auc = float(pr_auc_metric.result().numpy())

        # Recall@K
        k = min(config.DEFAULT_K_HOTSPOTS, len(y_pred))
        topk_idx = np.argsort(-y_pred)[:k]
        recall_at_k = float(y_test[topk_idx].sum() / max(y_test.sum(), 1.0))

        # Métricas con umbral 0.5
        y_pred_bin = (y_pred >= 0.5).astype(int)
        tp = ((y_pred_bin == 1) & (y_test == 1)).sum()
        fp = ((y_pred_bin == 1) & (y_test == 0)).sum()
        fn = ((y_pred_bin == 0) & (y_test == 1)).sum()
        tn = ((y_pred_bin == 0) & (y_test == 0)).sum()

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        accuracy = (tp + tn) / len(y_test)

        logger.info(f"PR-AUC: {pr_auc:.4f}")
        logger.info(f"Recall@{k}: {recall_at_k:.4f}")
        logger.info(f"Precisión: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Casos positivos en test: {y_test.sum():.0f}/{len(y_test)}")

        return {
            "pr_auc": float(pr_auc),
            "recall_at_k": float(recall_at_k),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
        }
    else:
        # Poisson/Regresión
        mae = np.mean(np.abs(y_test - y_pred))
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)

        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"RMSE: {rmse:.4f}")

        return {
            "mae": float(mae),
            "rmse": float(rmse),
        }


def main():
    args = parse_args()

    # Configurar seed
    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # Actualizar filtro de delito si se especificó
    if args.crime_type:
        config.MYSQL_CRIME_TYPE_FILTER = args.crime_type
        logger.info(f"Filtrando por tipo de delito: {args.crime_type}")

    ensure_reports_dir()

    logger.info("\n" + "="*60)
    logger.info("CARGA DE DATOS DESDE MySQL")
    logger.info("="*60)

    # Cargar datos desde MySQL
    df_raw = data.load_from_mysql()

    if df_raw.empty:
        logger.error("No se pudieron cargar datos desde MySQL. Abortando.")
        return

    # Crear panel
    panel = data.make_panel_district_week(df_raw)

    # Split temporal
    train_df, val_df, test_df = data.train_val_test_split_time(panel)

    # Obtener arrays
    X_train, y_train, feature_cols = data.get_feature_target_arrays(train_df, task=args.task)
    X_val, y_val, _ = data.get_feature_target_arrays(val_df, task=args.task)
    X_test, y_test, _ = data.get_feature_target_arrays(test_df, task=args.task)

    logger.info("\n" + "="*60)
    logger.info("CONSTRUCCIÓN Y ENTRENAMIENTO DEL MODELO")
    logger.info("="*60)
    logger.info(f"Arquitectura: {args.hidden_units}")
    logger.info(f"Dropout: {args.dropout}")
    logger.info(f"Loss: {args.loss}")
    logger.info(f"Task: {args.task}")

    # Construir modelo
    input_dim = X_train.shape[1]
    model = models.build_risk_model(
        input_dim=input_dim,
        hidden_units=args.hidden_units,
        dropout_rate=args.dropout,
        task=args.task,
    )

    # Compilar
    compile_kwargs = models.get_compile_kwargs(
        task=args.task,
        loss_variant=args.loss,
        use_custom_recall=True
    )

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        **compile_kwargs
    )

    # Entrenar
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
    )

    # Guardar modelo
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    model_path = os.path.join(config.REPORTS_DIR, "model_mysql.keras")
    model.save(model_path)
    logger.info(f"\nModelo guardado en {model_path}")

    # Evaluar en test
    test_metrics = evaluate_model(model, X_test, y_test, args.task)

    # Guardar resultados
    results = {
        "args": vars(args),
        "config": {
            "database": config.MYSQL_DATABASE,
            "table": config.MYSQL_TABLE_NAME,
            "crime_type": config.MYSQL_CRIME_TYPE_FILTER,
        },
        "data_info": {
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "n_features": len(feature_cols),
            "features": feature_cols,
        },
        "test_metrics": test_metrics,
    }

    results_path = os.path.join(config.REPORTS_DIR, "train_mysql_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResultados guardados en {results_path}")

    # Gráficos
    plot_path = os.path.join(config.REPORTS_DIR, "train_mysql_history.png")
    plot_training_history(history.history, args.task, plot_path)

    logger.info("\n" + "="*60)
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info("="*60)


if __name__ == "__main__":
    main()
