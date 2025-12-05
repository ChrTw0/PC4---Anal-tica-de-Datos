"""Custom training loop usando datos desde MySQL.

Este script es equivalente a src/custom_loop.py pero usa data_mysql.
Demuestra custom training loop con GradientTape + tf.function.
"""

import argparse
import json
import logging
import os
import sys
import time
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
    parser = argparse.ArgumentParser(description="Custom training loop con MySQL")
    parser.add_argument(
        "--task",
        type=str,
        default=config.DEFAULT_TASK,
        choices=["classification", "poisson"]
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--compare", action="store_true", help="Comparar con model.fit()")
    return parser.parse_args()


def ensure_reports_dir():
    os.makedirs(config.REPORTS_DIR, exist_ok=True)


def run_fit_baseline(X_train, y_train, X_val, y_val, task: str, epochs: int, batch_size: int):
    """Entrena usando model.fit() como baseline."""
    logger.info("\n" + "="*60)
    logger.info("BASELINE: Entrenamiento con model.fit()")
    logger.info("="*60)

    input_dim = X_train.shape[1]
    model = models.build_risk_model(input_dim=input_dim, task=task)

    compile_kwargs = models.get_compile_kwargs(task=task, loss_variant="weighted_bce")
    if task == "poisson":
        compile_kwargs = models.get_compile_kwargs(task=task, loss_variant="poisson")

    model.compile(optimizer=keras.optimizers.Adam(1e-3), **compile_kwargs)

    start = time.time()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )
    elapsed = time.time() - start

    logger.info(f"Tiempo total: {elapsed:.2f}s")
    return model, history.history, elapsed


def run_custom_loop(X_train, y_train, X_val, y_val, task: str, epochs: int, batch_size: int):
    """Entrena usando custom training loop con GradientTape."""
    logger.info("\n" + "="*60)
    logger.info("CUSTOM LOOP: Entrenamiento con GradientTape + tf.function")
    logger.info("="*60)

    input_dim = X_train.shape[1]
    model = models.build_risk_model(input_dim=input_dim, task=task)

    compile_kwargs = models.get_compile_kwargs(task=task, loss_variant="weighted_bce")
    if task == "poisson":
        compile_kwargs = models.get_compile_kwargs(task=task, loss_variant="poisson")

    loss_fn = compile_kwargs["loss"]
    metric_objs = compile_kwargs["metrics"]

    optimizer = keras.optimizers.Adam(1e-3)

    # Listas para guardar historia
    train_loss_results = []
    val_loss_results = []
    val_main_metric_results = []

    val_main_metric_name = metric_objs[0].name if metric_objs else "metric"

    # Datasets
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(buffer_size=len(X_train), seed=config.RANDOM_SEED)
        .batch(batch_size)
    )
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

    @tf.function
    def train_step(x_batch, y_batch):
        """Paso de entrenamiento con GradientTape."""
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            logits = tf.reshape(logits, tf.shape(y_batch))
            loss_value = loss_fn(y_batch, logits)
            # Agregar pérdidas adicionales del modelo (regularización)
            if model.losses:
                loss_value += tf.add_n(model.losses)

        # Calcular gradientes y actualizar
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Actualizar métricas
        for m in metric_objs:
            m.update_state(y_batch, logits)

        return loss_value

    @tf.function
    def val_step(x_batch, y_batch):
        """Paso de validación."""
        logits = model(x_batch, training=False)
        logits = tf.reshape(logits, tf.shape(y_batch))
        loss_value = loss_fn(y_batch, logits)
        if model.losses:
            loss_value += tf.add_n(model.losses)

        for m in metric_objs:
            m.update_state(y_batch, logits)

        return loss_value

    start = time.time()

    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch+1}/{epochs}")

        # Reset métricas
        for m in metric_objs:
            m.reset_states()

        # Entrenamiento
        epoch_loss_avg = tf.keras.metrics.Mean()
        for x_batch, y_batch in train_ds:
            loss_value = train_step(x_batch, y_batch)
            epoch_loss_avg.update_state(loss_value)

        train_loss = float(epoch_loss_avg.result().numpy())
        train_loss_results.append(train_loss)

        # Validación
        for m in metric_objs:
            m.reset_states()

        val_loss_avg = tf.keras.metrics.Mean()
        for x_batch, y_batch in val_ds:
            loss_value = val_step(x_batch, y_batch)
            val_loss_avg.update_state(loss_value)

        val_loss = float(val_loss_avg.result().numpy())
        val_loss_results.append(val_loss)

        # Métrica principal
        main_metric_val = None
        if metric_objs:
            main_metric_val = float(metric_objs[0].result().numpy())
        val_main_metric_results.append(main_metric_val)

        logger.info(
            f"train_loss={train_loss:.4f} - "
            f"val_loss={val_loss:.4f} - "
            f"val_{val_main_metric_name}={main_metric_val:.4f}"
        )

    elapsed = time.time() - start
    logger.info(f"\nTiempo total: {elapsed:.2f}s")

    history = {
        "loss": train_loss_results,
        "val_loss": val_loss_results,
        f"val_{val_main_metric_name}": val_main_metric_results,
    }

    return model, history, elapsed


def plot_comparison(fit_history, custom_history, output_path: str):
    """Genera gráfico comparativo."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(fit_history["loss"], label="fit - train", marker="o")
    axes[0].plot(fit_history.get("val_loss", []), label="fit - val", marker="o")
    axes[0].plot(custom_history["loss"], label="custom - train", marker="s", linestyle="--")
    axes[0].plot(custom_history.get("val_loss", []), label="custom - val", marker="s", linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Comparison: fit() vs Custom Loop")
    axes[0].legend()
    axes[0].grid(True)

    # Tiempo
    # (Para el segundo plot, mostrar las métricas de validación principales si existen)
    val_metric_keys = [k for k in fit_history.keys() if k.startswith("val_") and k != "val_loss"]
    if val_metric_keys:
        metric_key = val_metric_keys[0]
        axes[1].plot(fit_history.get(metric_key, []), label=f"fit - {metric_key}", marker="o")

        # Buscar métrica correspondiente en custom
        custom_metric_key = [k for k in custom_history.keys() if k.startswith("val_") and k != "val_loss"]
        if custom_metric_key:
            axes[1].plot(custom_history[custom_metric_key[0]], label=f"custom - {custom_metric_key[0]}", marker="s", linestyle="--")

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Metric")
        axes[1].set_title("Validation Metric Comparison")
        axes[1].legend()
        axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Gráfico comparativo guardado en {output_path}")


def main():
    args = parse_args()

    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    ensure_reports_dir()

    logger.info("\n" + "="*60)
    logger.info("CARGA DE DATOS DESDE MySQL")
    logger.info("="*60)

    df_raw = data.load_from_mysql()

    if df_raw.empty:
        logger.error("No se pudieron cargar datos. Abortando.")
        return

    panel = data.make_panel_district_week(df_raw)
    train_df, val_df, test_df = data.train_val_test_split_time(panel)

    X_train, y_train, _ = data.get_feature_target_arrays(train_df, task=args.task)
    X_val, y_val, _ = data.get_feature_target_arrays(val_df, task=args.task)

    # Custom loop (siempre se ejecuta)
    cl_model, cl_history, cl_time = run_custom_loop(
        X_train, y_train, X_val, y_val,
        args.task, args.epochs, args.batch_size
    )

    results = {
        "custom_loop": {
            "history": cl_history,
            "time_sec": cl_time,
        }
    }

    # Baseline fit (opcional)
    if args.compare:
        fit_model, fit_history, fit_time = run_fit_baseline(
            X_train, y_train, X_val, y_val,
            args.task, args.epochs, args.batch_size
        )

        results["fit"] = {
            "history": fit_history,
            "time_sec": fit_time,
        }

        # Gráfico comparativo
        plot_path = os.path.join(config.REPORTS_DIR, "custom_vs_fit_mysql.png")
        plot_comparison(fit_history, cl_history, plot_path)

        logger.info("\n" + "="*60)
        logger.info("COMPARACIÓN DE TIEMPOS")
        logger.info("="*60)
        logger.info(f"model.fit():   {fit_time:.2f}s")
        logger.info(f"custom loop:   {cl_time:.2f}s")
        logger.info(f"Diferencia:    {abs(cl_time - fit_time):.2f}s")

    # Guardar resultados
    results_path = os.path.join(config.REPORTS_DIR, "custom_loop_mysql_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResultados guardados en {results_path}")

    # Guardar modelo del custom loop
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    model_path = os.path.join(config.REPORTS_DIR, "custom_loop_model_mysql.keras")
    cl_model.save(model_path)
    logger.info(f"Modelo guardado en {model_path}")

    logger.info("\n" + "="*60)
    logger.info("CUSTOM TRAINING LOOP COMPLETADO")
    logger.info("="*60)


if __name__ == "__main__":
    main()
