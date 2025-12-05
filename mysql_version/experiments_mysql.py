"""Experimentos con 20+ modelos usando datos desde MySQL.

Este script es equivalente a src/experiments_20_models.py pero usa data_mysql.
Genera un leaderboard completo con variantes de loss, métricas, arquitecturas y training modes.
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    name: str
    loss_variant: str
    use_custom_recall: bool
    hidden_units: List[int]
    dropout: float
    training_mode: str  # "fit" o "custom_loop"
    use_tf_function: bool


@dataclass
class ExperimentResult:
    name: str
    pr_auc: float
    recall_at_k: float
    f1: float
    loss: float
    train_time_sec: float
    params: Dict


def build_experiments() -> List[ExperimentConfig]:
    """Construye la lista de 20+ experimentos a ejecutar."""
    exps: List[ExperimentConfig] = []

    # 1. Variantes de loss (5 experimentos)
    loss_variants = ["weighted_bce", "focal", "bce", "bce_ls", "bce_pos_weight_3"]
    for lv in loss_variants:
        exps.append(
            ExperimentConfig(
                name=f"loss_{lv}",
                loss_variant=lv,
                use_custom_recall=True,
                hidden_units=config.DEFAULT_HIDDEN_UNITS,
                dropout=config.DEFAULT_DROPOUT,
                training_mode="fit",
                use_tf_function=True,
            )
        )

    # 2. Variantes de métricas (5 experimentos)
    for i, use_cr in enumerate([True, False, True, False, True]):
        exps.append(
            ExperimentConfig(
                name=f"metrics_recall_{i}_{int(use_cr)}",
                loss_variant="weighted_bce",
                use_custom_recall=use_cr,
                hidden_units=config.DEFAULT_HIDDEN_UNITS,
                dropout=config.DEFAULT_DROPOUT,
                training_mode="fit",
                use_tf_function=True,
            )
        )

    # 3. Variantes de arquitectura (5 experimentos)
    archs = [
        [16],
        [64, 32],
        [64, 32, 16],
        [128],
        [32, 32],
    ]
    dropouts = [0.0, 0.1, 0.2, 0.3, 0.4]
    for i in range(5):
        exps.append(
            ExperimentConfig(
                name=f"arch_{i}",
                loss_variant="weighted_bce",
                use_custom_recall=True,
                hidden_units=archs[i],
                dropout=dropouts[i],
                training_mode="fit",
                use_tf_function=True,
            )
        )

    # 4. Variantes de training mode (5 experimentos)
    modes = [
        ("fit", True),
        ("fit", False),
        ("custom_loop", True),
        ("custom_loop", False),
        ("custom_loop", True),
    ]
    for i, (tm, tf_f) in enumerate(modes):
        exps.append(
            ExperimentConfig(
                name=f"train_{i}_{tm}_tf{int(tf_f)}",
                loss_variant="weighted_bce",
                use_custom_recall=True,
                hidden_units=config.DEFAULT_HIDDEN_UNITS,
                dropout=config.DEFAULT_DROPOUT,
                training_mode=tm,
                use_tf_function=tf_f,
            )
        )

    assert len(exps) >= 20, f"Se necesitan al menos 20 experimentos, se tienen {len(exps)}"
    logger.info(f"Total de experimentos configurados: {len(exps)}")
    return exps


def run_experiment(
    exp: ExperimentConfig,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
) -> ExperimentResult:
    """Ejecuta un experimento completo y retorna resultados."""

    input_dim = X_train.shape[1]

    model = models.RiskModel(
        input_dim=input_dim,
        hidden_units=exp.hidden_units,
        dropout_rate=exp.dropout,
        task="classification",
    )

    compile_kwargs = models.get_compile_kwargs(
        task="classification",
        loss_variant=exp.loss_variant,
        use_custom_recall=exp.use_custom_recall
    )

    loss_fn = compile_kwargs["loss"]
    metric_objs = compile_kwargs["metrics"]

    optimizer = keras.optimizers.Adam(1e-3)

    batch_size = 64
    epochs = 4  # Reducido para rapidez

    start = time.time()

    if exp.training_mode == "fit":
        # Entrenamiento con model.fit()
        model.compile(optimizer=optimizer, **compile_kwargs)
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )
    else:
        # Custom training loop
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)

        if exp.use_tf_function:
            @tf.function
            def train_step(xb, yb):
                with tf.GradientTape() as tape:
                    logits = model(xb, training=True)
                    logits = tf.reshape(logits, tf.shape(yb))
                    loss_value = loss_fn(yb, logits)
                    if model.losses:
                        loss_value += tf.add_n(model.losses)
                grads = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                for m in metric_objs:
                    m.update_state(yb, logits)
                return loss_value
        else:
            def train_step(xb, yb):
                with tf.GradientTape() as tape:
                    logits = model(xb, training=True)
                    logits = tf.reshape(logits, tf.shape(yb))
                    loss_value = loss_fn(yb, logits)
                    if model.losses:
                        loss_value += tf.add_n(model.losses)
                grads = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                for m in metric_objs:
                    m.update_state(yb, logits)
                return loss_value

        for epoch in range(epochs):
            for m in metric_objs:
                if hasattr(m, 'reset_states'):
                    m.reset_states()
            for xb, yb in train_ds:
                _ = train_step(xb, yb)

    train_time = time.time() - start

    # Evaluación en test
    y_scores = model.predict(X_test, batch_size=256, verbose=0).reshape(-1)
    y_true = y_test.reshape(-1)

    # Métricas
    pr_auc_metric = keras.metrics.AUC(curve="PR")
    pr_auc_metric.update_state(y_true, y_scores)
    pr_auc = float(pr_auc_metric.result().numpy())

    k = min(config.DEFAULT_K_HOTSPOTS, len(y_scores))
    topk_idx = np.argsort(-y_scores)[:k]
    recall_at_k = float(y_true[topk_idx].sum() / max(y_true.sum(), 1.0))

    y_pred_bin = (y_scores >= 0.5).astype(int)
    tp = float(((y_pred_bin == 1) & (y_true == 1)).sum())
    fp = float(((y_pred_bin == 1) & (y_true == 0)).sum())
    fn = float(((y_pred_bin == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)

    loss_metric = keras.losses.BinaryCrossentropy()
    loss_value = float(loss_metric(y_true, y_scores).numpy())

    params = asdict(exp)

    return ExperimentResult(
        name=exp.name,
        pr_auc=pr_auc,
        recall_at_k=recall_at_k,
        f1=f1,
        loss=loss_value,
        train_time_sec=train_time,
        params=params,
    )


def main():
    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    logger.info("="*80)
    logger.info("EXPERIMENTOS CON 20+ MODELOS USANDO MySQL")
    logger.info("="*80)

    # Cargar datos
    logger.info("\nCargando datos desde MySQL...")
    df_raw = data.load_from_mysql()

    if df_raw.empty:
        logger.error("No se pudieron cargar datos. Abortando.")
        return

    panel = data.make_panel_district_week(df_raw)
    train_df, val_df, test_df = data.train_val_test_split_time(panel)

    X_train, y_train, _ = data.get_feature_target_arrays(train_df, task="classification")
    X_val, y_val, _ = data.get_feature_target_arrays(val_df, task="classification")
    X_test, y_test, _ = data.get_feature_target_arrays(test_df, task="classification")

    logger.info(f"Datos cargados: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Construir experimentos
    exps = build_experiments()

    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    results: List[ExperimentResult] = []

    # Ejecutar experimentos
    logger.info(f"\nEjecutando {len(exps)} experimentos...\n")

    for i, exp in enumerate(exps):
        logger.info(f"[{i+1}/{len(exps)}] Ejecutando: {exp.name}")
        try:
            res = run_experiment(exp, X_train, y_train, X_val, y_val, X_test, y_test)
            results.append(res)
            logger.info(
                f"  ✓ PR-AUC={res.pr_auc:.4f}, Recall@K={res.recall_at_k:.4f}, "
                f"F1={res.f1:.4f}, Time={res.train_time_sec:.2f}s"
            )
        except Exception as e:
            logger.error(f"  ✗ Error en experimento {exp.name}: {e}")

    # Construir leaderboard
    logger.info("\n" + "="*80)
    logger.info("CONSTRUYENDO LEADERBOARD")
    logger.info("="*80)

    rows = []
    for r in results:
        row = {
            "name": r.name,
            "pr_auc": r.pr_auc,
            "recall_at_k": r.recall_at_k,
            "f1": r.f1,
            "loss": r.loss,
            "train_time_sec": r.train_time_sec,
        }
        row.update({f"param_{k}": v for k, v in r.params.items()})
        rows.append(row)

    leaderboard = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    leaderboard_path = os.path.join(config.REPORTS_DIR, "leaderboard_mysql.csv")
    leaderboard.to_csv(leaderboard_path, index=False)

    logger.info(f"\nLeaderboard guardado en: {leaderboard_path}")
    logger.info("\nTop 5 modelos por PR-AUC:")
    print(leaderboard[["name", "pr_auc", "recall_at_k", "f1", "train_time_sec"]].head(5).to_string(index=False))

    # Guardar mejor modelo
    if not leaderboard.empty:
        best_name = leaderboard.iloc[0]["name"]
        logger.info(f"\nMejor modelo: {best_name}")

        # Re-entrenar mejor modelo con train+val
        best_exp = next(e for e in exps if e.name == best_name)
        logger.info("Re-entrenando mejor modelo con train+val...")

        input_dim = X_train.shape[1]
        best_model = models.RiskModel(
            input_dim=input_dim,
            hidden_units=best_exp.hidden_units,
            dropout_rate=best_exp.dropout,
            task="classification",
        )
        compile_kwargs = models.get_compile_kwargs(
            task="classification",
            loss_variant=best_exp.loss_variant,
            use_custom_recall=best_exp.use_custom_recall,
        )
        best_model.compile(optimizer=keras.optimizers.Adam(1e-3), **compile_kwargs)
        best_model.fit(
            np.concatenate([X_train, X_val], axis=0),
            np.concatenate([y_train, y_val], axis=0),
            epochs=6,
            batch_size=64,
            verbose=0,
        )

        best_model_path = os.path.join(config.REPORTS_DIR, "best_model_mysql.keras")
        best_model.save(best_model_path)
        logger.info(f"Mejor modelo guardado en: {best_model_path}")

    # Gráficos comparativos
    logger.info("\nGenerando gráficos...")

    # 1. PR-AUC bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(range(len(leaderboard)), leaderboard["pr_auc"])
    ax.set_xticks(range(len(leaderboard)))
    ax.set_xticklabels(leaderboard["name"], rotation=90, ha="right")
    ax.set_ylabel("PR-AUC")
    ax.set_title(f"Comparación de {len(leaderboard)} Modelos - PR-AUC (datos MySQL)")
    ax.axhline(y=leaderboard["pr_auc"].mean(), color="r", linestyle="--", label="Media")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.REPORTS_DIR, "leaderboard_mysql_pr_auc.png"), dpi=150)
    plt.close(fig)

    # 2. Scatter: PR-AUC vs Time
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        leaderboard["train_time_sec"],
        leaderboard["pr_auc"],
        s=100,
        alpha=0.6,
        c=leaderboard["f1"],
        cmap="viridis"
    )
    ax.set_xlabel("Training Time (seconds)")
    ax.set_ylabel("PR-AUC")
    ax.set_title("PR-AUC vs Training Time (color = F1-Score)")
    plt.colorbar(scatter, label="F1-Score")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.REPORTS_DIR, "leaderboard_mysql_scatter.png"), dpi=150)
    plt.close(fig)

    logger.info("Gráficos guardados")

    # Resumen JSON
    summary = {
        "total_experiments": len(results),
        "best_model": {
            "name": best_name if not leaderboard.empty else None,
            "pr_auc": float(leaderboard.iloc[0]["pr_auc"]) if not leaderboard.empty else None,
        },
        "mean_pr_auc": float(leaderboard["pr_auc"].mean()),
        "std_pr_auc": float(leaderboard["pr_auc"].std()),
        "mean_time_sec": float(leaderboard["train_time_sec"].mean()),
    }

    summary_path = os.path.join(config.REPORTS_DIR, "experiments_mysql_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nResumen guardado en: {summary_path}")

    logger.info("\n" + "="*80)
    logger.info("EXPERIMENTOS COMPLETADOS")
    logger.info("="*80)
    logger.info(f"Total experimentos: {len(results)}")
    logger.info(f"PR-AUC promedio: {summary['mean_pr_auc']:.4f} ± {summary['std_pr_auc']:.4f}")
    logger.info(f"Tiempo promedio: {summary['mean_time_sec']:.2f}s")
    logger.info("="*80)


if __name__ == "__main__":
    main()
