"""Módulo de datos adaptado para MySQL.

Este módulo extiende data.py pero carga datos desde MySQL en lugar de CSV.
Mantiene la misma API que data.py para compatibilidad con el resto del código.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

import config_mysql as config
import database

logger = logging.getLogger(__name__)


def load_from_mysql() -> pd.DataFrame:
    """Carga datos de denuncias desde MySQL.

    Esta función reemplaza a load_csv() pero retorna el mismo formato.

    Returns:
        DataFrame con columnas: district, date, count
    """
    logger.info("Cargando datos desde MySQL...")

    # Intentar cargar datos agregados directamente desde MySQL
    df = database.load_denuncias_aggregated(
        table_name=config.MYSQL_TABLE_NAME,
        district_column=config.MYSQL_DISTRICT_COLUMN,
        date_column=config.MYSQL_DATE_COLUMN,
        crime_type=config.MYSQL_CRIME_TYPE_FILTER,
        crime_column=config.MYSQL_CRIME_TYPE_COLUMN,
        start_date=config.MYSQL_START_DATE,
        end_date=config.MYSQL_END_DATE,
    )

    if df.empty:
        logger.warning("No se pudieron cargar datos desde MySQL. Generando datos sintéticos...")
        df = generate_synthetic_data()
    else:
        logger.info(f"Datos cargados exitosamente: {len(df)} filas, {df['district'].nunique()} distritos")

        # Validar columnas requeridas
        required_cols = ['district', 'date', 'count']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columnas faltantes: {missing_cols}")

    return df


def generate_synthetic_data(n_districts: int = 20, n_weeks: int = 80) -> pd.DataFrame:
    """Genera un panel sintético distrito-semana para demo.

    Mantiene la misma firma que data.py para compatibilidad.

    Args:
        n_districts: Número de distritos
        n_weeks: Número de semanas

    Returns:
        DataFrame con columnas: district, date, count
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    districts = [f"DIST_{i:02d}" for i in range(1, n_districts + 1)]
    dates = pd.date_range("2020-01-06", periods=n_weeks, freq="W-MON")

    rows = []
    for d in districts:
        base_rate = rng.uniform(0.1, 1.0)
        for dt in dates:
            lam = base_rate * rng.uniform(0.5, 2.0)
            cnt = rng.poisson(lam)
            rows.append({"district": d, "date": dt.date().isoformat(), "count": int(cnt)})

    df = pd.DataFrame(rows)
    logger.info(f"Generados {len(df)} registros sintéticos")
    return df


def make_panel_district_week(df: pd.DataFrame) -> pd.DataFrame:
    """Transforma el dataset a panel distrito-semana.

    Mantiene exactamente la misma lógica que data.py.

    Args:
        df: DataFrame con columnas district, date, count

    Returns:
        Panel procesado con lags, rolling y targets
    """
    for col in config.REQUIRED_COLUMNS_MIN:
        if col not in df.columns:
            raise ValueError(f"Columna requerida faltante: {col}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Definir semana como el lunes de cada semana
    df["week_start"] = df["date"].dt.to_period("W-MON").dt.start_time

    group_cols = ["district", "week_start"]
    agg_dict = {"count": "sum"}

    # Mantener features extra agregadas por suma o media simple
    extra_cols = [c for c in df.columns if c not in ["district", "date", "count", "week_start"]]
    for c in extra_cols:
        agg_dict[c] = "mean"

    panel = df.groupby(group_cols, as_index=False).agg(agg_dict).sort_values(["district", "week_start"])

    # Ordenar y crear índices por distrito
    panel = panel.sort_values(["district", "week_start"]).reset_index(drop=True)

    # Crear lags y rolling por distrito
    def _add_lags(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("week_start").reset_index(drop=True)
        for l in config.LAG_STEPS:
            g[f"count_lag{l}"] = g["count"].shift(l)
        # Rolling mean debe usar SOLO valores pasados (shift antes de rolling)
        g["count_roll_mean4"] = g["count"].shift(1).rolling(config.ROLLING_WINDOW, min_periods=1).mean()
        return g

    panel = panel.groupby("district", group_keys=False).apply(_add_lags)

    # Crear target binario y target de conteo t+1 (para modo Poisson)
    panel["target_count_t1"] = panel.groupby("district")["count"].shift(-1)
    panel["target_bin_t1"] = (panel["target_count_t1"] >= 1).astype(float)

    # Eliminar filas sin target (última semana por distrito)
    panel = panel.dropna(subset=["target_count_t1"]).reset_index(drop=True)

    logger.info(f"Panel creado: {len(panel)} filas, {panel['district'].nunique()} distritos")
    return panel


def train_val_test_split_time(
    df: pd.DataFrame,
    val_size: float = 0.2,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split temporal simple respetando el orden.

    Mantiene exactamente la misma lógica que data.py.

    Args:
        df: Panel con datos ordenados por tiempo
        val_size: Proporción de validación
        test_size: Proporción de test

    Returns:
        Tuple con (train_df, val_df, test_df)
    """
    df = df.sort_values(["week_start"]).reset_index(drop=True)
    n = len(df)
    n_test = int(n * test_size)
    n_val = int((n - n_test) * val_size)

    train = df.iloc[: n - n_test - n_val]
    val = df.iloc[n - n_test - n_val : n - n_test]
    test = df.iloc[n - n_test :]

    logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def get_feature_target_arrays(
    df: pd.DataFrame,
    task: str = "classification",
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Devuelve X, y según el modo de tarea.

    Mantiene exactamente la misma lógica que data.py.

    Args:
        df: Panel con features y targets
        task: "classification" o "poisson"

    Returns:
        Tuple con (X, y, feature_cols)
    """
    feature_cols = [
        c
        for c in df.columns
        if c
        not in [
            "district",
            "date",
            "week_start",
            "target_bin_t1",
            "target_count_t1",
            "count",  # Excluir count actual (data leakage)
        ]
        and np.issubdtype(df[c].dtype, np.number)
    ]

    if task == "classification":
        target_col = "target_bin_t1"
    elif task == "poisson":
        target_col = "target_count_t1"
    else:
        raise ValueError("task debe ser 'classification' o 'poisson'")

    X = df[feature_cols].values.astype("float32")
    y = df[target_col].values.astype("float32")

    logger.info(f"Arrays generados: X.shape={X.shape}, y.shape={y.shape}")
    return X, y, feature_cols


# ==========================================
# Funciones de utilidad específicas de MySQL
# ==========================================

def get_database_info() -> dict:
    """Obtiene información sobre la base de datos.

    Returns:
        Dict con información de tablas, columnas, rangos de fechas, etc.
    """
    info = {}

    # Tablas disponibles
    info['tables'] = database.get_table_names()

    # Info de la tabla principal
    if config.MYSQL_TABLE_NAME in info['tables']:
        info['table_info'] = database.get_table_info(config.MYSQL_TABLE_NAME).to_dict('records')
        info['districts'] = database.get_available_districts(
            config.MYSQL_TABLE_NAME,
            config.MYSQL_DISTRICT_COLUMN
        )
        info['crime_types'] = database.get_available_crime_types(
            config.MYSQL_TABLE_NAME,
            config.MYSQL_CRIME_TYPE_COLUMN
        )
        info['date_range'] = database.get_date_range(
            config.MYSQL_TABLE_NAME,
            config.MYSQL_DATE_COLUMN
        )

    return info


def print_database_summary():
    """Imprime un resumen de la información de la base de datos."""
    logger.info("=" * 60)
    logger.info("RESUMEN DE BASE DE DATOS MySQL")
    logger.info("=" * 60)

    info = get_database_info()

    logger.info(f"\nTablas disponibles: {', '.join(info.get('tables', []))}")

    if 'districts' in info:
        logger.info(f"\nNúmero de distritos: {len(info['districts'])}")
        logger.info(f"Ejemplos: {', '.join(info['districts'][:5])}")

    if 'crime_types' in info:
        logger.info(f"\nTipos de delitos: {len(info['crime_types'])}")
        logger.info(f"Ejemplos: {', '.join(info['crime_types'][:5])}")

    if 'date_range' in info:
        logger.info(f"\nRango de fechas:")
        logger.info(f"  Desde: {info['date_range']['min_date']}")
        logger.info(f"  Hasta: {info['date_range']['max_date']}")

    logger.info("\n" + "=" * 60)
