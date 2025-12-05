"""Módulo de datos geoespaciales para predicción de hotspots.

Este módulo carga datos de extorsión con coordenadas (lat, long) y crea
un panel grid-semana en lugar de distrito-semana para predicción espacial.

Flujo:
1. Cargar extorsiones con coordenadas desde MySQL (Lima + Callao)
2. Crear grids espaciales (~1km x 1km)
3. Agregar por grid-semana
4. Crear features: lags, rolling, features de vecindad
5. Crear target: extorsiones en grid en t+1
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

import config_mysql as config
import database

logger = logging.getLogger(__name__)

# Configuración geoespacial
GRID_SIZE = 0.01  # grados (~1.1 km en latitud)
LIMA_LAT_RANGE = (-12.6, -11.4)  # Lima + Callao
LIMA_LON_RANGE = (-77.5, -76.5)


def load_extorsion_geospatial() -> pd.DataFrame:
    """Carga datos de extorsión con coordenadas desde MySQL.

    Filtros aplicados:
    - modalidad_hecho = 'EXTORSION'
    - departamento_hecho IN ('LIMA', 'CALLAO')
    - Coordenadas válidas (no NULL, no 0)
    - Desde 2020-01-01

    Returns:
        DataFrame con columnas: fecha_hora_hecho, distrito_hecho, lat_hecho, long_hecho
    """
    logger.info("Cargando datos de extorsión con coordenadas desde MySQL...")

    engine = database.get_sqlalchemy_engine()

    query = """
    SELECT
        fecha_hora_hecho,
        distrito_hecho,
        departamento_hecho,
        lat_hecho,
        long_hecho
    FROM denuncias
    WHERE modalidad_hecho = 'EXTORSION'
        AND departamento_hecho IN ('LIMA', 'CALLAO')
        AND lat_hecho IS NOT NULL
        AND long_hecho IS NOT NULL
        AND lat_hecho != 0
        AND long_hecho != 0
        AND fecha_hora_hecho >= '2020-01-01'
    ORDER BY fecha_hora_hecho
    """

    df = pd.read_sql(query, engine)
    engine.dispose()

    logger.info(f"Cargados {len(df):,} casos de extorsión con coordenadas")
    return df


def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra outliers geográficos fuera de Lima-Callao.

    Args:
        df: DataFrame con columnas lat_hecho, long_hecho

    Returns:
        DataFrame filtrado
    """
    before = len(df)

    df_clean = df[
        (df['lat_hecho'] >= LIMA_LAT_RANGE[0]) &
        (df['lat_hecho'] <= LIMA_LAT_RANGE[1]) &
        (df['long_hecho'] >= LIMA_LON_RANGE[0]) &
        (df['long_hecho'] <= LIMA_LON_RANGE[1])
    ].copy()

    removed = before - len(df_clean)
    if removed > 0:
        logger.info(f"Removidos {removed} outliers ({removed/before*100:.2f}%)")

    logger.info(f"Datos limpios: {len(df_clean):,} casos")
    return df_clean


def create_spatial_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Crea grids espaciales de ~1km x 1km.

    Args:
        df: DataFrame con columnas lat_hecho, long_hecho

    Returns:
        DataFrame con columnas adicionales: grid_lat, grid_lon, grid_id
    """
    df = df.copy()

    # Redondear coordenadas al grid más cercano
    df['grid_lat'] = (df['lat_hecho'] / GRID_SIZE).round() * GRID_SIZE
    df['grid_lon'] = (df['long_hecho'] / GRID_SIZE).round() * GRID_SIZE

    # Crear ID único de grid
    df['grid_id'] = df['grid_lat'].astype(str) + '_' + df['grid_lon'].astype(str)

    logger.info(f"Grids únicos creados: {df['grid_id'].nunique():,}")
    return df


def aggregate_by_grid_week(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega datos por grid-semana.

    IMPORTANTE: Crea un panel COMPLETO con todas las combinaciones (grid, semana),
    incluyendo grids sin extorsiones (count=0). Esto es necesario para que el
    target binario tenga casos negativos.

    Args:
        df: DataFrame con columnas fecha_hora_hecho, grid_id

    Returns:
        Panel grid-semana con conteos de extorsiones
    """
    df = df.copy()
    df['fecha_hora_hecho'] = pd.to_datetime(df['fecha_hora_hecho'])

    # Crear semana (lunes como inicio)
    df['week_start'] = df['fecha_hora_hecho'].dt.to_period('W-MON').dt.start_time

    # Agregar por grid-semana (solo grids con extorsiones)
    panel_with_crimes = df.groupby(['grid_id', 'week_start'], as_index=False).agg({
        'lat_hecho': 'count',  # Contar casos
        'grid_lat': 'first',   # Mantener coordenadas
        'grid_lon': 'first'
    }).rename(columns={'lat_hecho': 'count'})

    # Crear mapeo de grid_id -> coordenadas
    grid_coords = df[['grid_id', 'grid_lat', 'grid_lon']].drop_duplicates()

    # Crear panel COMPLETO: todas las combinaciones (grid, semana)
    all_grids = grid_coords['grid_id'].unique()
    all_weeks = df['week_start'].unique()

    logger.info(f"Creando panel completo: {len(all_grids)} grids × {len(all_weeks)} semanas = {len(all_grids) * len(all_weeks):,} filas")

    # Producto cartesiano
    complete_panel = pd.DataFrame([
        (grid, week)
        for grid in all_grids
        for week in all_weeks
    ], columns=['grid_id', 'week_start'])

    # Merge con conteos (left join para mantener grids sin extorsiones)
    panel = complete_panel.merge(
        panel_with_crimes,
        on=['grid_id', 'week_start'],
        how='left'
    )

    # Rellenar NaNs: count=0, coordenadas desde grid_coords
    panel['count'] = panel['count'].fillna(0).astype(int)
    panel = panel.merge(grid_coords, on='grid_id', how='left', suffixes=('', '_from_map'))

    # Si hay duplicados de coordenadas, usar las del mapa
    if 'grid_lat_from_map' in panel.columns:
        panel['grid_lat'] = panel['grid_lat'].fillna(panel['grid_lat_from_map'])
        panel['grid_lon'] = panel['grid_lon'].fillna(panel['grid_lon_from_map'])
        panel = panel.drop(columns=['grid_lat_from_map', 'grid_lon_from_map'])

    # Ordenar
    panel = panel.sort_values(['grid_id', 'week_start']).reset_index(drop=True)

    zeros = (panel['count'] == 0).sum()
    logger.info(f"Panel completo creado: {len(panel):,} filas")
    logger.info(f"  Con extorsiones: {len(panel) - zeros:,} ({(len(panel) - zeros)/len(panel)*100:.1f}%)")
    logger.info(f"  Sin extorsiones: {zeros:,} ({zeros/len(panel)*100:.1f}%)")
    logger.info(f"Semanas únicas: {panel['week_start'].nunique()}")
    logger.info(f"Grids únicos: {panel['grid_id'].nunique()}")

    return panel


def create_temporal_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Crea features temporales: lags y rolling mean por grid.

    Args:
        panel: Panel grid-semana con columna count

    Returns:
        Panel con features adicionales
    """
    def _add_features_per_grid(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values('week_start').reset_index(drop=True)

        # Lags (1, 2, 3, 4 semanas atrás)
        for lag in config.LAG_STEPS:
            g[f'count_lag{lag}'] = g['count'].shift(lag)

        # Rolling mean (últimas 4 semanas, excluyendo actual)
        g['count_roll_mean4'] = g['count'].shift(1).rolling(
            config.ROLLING_WINDOW,
            min_periods=1
        ).mean()

        return g

    panel = panel.groupby('grid_id', group_keys=False).apply(_add_features_per_grid)

    logger.info("Features temporales creadas: lags y rolling mean")
    return panel


def create_spatial_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Crea features espaciales: conteos de grids vecinos.

    Para cada grid, calcula el promedio de extorsiones en grids vecinos
    (los 8 grids adyacentes en una ventana de 3x3).

    Args:
        panel: Panel con columnas grid_lat, grid_lon, count, week_start

    Returns:
        Panel con features de vecindad
    """
    # Crear diccionario semana -> DataFrame para acceso rápido
    weekly_data = {}
    for week, group in panel.groupby('week_start'):
        weekly_data[week] = group.set_index(['grid_lat', 'grid_lon'])['count'].to_dict()

    def _get_neighbor_mean(row):
        """Calcula el promedio de extorsiones en grids vecinos."""
        week = row['week_start']
        lat = row['grid_lat']
        lon = row['grid_lon']

        if week not in weekly_data:
            return 0.0

        week_dict = weekly_data[week]

        # Vecinos (8 direcciones)
        neighbors = []
        for dlat in [-GRID_SIZE, 0, GRID_SIZE]:
            for dlon in [-GRID_SIZE, 0, GRID_SIZE]:
                if dlat == 0 and dlon == 0:
                    continue  # Excluir el grid actual
                neighbor_key = (lat + dlat, lon + dlon)
                if neighbor_key in week_dict:
                    neighbors.append(week_dict[neighbor_key])

        return np.mean(neighbors) if neighbors else 0.0

    logger.info("Calculando features de vecindad (esto puede tardar)...")
    panel['neighbor_mean'] = panel.apply(_get_neighbor_mean, axis=1)

    logger.info("Features espaciales creadas: neighbor_mean")
    return panel


def create_targets(panel: pd.DataFrame) -> pd.DataFrame:
    """Crea targets: extorsiones en t+1 por grid.

    Args:
        panel: Panel con columna count

    Returns:
        Panel con target_count_t1 y target_bin_t1
    """
    # Target de conteo (para regresión Poisson)
    panel['target_count_t1'] = panel.groupby('grid_id')['count'].shift(-1)

    # Target binario (para clasificación)
    panel['target_bin_t1'] = (panel['target_count_t1'] >= 1).astype(float)

    # Eliminar filas sin target (última semana por grid)
    panel = panel.dropna(subset=['target_count_t1']).reset_index(drop=True)

    logger.info(f"Targets creados. Panel final: {len(panel):,} filas")

    # Estadísticas de balance
    pos_rate = panel['target_bin_t1'].mean()
    logger.info(f"Balance de clases: {pos_rate*100:.2f}% positivo")

    return panel


def make_panel_grid_week() -> pd.DataFrame:
    """Pipeline completo para crear panel grid-semana.

    Returns:
        Panel procesado con features y targets
    """
    logger.info("="*60)
    logger.info("CREANDO PANEL GRID-SEMANA")
    logger.info("="*60)

    # 1. Cargar datos
    df = load_extorsion_geospatial()

    # 2. Filtrar outliers
    df = filter_outliers(df)

    # 3. Crear grids espaciales
    df = create_spatial_grid(df)

    # 4. Agregar por grid-semana
    panel = aggregate_by_grid_week(df)

    # 5. Features temporales
    panel = create_temporal_features(panel)

    # 6. Features espaciales (vecindad)
    panel = create_spatial_features(panel)

    # 7. Crear targets
    panel = create_targets(panel)

    logger.info("="*60)
    logger.info("PANEL GRID-SEMANA COMPLETADO")
    logger.info("="*60)

    return panel


def train_val_test_split_time(
    df: pd.DataFrame,
    val_size: float = 0.2,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split temporal respetando el orden cronológico.

    Args:
        df: Panel ordenado por tiempo
        val_size: Proporción de validación
        test_size: Proporción de test

    Returns:
        Tuple con (train_df, val_df, test_df)
    """
    df = df.sort_values(['week_start']).reset_index(drop=True)
    n = len(df)
    n_test = int(n * test_size)
    n_val = int((n - n_test) * val_size)

    train = df.iloc[: n - n_test - n_val]
    val = df.iloc[n - n_test - n_val : n - n_test]
    test = df.iloc[n - n_test :]

    logger.info(f"Split temporal: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def get_feature_target_arrays(
    df: pd.DataFrame,
    task: str = "classification",
    scaler=None,
    fit_scaler: bool = True,
) -> Tuple[np.ndarray, np.ndarray, list, object]:
    """Extrae arrays X, y para entrenamiento con preprocesamiento.

    Args:
        df: Panel con features y targets
        task: "classification" o "poisson"
        scaler: StandardScaler pre-entrenado (para val/test)
        fit_scaler: Si True, entrena un nuevo scaler. Si False, usa el provisto.

    Returns:
        Tuple con (X, y, feature_cols, scaler)
    """
    from sklearn.preprocessing import StandardScaler

    # Columnas de features
    feature_cols = [
        c for c in df.columns
        if c not in [
            'grid_id', 'week_start', 'grid_lat', 'grid_lon',
            'target_bin_t1', 'target_count_t1',
            'count',  # Excluir count actual (data leakage)
        ]
        and np.issubdtype(df[c].dtype, np.number)
    ]

    # Target
    if task == "classification":
        target_col = "target_bin_t1"
    elif task == "poisson":
        target_col = "target_count_t1"
    else:
        raise ValueError("task debe ser 'classification' o 'poisson'")

    # Extraer valores
    X = df[feature_cols].copy()
    y = df[target_col].values.astype('float32')

    # 1. Rellenar NaNs con 0 (grids inactivos al inicio)
    nan_counts_before = X.isna().sum().sum()
    if nan_counts_before > 0:
        logger.info(f"Rellenando {nan_counts_before} NaNs con 0")
        X = X.fillna(0.0)

    # 2. Convertir a numpy
    X = X.values.astype('float32')

    # 3. Normalizar features
    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        logger.info("StandardScaler entrenado y aplicado")
    elif scaler is not None:
        X = scaler.transform(X)
        logger.info("StandardScaler aplicado (pre-entrenado)")
    else:
        logger.warning("No se aplicó normalización (scaler=None, fit_scaler=False)")

    logger.info(f"Arrays generados: X.shape={X.shape}, y.shape={y.shape}")
    logger.info(f"Features: {feature_cols}")

    # Verificar distribución
    logger.info(f"X stats: min={X.min():.2f}, max={X.max():.2f}, mean={X.mean():.2f}, std={X.std():.2f}")
    logger.info(f"y stats: positivos={y.sum():.0f} ({y.mean()*100:.2f}%)")

    return X, y, feature_cols, scaler


# Funciones de utilidad

def get_grid_coordinates(panel: pd.DataFrame) -> pd.DataFrame:
    """Obtiene coordenadas únicas de cada grid.

    Args:
        panel: Panel con columnas grid_id, grid_lat, grid_lon

    Returns:
        DataFrame con grid_id, grid_lat, grid_lon únicos
    """
    grid_coords = panel[['grid_id', 'grid_lat', 'grid_lon']].drop_duplicates()
    return grid_coords.reset_index(drop=True)


def get_grid_statistics(panel: pd.DataFrame) -> pd.DataFrame:
    """Calcula estadísticas por grid.

    Args:
        panel: Panel grid-semana

    Returns:
        DataFrame con estadísticas por grid
    """
    stats = panel.groupby('grid_id').agg({
        'count': ['sum', 'mean', 'std', 'max'],
        'grid_lat': 'first',
        'grid_lon': 'first'
    })

    stats.columns = ['total_crimes', 'mean_weekly', 'std_weekly', 'max_weekly', 'grid_lat', 'grid_lon']
    stats = stats.reset_index()

    return stats.sort_values('total_crimes', ascending=False)
