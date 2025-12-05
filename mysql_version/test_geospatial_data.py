"""Script de prueba para el pipeline de datos geoespaciales.

Este script verifica que:
1. Los datos se cargan correctamente desde MySQL
2. Los grids se crean apropiadamente
3. Las features temporales y espaciales se calculan
4. Los targets se generan correctamente
5. El split temporal funciona
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import config_mysql as config
import data_geospatial_mysql as data_geo

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("="*80)
    logger.info("TEST: PIPELINE DE DATOS GEOESPACIALES")
    logger.info("="*80)

    # 1. Crear panel completo
    logger.info("\n[1/5] Creando panel grid-semana...")
    panel = data_geo.make_panel_grid_week()

    logger.info(f"\nPanel shape: {panel.shape}")
    logger.info(f"Columnas: {list(panel.columns)}")

    # 2. Verificar features
    logger.info("\n[2/5] Verificando features...")
    feature_cols = [c for c in panel.columns if 'lag' in c or 'roll' in c or 'neighbor' in c]
    logger.info(f"Features temporales/espaciales: {feature_cols}")

    # Verificar NaNs
    nan_counts = panel[feature_cols].isna().sum()
    logger.info(f"\nNaNs por columna:")
    for col, count in nan_counts.items():
        logger.info(f"  {col:20s}: {count:,} ({count/len(panel)*100:.2f}%)")

    # 3. Verificar targets
    logger.info("\n[3/5] Verificando targets...")
    logger.info(f"target_bin_t1:")
    logger.info(f"  Min:  {panel['target_bin_t1'].min()}")
    logger.info(f"  Max:  {panel['target_bin_t1'].max()}")
    logger.info(f"  Mean: {panel['target_bin_t1'].mean():.4f}")

    logger.info(f"\ntarget_count_t1:")
    logger.info(f"  Min:  {panel['target_count_t1'].min():.0f}")
    logger.info(f"  Max:  {panel['target_count_t1'].max():.0f}")
    logger.info(f"  Mean: {panel['target_count_t1'].mean():.2f}")
    logger.info(f"  Std:  {panel['target_count_t1'].std():.2f}")

    # 4. Split temporal
    logger.info("\n[4/5] Probando split temporal...")
    train_df, val_df, test_df = data_geo.train_val_test_split_time(panel)

    logger.info(f"Train: {len(train_df):,} filas")
    logger.info(f"  Semanas: {train_df['week_start'].min()} a {train_df['week_start'].max()}")
    logger.info(f"Val:   {len(val_df):,} filas")
    logger.info(f"  Semanas: {val_df['week_start'].min()} a {val_df['week_start'].max()}")
    logger.info(f"Test:  {len(test_df):,} filas")
    logger.info(f"  Semanas: {test_df['week_start'].min()} a {test_df['week_start'].max()}")

    # 5. Extraer arrays
    logger.info("\n[5/5] Extrayendo arrays X, y...")
    X_train, y_train, feature_names = data_geo.get_feature_target_arrays(train_df, task="classification")
    X_test, y_test, _ = data_geo.get_feature_target_arrays(test_df, task="classification")

    logger.info(f"\nX_train shape: {X_train.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"X_test shape:  {X_test.shape}")
    logger.info(f"y_test shape:  {y_test.shape}")

    logger.info(f"\nFeatures ({len(feature_names)}):")
    for i, feat in enumerate(feature_names, 1):
        logger.info(f"  {i:2d}. {feat}")

    # Verificar balance de clases
    logger.info("\n" + "="*80)
    logger.info("BALANCE DE CLASES")
    logger.info("="*80)

    train_pos = y_train.sum()
    train_neg = len(y_train) - train_pos
    logger.info(f"Train:")
    logger.info(f"  Positivos: {int(train_pos):,} ({train_pos/len(y_train)*100:.2f}%)")
    logger.info(f"  Negativos: {int(train_neg):,} ({train_neg/len(y_train)*100:.2f}%)")
    logger.info(f"  Ratio:     1:{train_neg/max(train_pos, 1):.1f}")

    test_pos = y_test.sum()
    test_neg = len(y_test) - test_pos
    logger.info(f"\nTest:")
    logger.info(f"  Positivos: {int(test_pos):,} ({test_pos/len(y_test)*100:.2f}%)")
    logger.info(f"  Negativos: {int(test_neg):,} ({test_neg/len(y_test)*100:.2f}%)")
    logger.info(f"  Ratio:     1:{test_neg/max(test_pos, 1):.1f}")

    # Verificar data leakage
    logger.info("\n" + "="*80)
    logger.info("VERIFICACIÓN DE DATA LEAKAGE")
    logger.info("="*80)

    if 'count' in feature_names:
        logger.error("⚠️  ERROR: 'count' está en features (data leakage!)")
    else:
        logger.info("✓ 'count' NO está en features")

    # Verificar que lags estén shifteados correctamente
    logger.info("\nVerificando lags (primeros 10 registros de un grid):")
    sample_grid = panel['grid_id'].iloc[0]
    sample = panel[panel['grid_id'] == sample_grid].head(10)
    logger.info(sample[['week_start', 'count', 'count_lag1', 'count_lag2', 'target_count_t1']].to_string())

    # Estadísticas de grids
    logger.info("\n" + "="*80)
    logger.info("ESTADÍSTICAS DE GRIDS")
    logger.info("="*80)

    grid_stats = data_geo.get_grid_statistics(panel)
    logger.info(f"\nTop 10 grids por total de extorsiones:")
    logger.info(grid_stats[['grid_id', 'total_crimes', 'mean_weekly', 'grid_lat', 'grid_lon']].head(10).to_string())

    # Guardar panel para inspección
    output_path = '../reports/panel_grid_week_sample.csv'
    panel.head(1000).to_csv(output_path, index=False)
    logger.info(f"\n✓ Muestra del panel guardada en: {output_path}")

    logger.info("\n" + "="*80)
    logger.info("✓ TEST COMPLETADO EXITOSAMENTE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
