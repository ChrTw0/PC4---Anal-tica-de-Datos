"""Script de prueba para verificar la conexión a MySQL.

Este script verifica:
1. Conexión a la base de datos
2. Tablas disponibles
3. Estructura de la tabla de denuncias
4. Tipos de delitos y distritos disponibles
5. Carga de datos y creación de panel
"""

import logging
import sys
from pathlib import Path

# Agregar directorio raíz al path para importar desde src
sys.path.insert(0, str(Path(__file__).parent.parent))

import database
import data_mysql
import config_mysql
from src import models

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_connection():
    """Prueba 1: Verificar conexión básica"""
    logger.info("\n" + "="*60)
    logger.info("PRUEBA 1: Conexión a MySQL")
    logger.info("="*60)

    try:
        success = database.test_connection()
        if success:
            logger.info("✅ Conexión exitosa a MySQL")
            return True
        else:
            logger.error("❌ Falló la conexión a MySQL")
            return False
    except Exception as e:
        logger.error(f"❌ Error en la conexión: {e}")
        return False


def test_database_info():
    """Prueba 2: Información de la base de datos"""
    logger.info("\n" + "="*60)
    logger.info("PRUEBA 2: Información de Base de Datos")
    logger.info("="*60)

    try:
        # Obtener tablas
        tables = database.get_table_names()
        logger.info(f"\nTablas encontradas ({len(tables)}):")
        for table in tables:
            logger.info(f"  - {table}")

        # Verificar si existe la tabla de denuncias
        if config_mysql.MYSQL_TABLE_NAME not in tables:
            logger.warning(f"⚠️ Tabla '{config_mysql.MYSQL_TABLE_NAME}' no encontrada")
            logger.info("Puedes cambiar MYSQL_TABLE_NAME en src/config_mysql.py")
            return False

        # Obtener estructura de la tabla
        logger.info(f"\nEstructura de '{config_mysql.MYSQL_TABLE_NAME}':")
        table_info = database.get_table_info(config_mysql.MYSQL_TABLE_NAME)
        print(table_info.to_string(index=False))

        logger.info("✅ Información de base de datos obtenida")
        return True

    except Exception as e:
        logger.error(f"❌ Error al obtener información: {e}")
        return False


def test_available_data():
    """Prueba 3: Datos disponibles"""
    logger.info("\n" + "="*60)
    logger.info("PRUEBA 3: Datos Disponibles")
    logger.info("="*60)

    try:
        # Tipos de delitos
        crime_types = database.get_available_crime_types(
            config_mysql.MYSQL_TABLE_NAME,
            config_mysql.MYSQL_CRIME_TYPE_COLUMN
        )
        logger.info(f"\nTipos de delitos ({len(crime_types)}):")
        for ct in crime_types[:10]:  # Solo primeros 10
            logger.info(f"  - {ct}")
        if len(crime_types) > 10:
            logger.info(f"  ... y {len(crime_types) - 10} más")

        # Distritos
        districts = database.get_available_districts(
            config_mysql.MYSQL_TABLE_NAME,
            config_mysql.MYSQL_DISTRICT_COLUMN
        )
        logger.info(f"\nDistritos ({len(districts)}):")
        for d in districts[:10]:  # Solo primeros 10
            logger.info(f"  - {d}")
        if len(districts) > 10:
            logger.info(f"  ... y {len(districts) - 10} más")

        # Rango de fechas
        date_range = database.get_date_range(
            config_mysql.MYSQL_TABLE_NAME,
            config_mysql.MYSQL_DATE_COLUMN
        )
        logger.info(f"\nRango de fechas:")
        logger.info(f"  Desde: {date_range['min_date']}")
        logger.info(f"  Hasta: {date_range['max_date']}")

        logger.info("✅ Datos disponibles consultados exitosamente")
        return True

    except Exception as e:
        logger.error(f"❌ Error al consultar datos: {e}")
        return False


def test_load_data():
    """Prueba 4: Cargar datos agregados"""
    logger.info("\n" + "="*60)
    logger.info("PRUEBA 4: Cargar Datos Agregados")
    logger.info("="*60)

    try:
        # Cargar datos usando data_mysql
        df = data_mysql.load_from_mysql()

        if df.empty:
            logger.warning("⚠️ No se cargaron datos desde MySQL")
            return False

        logger.info(f"\nDataFrame cargado:")
        logger.info(f"  Filas: {len(df)}")
        logger.info(f"  Columnas: {list(df.columns)}")
        logger.info(f"  Distritos únicos: {df['district'].nunique()}")
        logger.info(f"  Rango de fechas: {df['date'].min()} a {df['date'].max()}")
        logger.info(f"  Total de denuncias: {df['count'].sum()}")
        logger.info(f"  Promedio por distrito-día: {df['count'].mean():.2f}")

        logger.info(f"\nPrimeras 5 filas:")
        print(df.head().to_string(index=False))

        logger.info("✅ Datos cargados exitosamente")
        return True

    except Exception as e:
        logger.error(f"❌ Error al cargar datos: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_create_panel():
    """Prueba 5: Crear panel distrito-semana"""
    logger.info("\n" + "="*60)
    logger.info("PRUEBA 5: Crear Panel Distrito-Semana")
    logger.info("="*60)

    try:
        # Cargar datos
        df = data_mysql.load_from_mysql()

        if df.empty:
            logger.warning("⚠️ No hay datos para procesar")
            return False

        # Crear panel
        panel = data_mysql.make_panel_district_week(df)

        logger.info(f"\nPanel creado:")
        logger.info(f"  Filas: {len(panel)}")
        logger.info(f"  Columnas: {list(panel.columns)}")
        logger.info(f"  Distritos: {panel['district'].nunique()}")
        logger.info(f"  Semanas: {panel['week_start'].nunique()}")

        # Verificar features y targets
        feature_cols = [c for c in panel.columns if 'lag' in c or 'roll' in c]
        logger.info(f"  Features creadas: {feature_cols}")

        target_cols = [c for c in panel.columns if 'target' in c]
        logger.info(f"  Targets creados: {target_cols}")

        logger.info(f"\nPrimeras 5 filas del panel:")
        print(panel.head().to_string(index=False))

        logger.info("✅ Panel creado exitosamente")
        return True

    except Exception as e:
        logger.error(f"❌ Error al crear panel: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_train_test_split():
    """Prueba 6: Split train/val/test"""
    logger.info("\n" + "="*60)
    logger.info("PRUEBA 6: Split Train/Val/Test")
    logger.info("="*60)

    try:
        # Cargar y procesar datos
        df = data_mysql.load_from_mysql()
        panel = data_mysql.make_panel_district_week(df)

        # Split
        train_df, val_df, test_df = data_mysql.train_val_test_split_time(panel)

        logger.info(f"\nSplits creados:")
        logger.info(f"  Train: {len(train_df)} filas")
        logger.info(f"  Val:   {len(val_df)} filas")
        logger.info(f"  Test:  {len(test_df)} filas")

        # Obtener arrays
        X_train, y_train, feature_cols = data_mysql.get_feature_target_arrays(
            train_df, task="classification"
        )
        X_test, y_test, _ = data_mysql.get_feature_target_arrays(
            test_df, task="classification"
        )

        logger.info(f"\nArrays generados:")
        logger.info(f"  X_train: {X_train.shape}")
        logger.info(f"  y_train: {y_train.shape}, positivos={y_train.sum():.0f}")
        logger.info(f"  X_test: {X_test.shape}")
        logger.info(f"  y_test: {y_test.shape}, positivos={y_test.sum():.0f}")
        logger.info(f"  Features: {feature_cols}")

        logger.info("✅ Split y arrays creados exitosamente")
        return True

    except Exception as e:
        logger.error(f"❌ Error en split: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Ejecuta todas las pruebas"""
    logger.info("\n" + "="*80)
    logger.info("PRUEBAS DE CONEXIÓN Y CARGA DE DATOS DESDE MySQL")
    logger.info("="*80)
    logger.info(f"\nConfiguración:")
    logger.info(f"  Host: {config_mysql.MYSQL_HOST}:{config_mysql.MYSQL_PORT}")
    logger.info(f"  Database: {config_mysql.MYSQL_DATABASE}")
    logger.info(f"  User: {config_mysql.MYSQL_USER}")
    logger.info(f"  Tabla: {config_mysql.MYSQL_TABLE_NAME}")
    logger.info(f"  Filtro de delito: {config_mysql.MYSQL_CRIME_TYPE_FILTER or 'Ninguno (todos)'}")

    results = []

    # Ejecutar pruebas
    results.append(("Conexión MySQL", test_connection()))
    results.append(("Información BD", test_database_info()))
    results.append(("Datos disponibles", test_available_data()))
    results.append(("Cargar datos", test_load_data()))
    results.append(("Crear panel", test_create_panel()))
    results.append(("Train/test split", test_train_test_split()))

    # Resumen final
    logger.info("\n" + "="*80)
    logger.info("RESUMEN DE PRUEBAS")
    logger.info("="*80)

    for test_name, result in results:
        status = "✅ PASÓ" if result else "❌ FALLÓ"
        logger.info(f"{status} - {test_name}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    logger.info("\n" + "="*80)
    logger.info(f"RESULTADO FINAL: {passed}/{total} pruebas pasaron")
    logger.info("="*80)

    if passed == total:
        logger.info("\n🎉 ¡Todas las pruebas pasaron! El sistema está listo para usar.")
    else:
        logger.warning("\n⚠️ Algunas pruebas fallaron. Revisa los logs arriba.")
        logger.info("\nConsejos:")
        logger.info("1. Verifica que MySQL esté corriendo")
        logger.info("2. Verifica las credenciales en src/config_mysql.py")
        logger.info("3. Verifica que la tabla y columnas existan")
        logger.info("4. Verifica que haya datos en la tabla")


if __name__ == "__main__":
    main()
