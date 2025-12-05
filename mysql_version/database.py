"""Módulo para conexión y consultas a MySQL.

Conecta a la base de datos denuncias_peru y proporciona funciones
para cargar datos de denuncias de manera eficiente.
"""

import logging
from typing import Optional, List, Dict, Any

import pandas as pd
import pymysql
from pymysql import Connection
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

import config_mysql as config

logger = logging.getLogger(__name__)


def get_connection() -> Connection:
    """Crea una conexión directa a MySQL usando pymysql.

    Returns:
        Connection: Conexión activa a la base de datos

    Raises:
        pymysql.Error: Si hay error en la conexión
    """
    try:
        conn = pymysql.connect(
            host=config.MYSQL_HOST,
            port=config.MYSQL_PORT,
            user=config.MYSQL_USER,
            password=config.MYSQL_PASSWORD,
            database=config.MYSQL_DATABASE,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        logger.info(f"Conexión exitosa a {config.MYSQL_DATABASE}")
        return conn
    except pymysql.Error as e:
        logger.error(f"Error al conectar a MySQL: {e}")
        raise


def get_sqlalchemy_engine() -> Engine:
    """Crea un engine de SQLAlchemy para pandas.

    Returns:
        Engine: SQLAlchemy engine configurado
    """
    connection_string = (
        f"mysql+pymysql://{config.MYSQL_USER}:{config.MYSQL_PASSWORD}"
        f"@{config.MYSQL_HOST}:{config.MYSQL_PORT}/{config.MYSQL_DATABASE}"
        f"?charset=utf8mb4"
    )
    engine = create_engine(connection_string, pool_pre_ping=True)
    logger.info("SQLAlchemy engine creado")
    return engine


def test_connection() -> bool:
    """Prueba la conexión a la base de datos.

    Returns:
        bool: True si la conexión es exitosa, False en caso contrario
    """
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()
            logger.info(f"MySQL version: {version}")
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Test de conexión fallido: {e}")
        return False


def get_table_names() -> List[str]:
    """Obtiene la lista de tablas disponibles en la base de datos.

    Returns:
        List[str]: Lista de nombres de tablas
    """
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute("SHOW TABLES")
            tables = [row[list(row.keys())[0]] for row in cursor.fetchall()]
        conn.close()
        logger.info(f"Tablas encontradas: {tables}")
        return tables
    except Exception as e:
        logger.error(f"Error al obtener tablas: {e}")
        return []


def get_table_info(table_name: str) -> pd.DataFrame:
    """Obtiene información sobre las columnas de una tabla.

    Args:
        table_name: Nombre de la tabla

    Returns:
        DataFrame con información de columnas (Field, Type, Null, Key, Default, Extra)
    """
    try:
        engine = get_sqlalchemy_engine()
        query = f"DESCRIBE {table_name}"
        df = pd.read_sql(query, engine)
        engine.dispose()
        return df
    except Exception as e:
        logger.error(f"Error al obtener info de tabla {table_name}: {e}")
        return pd.DataFrame()


def load_denuncias_raw(
    table_name: str = "denuncias",
    limit: Optional[int] = None,
    date_column: str = "fecha_denuncia",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Carga datos crudos de denuncias desde MySQL.

    Args:
        table_name: Nombre de la tabla de denuncias
        limit: Límite de filas a cargar (None = todas)
        date_column: Nombre de la columna de fecha
        start_date: Fecha inicio (formato 'YYYY-MM-DD')
        end_date: Fecha fin (formato 'YYYY-MM-DD')

    Returns:
        DataFrame con los datos de denuncias
    """
    try:
        engine = get_sqlalchemy_engine()

        # Construir query
        query = f"SELECT * FROM {table_name}"
        conditions = []

        if start_date:
            conditions.append(f"{date_column} >= '{start_date}'")
        if end_date:
            conditions.append(f"{date_column} <= '{end_date}'")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += f" ORDER BY {date_column}"

        if limit:
            query += f" LIMIT {limit}"

        logger.info(f"Ejecutando query: {query[:200]}...")
        df = pd.read_sql(query, engine)
        logger.info(f"Cargadas {len(df)} filas desde {table_name}")

        engine.dispose()
        return df

    except Exception as e:
        logger.error(f"Error al cargar denuncias: {e}")
        return pd.DataFrame()


def load_denuncias_aggregated(
    table_name: str = "denuncias",
    district_column: str = "distrito",
    date_column: str = "fecha_denuncia",
    crime_type: Optional[str] = None,
    crime_column: str = "tipo_delito",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Carga denuncias ya agregadas por distrito y fecha.

    Esta función es más eficiente que cargar todo y luego agregar.

    Args:
        table_name: Nombre de la tabla de denuncias
        district_column: Nombre de la columna de distrito
        date_column: Nombre de la columna de fecha
        crime_type: Tipo de delito específico a filtrar (ej: 'EXTORSION')
        crime_column: Nombre de la columna de tipo de delito
        start_date: Fecha inicio (formato 'YYYY-MM-DD')
        end_date: Fecha fin (formato 'YYYY-MM-DD')

    Returns:
        DataFrame con columnas: district, date, count
    """
    try:
        engine = get_sqlalchemy_engine()

        # Construir query de agregación
        query = f"""
        SELECT
            {district_column} as district,
            DATE({date_column}) as date,
            COUNT(*) as count
        FROM {table_name}
        WHERE 1=1
        """

        if crime_type:
            query += f" AND {crime_column} = '{crime_type}'"
        if start_date:
            query += f" AND {date_column} >= '{start_date}'"
        if end_date:
            query += f" AND {date_column} <= '{end_date}'"

        query += f"""
        GROUP BY {district_column}, DATE({date_column})
        ORDER BY date, district
        """

        logger.info(f"Ejecutando query agregada: {query[:300]}...")
        df = pd.read_sql(query, engine)
        logger.info(f"Cargadas {len(df)} filas agregadas")

        engine.dispose()
        return df

    except Exception as e:
        logger.error(f"Error al cargar denuncias agregadas: {e}")
        return pd.DataFrame()


def execute_custom_query(query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Ejecuta una query SQL personalizada y retorna los resultados.

    Args:
        query: Query SQL a ejecutar
        params: Diccionario de parámetros para la query (opcional)

    Returns:
        DataFrame con los resultados
    """
    try:
        engine = get_sqlalchemy_engine()

        if params:
            df = pd.read_sql(text(query), engine, params=params)
        else:
            df = pd.read_sql(query, engine)

        logger.info(f"Query ejecutada exitosamente, {len(df)} filas retornadas")
        engine.dispose()
        return df

    except Exception as e:
        logger.error(f"Error al ejecutar query: {e}")
        return pd.DataFrame()


def get_available_crime_types(
    table_name: str = "denuncias",
    crime_column: str = "tipo_delito"
) -> List[str]:
    """Obtiene la lista de tipos de delitos disponibles en la base de datos.

    Args:
        table_name: Nombre de la tabla de denuncias
        crime_column: Nombre de la columna de tipo de delito

    Returns:
        Lista de tipos de delitos únicos
    """
    try:
        engine = get_sqlalchemy_engine()
        query = f"SELECT DISTINCT {crime_column} FROM {table_name} ORDER BY {crime_column}"
        df = pd.read_sql(query, engine)
        engine.dispose()

        crime_types = df[crime_column].dropna().tolist()
        logger.info(f"Tipos de delitos encontrados: {len(crime_types)}")
        return crime_types

    except Exception as e:
        logger.error(f"Error al obtener tipos de delito: {e}")
        return []


def get_available_districts(
    table_name: str = "denuncias",
    district_column: str = "distrito"
) -> List[str]:
    """Obtiene la lista de distritos disponibles en la base de datos.

    Args:
        table_name: Nombre de la tabla de denuncias
        district_column: Nombre de la columna de distrito

    Returns:
        Lista de distritos únicos
    """
    try:
        engine = get_sqlalchemy_engine()
        query = f"SELECT DISTINCT {district_column} FROM {table_name} ORDER BY {district_column}"
        df = pd.read_sql(query, engine)
        engine.dispose()

        districts = df[district_column].dropna().tolist()
        logger.info(f"Distritos encontrados: {len(districts)}")
        return districts

    except Exception as e:
        logger.error(f"Error al obtener distritos: {e}")
        return []


def get_date_range(
    table_name: str = "denuncias",
    date_column: str = "fecha_denuncia"
) -> Dict[str, str]:
    """Obtiene el rango de fechas disponible en la base de datos.

    Args:
        table_name: Nombre de la tabla de denuncias
        date_column: Nombre de la columna de fecha

    Returns:
        Dict con 'min_date' y 'max_date' en formato 'YYYY-MM-DD'
    """
    try:
        engine = get_sqlalchemy_engine()
        query = f"""
        SELECT
            MIN(DATE({date_column})) as min_date,
            MAX(DATE({date_column})) as max_date
        FROM {table_name}
        """
        df = pd.read_sql(query, engine)
        engine.dispose()

        result = {
            'min_date': str(df['min_date'].iloc[0]) if not df.empty else None,
            'max_date': str(df['max_date'].iloc[0]) if not df.empty else None
        }
        logger.info(f"Rango de fechas: {result}")
        return result

    except Exception as e:
        logger.error(f"Error al obtener rango de fechas: {e}")
        return {'min_date': None, 'max_date': None}
