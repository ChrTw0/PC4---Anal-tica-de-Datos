"""Configuración específica para MySQL.

Este archivo extiende config.py con parámetros para conexión a MySQL.
Importa todo de config.py y agrega configuración de base de datos.
"""

import sys
from pathlib import Path

# Agregar src al path para importar config
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import *  # Importa toda la configuración base

# ==========================================
# Configuración de MySQL
# ==========================================

MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PASSWORD = "1234"
MYSQL_DATABASE = "denuncias_peru"

# ==========================================
# Configuración de columnas de MySQL
# ==========================================

# Nombres de columnas esperadas en la tabla de denuncias
MYSQL_TABLE_NAME = "denuncias"
MYSQL_DISTRICT_COLUMN = "distrito_hecho"
MYSQL_DATE_COLUMN = "fecha_hora_hecho"
MYSQL_CRIME_TYPE_COLUMN = "materia_hecho"  # Puedes cambiar a "subtipo_hecho" si prefieres más detalle

# Tipo de delito específico a analizar (None = todos)
# Ejemplos: "EXTORSION", "ROBO AGRAVADO", "HURTO SIMPLE", etc.
MYSQL_CRIME_TYPE_FILTER = None  # Cambia a None para ver todos o especifica uno

# Rango de fechas (None = todas las fechas disponibles)
MYSQL_START_DATE = None  # Formato: "2020-01-01"
MYSQL_END_DATE = None    # Formato: "2023-12-31"

# Límite de filas para pruebas rápidas (None = sin límite)
MYSQL_LIMIT_ROWS = None
