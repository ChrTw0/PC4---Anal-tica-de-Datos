"""Script para detectar automáticamente los nombres correctos de columnas.

Este script analiza la tabla 'denuncias' y sugiere qué columnas usar
para distrito, fecha y tipo de delito.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import database

print("="*70)
print("DETECCIÓN AUTOMÁTICA DE COLUMNAS")
print("="*70)

# Obtener estructura de la tabla
print("\n1. ESTRUCTURA DE LA TABLA 'denuncias':")
print("-" * 70)
info = database.get_table_info("denuncias")
print(info.to_string(index=False))

# Cargar muestra de datos
print("\n2. MUESTRA DE DATOS (primeras 3 filas):")
print("-" * 70)
engine = database.get_sqlalchemy_engine()
df_sample = pd.read_sql("SELECT * FROM denuncias LIMIT 3", engine)
engine.dispose()

print(df_sample.to_string(index=False))

# Analizar columnas
print("\n3. ANÁLISIS DE COLUMNAS:")
print("-" * 70)

columns = list(df_sample.columns)
print(f"Total de columnas: {len(columns)}")
print(f"Nombres: {columns}")

# Detectar columna de distrito
print("\n4. DETECCIÓN DE COLUMNA DE DISTRITO:")
district_candidates = [c for c in columns if any(x in c.lower() for x in ['distrito', 'district', 'ubigeo', 'lugar', 'localidad'])]
if district_candidates:
    print(f"   ✓ Candidatos encontrados: {district_candidates}")
    print(f"   → Sugerencia: MYSQL_DISTRICT_COLUMN = '{district_candidates[0]}'")
else:
    print("   ✗ No se encontró columna obvia de distrito")
    print("   → Revisa manualmente las columnas disponibles")

# Detectar columna de fecha
print("\n5. DETECCIÓN DE COLUMNA DE FECHA:")
date_candidates = [c for c in columns if any(x in c.lower() for x in ['fecha', 'date', 'dia', 'day', 'timestamp'])]
if date_candidates:
    print(f"   ✓ Candidatos encontrados: {date_candidates}")
    print(f"   → Sugerencia: MYSQL_DATE_COLUMN = '{date_candidates[0]}'")
else:
    print("   ✗ No se encontró columna obvia de fecha")

# Detectar columna de tipo de delito
print("\n6. DETECCIÓN DE COLUMNA DE TIPO DE DELITO:")
crime_candidates = [c for c in columns if any(x in c.lower() for x in ['delito', 'crime', 'tipo', 'type', 'categoria', 'materia'])]
if crime_candidates:
    print(f"   ✓ Candidatos encontrados: {crime_candidates}")
    print(f"   → Sugerencia: MYSQL_CRIME_TYPE_COLUMN = '{crime_candidates[0]}'")

    # Ver valores únicos del primer candidato
    engine = database.get_sqlalchemy_engine()
    query = f"SELECT DISTINCT {crime_candidates[0]} FROM denuncias LIMIT 20"
    df_crimes = pd.read_sql(query, engine)
    engine.dispose()

    print(f"\n   Valores únicos en '{crime_candidates[0]}' (primeros 20):")
    for val in df_crimes[crime_candidates[0]].dropna():
        print(f"      - {val}")
else:
    print("   ✗ No se encontró columna obvia de tipo de delito")

print("\n" + "="*70)
print("CONFIGURACIÓN SUGERIDA PARA config_mysql.py:")
print("="*70)

if district_candidates:
    print(f"MYSQL_DISTRICT_COLUMN = '{district_candidates[0]}'")
if date_candidates:
    print(f"MYSQL_DATE_COLUMN = '{date_candidates[0]}'")
if crime_candidates:
    print(f"MYSQL_CRIME_TYPE_COLUMN = '{crime_candidates[0]}'")

print("\nCopia estas líneas en config_mysql.py y ajusta si es necesario.")
print("="*70)
