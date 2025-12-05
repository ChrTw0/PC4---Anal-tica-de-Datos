"""Script rápido para ver la estructura de la tabla denuncias."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import database

# Ver tablas disponibles
print("=== TABLAS DISPONIBLES ===")
tables = database.get_table_names()
for t in tables:
    print(f"  - {t}")

# Ver estructura de la tabla denuncias
print("\n=== ESTRUCTURA DE LA TABLA 'denuncias' ===")
info = database.get_table_info("denuncias")
print(info.to_string(index=False))

# Ver primeras filas
print("\n=== PRIMERAS 5 FILAS ===")
import pandas as pd
engine = database.get_sqlalchemy_engine()
df = pd.read_sql("SELECT * FROM denuncias LIMIT 5", engine)
engine.dispose()
print(df.to_string(index=False))

print("\n=== COLUMNAS DETECTADAS ===")
print(list(df.columns))
