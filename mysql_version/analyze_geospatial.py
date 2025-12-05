"""Análisis de Datos Geoespaciales para Extorsión.

Este script analiza las coordenadas (lat, long) de extorsiones
para crear un modelo de predicción de hotspots geoespaciales.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import database

print("="*80)
print("ANÁLISIS GEOESPACIAL DE EXTORSIÓN - LIMA METROPOLITANA + CALLAO")
print("="*80)

# 1. Cargar datos de extorsión con coordenadas
print("\n[1/6] Cargando datos de EXTORSIÓN (Lima + Callao) desde MySQL...")

engine = database.get_sqlalchemy_engine()

# Query para obtener extorsiones con coordenadas
# Nota: modalidad_hecho contiene el tipo de delito específico
# Filtrado: Solo Lima Metropolitana + Callao (región metropolitana completa)
query = """
SELECT
    fecha_hora_hecho,
    distrito_hecho,
    departamento_hecho,
    lat_hecho,
    long_hecho,
    materia_hecho,
    modalidad_hecho,
    subtipo_hecho
FROM denuncias
WHERE modalidad_hecho = 'EXTORSION'
    AND departamento_hecho IN ('LIMA', 'CALLAO')
    AND lat_hecho IS NOT NULL
    AND long_hecho IS NOT NULL
    AND lat_hecho != 0
    AND long_hecho != 0
    AND fecha_hora_hecho >= '2020-01-01'
ORDER BY fecha_hora_hecho DESC
LIMIT 100000
"""

df = pd.read_sql(query, engine)
engine.dispose()

print(f"   ✓ Cargados {len(df):,} casos de extorsión con coordenadas")

# 2. Análisis básico
print("\n[2/6] ESTADÍSTICAS BÁSICAS")
print("-"*80)

print(f"Fechas:")
print(f"  Desde: {df['fecha_hora_hecho'].min()}")
print(f"  Hasta: {df['fecha_hora_hecho'].max()}")

print(f"\nDistritos únicos: {df['distrito_hecho'].nunique()}")
print(f"Top 10 distritos con más extorsiones:")
top_districts = df['distrito_hecho'].value_counts().head(10)
for dist, count in top_districts.items():
    print(f"  {dist:30s}: {count:,}")

# 3. Análisis de coordenadas
print("\n[3/6] ANÁLISIS DE COORDENADAS")
print("-"*80)

print(f"Latitud:")
print(f"  Min:    {df['lat_hecho'].min():.6f}")
print(f"  Max:    {df['lat_hecho'].max():.6f}")
print(f"  Media:  {df['lat_hecho'].mean():.6f}")
print(f"  Std:    {df['lat_hecho'].std():.6f}")

print(f"\nLongitud:")
print(f"  Min:    {df['long_hecho'].min():.6f}")
print(f"  Max:    {df['long_hecho'].max():.6f}")
print(f"  Media:  {df['long_hecho'].mean():.6f}")
print(f"  Std:    {df['long_hecho'].std():.6f}")

# Detectar outliers (fuera de Lima Metropolitana + Callao)
# Lima aproximado: lat [-12.5, -11.5], lon [-77.5, -76.5]
lima_lat_range = (-12.6, -11.4)  # Lima + Callao con margen
lima_lon_range = (-77.5, -76.5)

outliers = df[
    (df['lat_hecho'] < lima_lat_range[0]) |
    (df['lat_hecho'] > lima_lat_range[1]) |
    (df['long_hecho'] < lima_lon_range[0]) |
    (df['long_hecho'] > lima_lon_range[1])
]

print(f"\n⚠️  Outliers (fuera de Lima-Callao): {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
if len(outliers) > 0:
    print("   Primeros 5 outliers:")
    print(outliers[['lat_hecho', 'long_hecho', 'distrito_hecho']].head())

# Filtrar outliers
df_clean = df[
    (df['lat_hecho'] >= lima_lat_range[0]) &
    (df['lat_hecho'] <= lima_lat_range[1]) &
    (df['long_hecho'] >= lima_lon_range[0]) &
    (df['long_hecho'] <= lima_lon_range[1])
].copy()

print(f"   ✓ Datos limpios: {len(df_clean):,} casos")

# 4. Crear Grids Espaciales
print("\n[4/6] CREACIÓN DE GRIDS ESPACIALES")
print("-"*80)

# Grid de ~1km x 1km (aprox 0.01 grados)
grid_size = 0.01  # grados (~1.1 km en latitud)

df_clean['grid_lat'] = (df_clean['lat_hecho'] / grid_size).round() * grid_size
df_clean['grid_lon'] = (df_clean['long_hecho'] / grid_size).round() * grid_size
df_clean['grid_id'] = df_clean['grid_lat'].astype(str) + '_' + df_clean['grid_lon'].astype(str)

print(f"Grid size: {grid_size} grados (~{grid_size*111:.1f} km)")
print(f"Total grids únicos: {df_clean['grid_id'].nunique():,}")

# Contar extorsiones por grid
grid_counts = df_clean['grid_id'].value_counts()

print(f"\nDistribución de extorsiones por grid:")
print(f"  Media:   {grid_counts.mean():.2f}")
print(f"  Mediana: {grid_counts.median():.0f}")
print(f"  Max:     {grid_counts.max()}")
print(f"  Min:     {grid_counts.min()}")

print(f"\nTop 10 grids con más extorsiones:")
for i, (grid_id, count) in enumerate(grid_counts.head(10).items(), 1):
    lat, lon = grid_id.split('_')
    print(f"  {i:2d}. Grid [{lat}, {lon}]: {count:,} casos")

# 5. Análisis Temporal por Grid
print("\n[5/6] ANÁLISIS TEMPORAL")
print("-"*80)

df_clean['fecha'] = pd.to_datetime(df_clean['fecha_hora_hecho']).dt.date
df_clean['semana'] = pd.to_datetime(df_clean['fecha_hora_hecho']).dt.to_period('W-MON')

# Contar por grid-semana
grid_week = df_clean.groupby(['grid_id', 'semana']).size().reset_index(name='count')

print(f"Total grid-semana observaciones: {len(grid_week):,}")
print(f"Semanas únicas: {df_clean['semana'].nunique()}")
print(f"Promedio de extorsiones por grid-semana: {grid_week['count'].mean():.2f}")

# Estadísticas de grids activos
grids_per_week = grid_week.groupby('semana')['grid_id'].nunique()
print(f"\nGrids activos por semana:")
print(f"  Media:   {grids_per_week.mean():.1f}")
print(f"  Mediana: {grids_per_week.median():.0f}")
print(f"  Min:     {grids_per_week.min()}")
print(f"  Max:     {grids_per_week.max()}")

# 6. Visualización
print("\n[6/6] Generando visualizaciones...")

import os
os.makedirs('../reports', exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 6.1 Mapa de puntos
ax = axes[0, 0]
scatter = ax.scatter(df_clean['long_hecho'], df_clean['lat_hecho'],
                     c='red', alpha=0.1, s=1)
ax.set_xlabel('Longitud')
ax.set_ylabel('Latitud')
ax.set_title(f'Distribución Geográfica de Extorsiones (n={len(df_clean):,})')
ax.grid(True, alpha=0.3)

# 6.2 Mapa de calor (hexbin)
ax = axes[0, 1]
hexbin = ax.hexbin(df_clean['long_hecho'], df_clean['lat_hecho'],
                   gridsize=50, cmap='YlOrRd', mincnt=1)
ax.set_xlabel('Longitud')
ax.set_ylabel('Latitud')
ax.set_title('Mapa de Calor de Extorsiones (Hexbin)')
plt.colorbar(hexbin, ax=ax, label='Número de casos')
ax.grid(True, alpha=0.3)

# 6.3 Top 20 grids
ax = axes[1, 0]
grid_counts.head(20).plot(kind='barh', ax=ax)
ax.set_xlabel('Número de Extorsiones')
ax.set_ylabel('Grid ID')
ax.set_title('Top 20 Grids con Más Extorsiones')
ax.invert_yaxis()

# 6.4 Serie temporal de total de extorsiones
ax = axes[1, 1]
weekly_totals = df_clean.groupby('semana').size()
weekly_totals.plot(ax=ax, color='darkred', linewidth=2)
ax.set_xlabel('Semana')
ax.set_ylabel('Número de Extorsiones')
ax.set_title('Evolución Temporal de Extorsiones')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../reports/geospatial_analysis_lima.png', dpi=150, bbox_inches='tight')
print("   ✓ Visualización guardada en: reports/geospatial_analysis_lima.png")

# Guardar datos procesados para siguiente paso
df_clean.to_csv('../reports/extorsion_geospatial_lima_clean.csv', index=False)
print(f"   ✓ Datos limpios guardados en: reports/extorsion_geospatial_lima_clean.csv")

print("\n" + "="*80)
print("ANÁLISIS COMPLETADO")
print("="*80)
print(f"\nResumen:")
print(f"  - {len(df_clean):,} casos de extorsión con coordenadas válidas")
print(f"  - {df_clean['grid_id'].nunique():,} grids espaciales únicos")
print(f"  - {df_clean['distrito_hecho'].nunique()} distritos afectados")
print(f"  - Grid size: ~{grid_size*111:.1f} km")
print(f"\nPróximo paso: Crear modelo de predicción de hotspots geoespaciales")
