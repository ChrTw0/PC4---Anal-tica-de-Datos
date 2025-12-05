"""Análisis Exploratorio de Datos para detectar problemas.

Este script analiza los datos antes del entrenamiento para identificar:
- Desbalance de clases extremo
- Data leakage potencial
- Distribución de features
- Correlaciones sospechosas
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import config_mysql as config
import data_mysql as data

print("="*80)
print("ANÁLISIS EXPLORATORIO DE DATOS - DETECCIÓN DE PROBLEMAS")
print("="*80)

# 1. Cargar datos
print("\n[1/7] Cargando datos desde MySQL...")
df_raw = data.load_from_mysql()
print(f"   ✓ Datos cargados: {len(df_raw)} filas, {df_raw['district'].nunique()} distritos")

# 2. Crear panel
print("\n[2/7] Creando panel distrito-semana...")
panel = data.make_panel_district_week(df_raw)
print(f"   ✓ Panel creado: {len(panel)} filas")

# 3. Split
print("\n[3/7] Dividiendo train/val/test...")
train_df, val_df, test_df = data.train_val_test_split_time(panel)
X_train, y_train, feature_cols = data.get_feature_target_arrays(train_df, task="classification")
X_test, y_test, _ = data.get_feature_target_arrays(test_df, task="classification")

print(f"   Features: {feature_cols}")
print(f"   X_train: {X_train.shape}, X_test: {X_test.shape}")

# 4. ANÁLISIS DE DESBALANCE DE CLASES
print("\n[4/7] ANÁLISIS DE DESBALANCE DE CLASES")
print("-"*80)

total_train = len(y_train)
positivos_train = y_train.sum()
negativos_train = total_train - positivos_train
ratio_train = positivos_train / total_train

total_test = len(y_test)
positivos_test = y_test.sum()
negativos_test = total_test - positivos_test
ratio_test = positivos_test / total_test

print(f"TRAIN SET:")
print(f"  Total:      {total_train:,}")
print(f"  Positivos:  {int(positivos_train):,} ({ratio_train*100:.2f}%)")
print(f"  Negativos:  {int(negativos_train):,} ({(1-ratio_train)*100:.2f}%)")
print(f"  Ratio:      1:{negativos_train/max(positivos_train,1):.1f}")

print(f"\nTEST SET:")
print(f"  Total:      {total_test:,}")
print(f"  Positivos:  {int(positivos_test):,} ({ratio_test*100:.2f}%)")
print(f"  Negativos:  {int(negativos_test):,} ({(1-ratio_test)*100:.2f}%)")
print(f"  Ratio:      1:{negativos_test/max(positivos_test,1):.1f}")

if ratio_train > 0.95 or ratio_train < 0.05:
    print("\n⚠️  ALERTA: Desbalance EXTREMO de clases detectado!")
    print("    El modelo puede aprender a predecir siempre la clase mayoritaria.")

# 5. ANÁLISIS DE CORRELACIONES
print("\n[5/7] ANÁLISIS DE CORRELACIONES")
print("-"*80)

# Crear DataFrame con features y target
df_analysis = pd.DataFrame(X_train, columns=feature_cols)
df_analysis['target'] = y_train

# Correlaciones con el target
correlations = df_analysis.corr()['target'].drop('target').sort_values(ascending=False)
print("\nCorrelaciones con el target:")
for feat, corr in correlations.items():
    print(f"  {feat:20s}: {corr:+.4f}")

if correlations.max() > 0.99:
    print("\n⚠️  ALERTA: Correlación casi perfecta detectada (>0.99)!")
    print("    Posible data leakage o feature que predice directamente el target.")

# 6. DISTRIBUCIÓN DE FEATURES
print("\n[6/7] DISTRIBUCIÓN DE FEATURES")
print("-"*80)

for i, feat in enumerate(feature_cols):
    values = X_train[:, i]
    print(f"\n{feat}:")
    print(f"  Min:    {values.min():.2f}")
    print(f"  Max:    {values.max():.2f}")
    print(f"  Mean:   {values.mean():.2f}")
    print(f"  Median: {np.median(values):.2f}")
    print(f"  Std:    {values.std():.2f}")
    print(f"  Zeros:  {(values == 0).sum()} ({(values == 0).sum()/len(values)*100:.1f}%)")
    print(f"  NaNs:   {np.isnan(values).sum()}")

# 7. VERIFICAR SEPARABILIDAD
print("\n[7/7] VERIFICACIÓN DE SEPARABILIDAD")
print("-"*80)

# Para cada feature, ver si separa perfectamente las clases
for i, feat in enumerate(feature_cols):
    pos_values = X_train[y_train == 1, i]
    neg_values = X_train[y_train == 0, i]

    if len(pos_values) > 0 and len(neg_values) > 0:
        # Ver si hay overlap
        pos_min, pos_max = pos_values.min(), pos_values.max()
        neg_min, neg_max = neg_values.min(), neg_values.max()

        overlap = not (pos_max < neg_min or neg_max < pos_min)

        print(f"\n{feat}:")
        print(f"  Clase 0: [{neg_min:.2f}, {neg_max:.2f}]")
        print(f"  Clase 1: [{pos_min:.2f}, {pos_max:.2f}]")
        print(f"  Overlap: {'SÍ' if overlap else 'NO ⚠️'}")

        if not overlap:
            print(f"  ⚠️  ALERTA: Feature separa perfectamente las clases!")
            print(f"      Esto explica PR-AUC = 1.0")

# 8. GENERAR GRÁFICOS
print("\n[8/8] Generando gráficos de diagnóstico...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Análisis Exploratorio de Datos', fontsize=16)

# 8.1 Distribución del target
ax = axes[0, 0]
counts = pd.Series(y_train).value_counts()
ax.bar(['Negativo', 'Positivo'], [counts.get(0.0, 0), counts.get(1.0, 0)])
ax.set_title('Distribución del Target (Train)')
ax.set_ylabel('Cantidad')
for i, v in enumerate([counts.get(0.0, 0), counts.get(1.0, 0)]):
    ax.text(i, v, f'{int(v):,}\n({v/len(y_train)*100:.1f}%)', ha='center', va='bottom')

# 8.2 Correlaciones
ax = axes[0, 1]
correlations.plot(kind='barh', ax=ax)
ax.set_title('Correlaciones con Target')
ax.set_xlabel('Correlación')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

# 8.3 Matriz de correlación completa
ax = axes[0, 2]
corr_matrix = df_analysis.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('Matriz de Correlación')

# 8.4-8.6 Distribuciones de features top 3
for idx, feat in enumerate(feature_cols[:3]):
    ax = axes[1, idx]
    i = feature_cols.index(feat)

    pos_vals = X_train[y_train == 1, i]
    neg_vals = X_train[y_train == 0, i]

    ax.hist(neg_vals, bins=50, alpha=0.5, label='Clase 0', density=True)
    ax.hist(pos_vals, bins=50, alpha=0.5, label='Clase 1', density=True)
    ax.set_title(f'Distribución: {feat}')
    ax.set_xlabel('Valor')
    ax.set_ylabel('Densidad')
    ax.legend()

plt.tight_layout()
plt.savefig('reports/data_analysis.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Gráfico guardado en: reports/data_analysis.png")

print("\n" + "="*80)
print("ANÁLISIS COMPLETADO")
print("="*80)
print("\nRevisa los resultados arriba para identificar el problema.")
print("Busca especialmente:")
print("  - Desbalance extremo de clases")
print("  - Correlaciones > 0.99")
print("  - Features que separan perfectamente las clases")
print("  - Features con muchos NaNs o zeros")
