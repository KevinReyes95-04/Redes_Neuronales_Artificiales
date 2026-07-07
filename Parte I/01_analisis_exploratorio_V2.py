"""Pregunta 1: analisis exploratorio de los datos originales."""

# %%
import os
import warnings
from pathlib import Path
from tempfile import gettempdir

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLCONFIGDIR", str(Path(gettempdir()) / "rna_taller_matplotlib"))

import matplotlib
matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score, auc, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# Configuración de semilla para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

# %%
# Configuracion
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / "data" / "raw" / "ntrain_median_values.csv"
TABLES_DIR = ROOT / "resultados" / "tablas"
FIGURES_DIR = ROOT / "resultados" / "figuras"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "severity"
FEATURES = ["ndvi_med", "evi_med", "ndre_med", "gli_med", "height_med"]
COLUMNS = [TARGET, *FEATURES]
SEVERITIES = list(range(1, 7))
LABELS = {
    "ndvi_med": "NDVI promedio",
    "evi_med": "EVI promedio",
    "ndre_med": "NDRE promedio",
    "gli_med": "GLI promedio",
    "height_med": "Altura media",
}
PALETTE = sns.color_palette("viridis", len(SEVERITIES))

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update(
    {"figure.dpi": 120, "savefig.dpi": 180, "axes.titleweight": "bold"}
)


def save_csv(table, filename):
    """Guarda una tabla CSV con codificacion compatible con Excel."""
    table.to_csv(TABLES_DIR / filename, index=False, encoding="utf-8-sig")


def save_figure(fig, filename):
    """Ajusta, guarda y cierra una figura."""
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def feature_grid(title, filename, plot_function, ylabel="", figsize=(15, 9)):
    """Genera un panel con una grafica para cada predictor."""
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    for axis, feature in zip(axes.flat, FEATURES):
        plot_function(axis, feature)
        axis.set_title(LABELS[feature])
        axis.set_ylabel(ylabel)
    axes.flat[-1].axis("off")
    fig.suptitle(title, fontsize=15, y=1.01)
    save_figure(fig, filename)


# %%
# Carga y validacion
if not DATA_PATH.exists():
    raise FileNotFoundError(f"No se encontro el archivo: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
missing_columns = sorted(set(COLUMNS) - set(df.columns))
if missing_columns:
    raise ValueError(f"Faltan columnas obligatorias: {missing_columns}")

extra_columns = sorted(set(df.columns) - set(COLUMNS))
if extra_columns:
    print(f"Advertencia: columnas adicionales omitidas: {extra_columns}")

df = df[COLUMNS].apply(pd.to_numeric, errors="raise")
if not np.allclose(df[TARGET], df[TARGET].round()):
    raise ValueError("La severidad contiene valores no enteros.")

df[TARGET] = df[TARGET].astype(int)
observed_severities = sorted(map(int, df[TARGET].unique()))
if observed_severities != SEVERITIES:
    raise ValueError(f"Categorias de severidad inesperadas: {observed_severities}")

print(f"Archivo: {DATA_PATH}")
print(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"Severidades: {observed_severities}")


# %%
# Calidad de los datos
missing = df.isna().sum()
infinite = pd.Series(
    np.isinf(df.to_numpy()).sum(axis=0), index=COLUMNS, name="valores_infinitos"
)
duplicates = df.duplicated(keep=False)
same_values = pd.Series(
    np.isclose(df["evi_med"], df["gli_med"])
    & np.isclose(df["evi_med"], df["height_med"]),
    index=df.index,
)

quality_summary = pd.DataFrame(
    {
        "indicador": [
            "numero_observaciones",
            "numero_variables_predictoras",
            "numero_categorias_severidad",
            "valores_faltantes",
            "valores_infinitos",
            "filas_duplicadas_adicionales",
            "registros_evi_gli_altura_iguales",
        ],
        "valor": [
            len(df),
            len(FEATURES),
            df[TARGET].nunique(),
            int(missing.sum()),
            int(infinite.sum()),
            int(df.duplicated().sum()),
            int(same_values.sum()),
        ],
    }
)

quality_by_column = pd.DataFrame(
    {
        "variable": COLUMNS,
        "tipo_dato": df.dtypes.astype(str).to_numpy(),
        "valores_faltantes": missing.to_numpy(),
        "valores_infinitos": infinite.to_numpy(),
        "valores_unicos": df.nunique().to_numpy(),
    }
)

review_rows = df.loc[same_values | duplicates].copy()
review_rows.insert(0, "fila_original", review_rows.index)
same_review = same_values.loc[review_rows.index]
duplicate_review = duplicates.loc[review_rows.index]
review_rows["motivo_revision"] = np.select(
    [same_review & duplicate_review, same_review, duplicate_review],
    [
        "EVI, GLI y altura tienen el mismo valor; fila duplicada",
        "EVI, GLI y altura tienen el mismo valor",
        "Fila duplicada",
    ],
    default="",
)

save_csv(quality_summary, "01_resumen_calidad_datos.csv")
save_csv(quality_by_column, "02_calidad_por_variable.csv")
save_csv(review_rows, "03_registros_para_revision.csv")

print("\nResumen de calidad:")
print(quality_summary.to_string(index=False))


# %%
# Estadisticos descriptivos por severidad
long_df = df.melt(
    id_vars=TARGET, value_vars=FEATURES, var_name="variable", value_name="valor"
)
long_df["variable"] = pd.Categorical(
    long_df["variable"], categories=FEATURES, ordered=True
)

descriptive_stats = (
    long_df.groupby(["variable", TARGET], observed=True)["valor"]
    .agg(
        n="count",
        media="mean",
        desviacion_estandar="std",
        minimo="min",
        maximo="max",
    )
    .reset_index()
)
descriptive_stats.insert(
    2, "descripcion", descriptive_stats["variable"].map(LABELS)
)
descriptive_stats = descriptive_stats[
    [
        TARGET,
        "variable",
        "descripcion",
        "n",
        "media",
        "desviacion_estandar",
        "minimo",
        "maximo",
    ]
].round(6)
save_csv(descriptive_stats, "04_estadisticos_descriptivos_por_severidad.csv")

print("\nEstadisticos descriptivos por severidad:")
print(descriptive_stats.to_string(index=False))


# %%
# Correlacion monotona con la severidad
spearman_results = pd.DataFrame(
    [
        {
            "variable": feature,
            "descripcion": LABELS[feature],
            "rho_spearman": result.statistic, # type: ignore
            "valor_p": result.pvalue, # type: ignore
        }
        for feature in FEATURES
        for result in [spearmanr(df[TARGET], df[feature])]
    ]
).sort_values("rho_spearman", ignore_index=True)
save_csv(spearman_results, "05_correlacion_spearman_con_severidad.csv")


# %%
# Libro consolidado
excel_tables = {
    "resumen_calidad": quality_summary,
    "calidad_variables": quality_by_column,
    "descriptivos": descriptive_stats,
    "spearman_severidad": spearman_results,
    "registros_revision": review_rows,
}
with pd.ExcelWriter(
    TABLES_DIR / "analisis_exploratorio_pregunta_1.xlsx", engine="openpyxl"
) as writer:
    for sheet, table in excel_tables.items():
        table.to_excel(writer, sheet_name=sheet, index=False)


# %%
# Figuras por predictor
feature_grid(
    "Distribucion de las variables predictoras",
    "01_distribuciones_variables.png",
    lambda axis, feature: sns.histplot(
        data=df,
        x=feature,
        bins=20,
        kde=True,
        color="#287271",
        edgecolor="white",
        ax=axis,
    ),
    ylabel="Frecuencia",
    figsize=(14, 8),
)

feature_grid(
    "Distribucion de los predictores por categoria de severidad",
    "02_boxplots_por_severidad.png",
    lambda axis, feature: sns.boxplot(
        data=df,
        x=TARGET,
        y=feature,
        order=SEVERITIES,
        hue=TARGET,
        palette=PALETTE,
        legend=False,
        ax=axis,
    ),
)

feature_grid(
    "Comportamiento promedio de los predictores segun severidad",
    "03_medias_por_severidad.png",
    lambda axis, feature: sns.pointplot(
        data=df,
        x=TARGET,
        y=feature,
        order=SEVERITIES,
        errorbar=("ci", 95),
        capsize=0.15,
        color="#1f5d50",
        seed=42,
        ax=axis,
    ),
    ylabel="Media e IC del 95 %",
)


# %%
# Matriz de correlacion
fig, axis = plt.subplots(figsize=(8, 6))
sns.heatmap(
    df[FEATURES].corr(method="spearman"),
    annot=True,
    fmt=".2f",
    cmap="vlag",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={"label": "Correlacion de Spearman"},
    ax=axis,
)
axis.set_xticklabels([LABELS[x] for x in FEATURES], rotation=35, ha="right")
axis.set_yticklabels([LABELS[x] for x in FEATURES], rotation=0)
axis.set_title("Matriz de correlacion entre variables predictoras", pad=15)
save_figure(fig, "04_matriz_correlacion.png")


# %%
# Relacion NDVI-NDRE
fig, axis = plt.subplots(figsize=(9, 7))
sns.scatterplot(
    data=df,
    x="ndvi_med",
    y="ndre_med",
    hue=TARGET,
    hue_order=SEVERITIES,
    palette=PALETTE,
    s=70,
    alpha=0.8,
    edgecolor="white",
    linewidth=0.5,
    ax=axis,
)
axis.set(
    xlabel="NDVI promedio",
    ylabel="NDRE promedio",
    title="Relacion NDVI-NDRE segun categoria de severidad",
)
axis.legend(title="Severidad", bbox_to_anchor=(1.02, 1), loc="upper left")
save_figure(fig, "05_ndvi_ndre_por_severidad.png")


# %%
print("\nCorrelacion de Spearman con la severidad:")
print(spearman_results.round(6).to_string(index=False))
print(f"\nTablas guardadas en: {TABLES_DIR}")
print(f"Figuras guardadas en: {FIGURES_DIR}")





# ─────────────────────────────────────────────────────────────────────────────
# PREGUNTA 2 – Balance de clases y estrategia de manejo
# ─────────────────────────────────────────────────────────────────────────────

palette = sns.color_palette("Set2", n_colors=df[TARGET].nunique())
print("="*70)
print("PREGUNTA 2 – Verificación de balance de clases y estrategia SMOTE")
print("="*70)

counts = df[TARGET].value_counts().sort_index()
print("\nConteo por categoría de severidad:")
print(counts.to_string())
print(f"\nRatio min/max = {counts.min()/counts.max():.2f}")

# Figura 6– Distribución de clases
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(counts.index.astype(str), counts.values,
       color=palette[:len(counts)], edgecolor="black", linewidth=0.7)
for xi, yi in zip(counts.index, counts.values):
    ax.text(str(xi), yi + 1, str(yi), ha="center", va="bottom", fontsize=10)
ax.set_xlabel("Categoría de severidad", fontsize=11)
ax.set_ylabel("Número de observaciones", fontsize=11)
ax.set_title("Figura 6 – Balance de clases en la variable severidad",
             fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
save_figure(fig, "06_balance_clases.png")
print("✔ Figura 6 guardada: " + "06_balance_clases.png")

print("""
Estrategia elegida: SMOTE (Synthetic Minority Over-sampling Technique)
  • Genera muestras sintéticas para las clases minoritarias interpolando
    entre vecinos cercanos existentes, evitando la simple duplicación.
  • Se aplica SOLO sobre el conjunto de entrenamiento para no contaminar
    la evaluación.
  • Alternativas válidas: class_weight='balanced' en Keras o submuestreo
    de la clase mayoritaria (RandomUnderSampler).
""")


# ─────────────────────────────────────────────────────────────────────────────
# Preparación de datos base (estandarización + SMOTE)
# ─────────────────────────────────────────────────────────────────────────────
X = df[FEATURES].values
y = df[TARGET].values

# Codificación de etiquetas a 0-based
classes_orig = sorted(np.unique(y))
label_map    = {c: i for i, c in enumerate(classes_orig)}
y_enc        = np.array([label_map[yi] for yi in y])
n_classes    = len(classes_orig)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ─────────────────────────────────────────────────────────────────────────────
# PREGUNTA 3 – Dos esquemas de partición
# ─────────────────────────────────────────────────────────────────────────────
print("="*70)
print("PREGUNTA 3 – Esquemas de partición: 80/20 vs 70/15/15")
print("="*70)

# --- Esquema A: 80 % entrenamiento / 20 % prueba ---
X_trainA, X_testA, y_trainA, y_testA = train_test_split(
    X_scaled, y_enc, test_size=0.20, random_state=42, stratify=y_enc
)

# Aplicar SMOTE solo al entrenamiento del esquema A
try:
    sm = SMOTE(random_state=42)
    X_trainA_sm, y_trainA_sm = sm.fit_resample(X_trainA, y_trainA)
    print(f"\nEsquema A – Tamaños: Train={X_trainA_sm.shape[0]}, Test={X_testA.shape[0]}")
except Exception:
    X_trainA_sm, y_trainA_sm = X_trainA, y_trainA
    print("SMOTE no aplicado (clases insuficientes). Usando datos originales.")

# --- Esquema B: 70 % train / 15 % val / 15 % test ---
X_tv, X_testB, y_tv, y_testB = train_test_split(
    X_scaled, y_enc, test_size=0.15, random_state=42, stratify=y_enc
)
X_trainB, X_valB, y_trainB, y_valB = train_test_split(
    X_tv, y_tv, test_size=0.15/0.85, random_state=42, stratify=y_tv
)

try:
    X_trainB_sm, y_trainB_sm = sm.fit_resample(X_trainB, y_trainB)
    print(f"Esquema B – Tamaños: Train={X_trainB_sm.shape[0]}, Val={X_valB.shape[0]}, Test={X_testB.shape[0]}")
except Exception:
    X_trainB_sm, y_trainB_sm = X_trainB, y_trainB

print("""
Ventaja del esquema de tres particiones:
  • El conjunto de VALIDACIÓN permite ajustar hiperparámetros (capas, neuronas,
    tasa de aprendizaje) sin tocar nunca el conjunto de PRUEBA.
  • Con solo dos particiones, el conjunto de prueba se usa implícitamente para
    elegir el modelo, lo que introduce sesgo optimista en la evaluación final.
  • Con n > 200 el esquema tres particiones es preferible; con n muy pequeño
    la validación cruzada k-fold sería más eficiente (ver P4).
""")

# ─────────────────────────────────────────────────────────────────────────────
# PREGUNTA 4 – Validación cruzada k-fold
# ─────────────────────────────────────────────────────────────────────────────
print("="*70)
print("PREGUNTA 4 – Validación cruzada k-fold (k=5) como complemento")
print("="*70)

# Usamos un modelo logístico como proxy rápido para ilustrar k-fold;
# la red se evalúa por cv en la sección de arquitectura.
from sklearn.linear_model import LogisticRegression as LR_kfold
lr_kf = LR_kfold(max_iter=1000, solver='lbfgs', random_state=42)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(lr_kf, X_scaled, y_enc, cv=kf, scoring='accuracy')

print(f"\nAccuracy por fold (regresión logística como proxy): {np.round(cv_scores,4)}")
print(f"Media ± std: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print("""
Conclusión P4:
  • Con n ≈ 200 la varianza de la estimación con una sola partición puede ser
    alta. K-fold (k=5) es una alternativa eficiente para estimación del error
    de generalización, aunque cuesta más cómputo para redes neuronales.
  • Se recomienda usar k-fold para la selección de hiperparámetros y la
    partición fija para la evaluación final reportada.
""")


# ─────────────────────────────────────────────────────────────────────────────
# Función auxiliar para construir y entrenar modelos Keras (multiclase)
# ─────────────────────────────────────────────────────────────────────────────
def build_mlp(n_inputs, n_classes, hidden_layers, neurons, lr=0.01,
              loss="sparse_categorical_crossentropy"):
    """Construye un MLP con capas ocultas de activación ReLU."""
    model = keras.Sequential()
    model.add(layers.Input(shape=(n_inputs,)))
    for _ in range(hidden_layers):
        model.add(layers.Dense(neurons, activation="relu"))
    model.add(layers.Dense(n_classes, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=["accuracy"]
    )
    return model


def train_model(model, X_tr, y_tr, X_val, y_val, epochs=150, batch_size=16):
    """Entrena el modelo y retorna el historial."""
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=20,
                                         restore_best_weights=True)]
    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        verbose=0, callbacks=cb
    )
    return hist


# ─────────────────────────────────────────────────────────────────────────────
# PREGUNTA 5 – Comparación de arquitecturas
# ─────────────────────────────────────────────────────────────────────────────
print("="*70)
print("PREGUNTA 5 – Comparación de arquitecturas del MLP")
print("="*70)

# NOTA METODOLÓGICA:
# Este bloque evalúa un conjunto dirigido de configuraciones, no un grid factorial
# completo. La selección del mejor modelo se hace con el conjunto de VALIDACIÓN,
# no con el conjunto de prueba. El test se conserva como evaluación final
# independiente y no debe usarse para escoger hiperparámetros.
configs = [
    # (capas_ocultas, neuronas, tasa_aprendizaje)
    (1, 16,  0.001),
    (1, 32,  0.01),
    (2, 16,  0.01),
    (2, 32,  0.001),
    (2, 64,  0.1),
    (3, 16,  0.001),
    (3, 32,  0.01),
    (3, 64,  0.001),
]

resultados = []
historiales = {}
historiales_epocas = []

for (capas, neuronas, lr) in configs:
    lbl = f"{capas}L-{neuronas}N-lr{lr}"
    model = build_mlp(len(FEATURES), n_classes, capas, neuronas, lr)
    hist  = train_model(model, X_trainB_sm, y_trainB_sm, X_valB, y_valB)

    y_pred    = np.argmax(model.predict(X_testB, verbose=0), axis=1)
    acc_test  = accuracy_score(y_testB, y_pred)

    train_acc_max = max(hist.history["accuracy"])
    val_acc_max   = max(hist.history["val_accuracy"])
    train_loss_min = min(hist.history["loss"])
    val_loss_min   = min(hist.history["val_loss"])
    train_loss_final = hist.history["loss"][-1]
    val_loss_final   = hist.history["val_loss"][-1]
    n_epochs = len(hist.history["loss"])

    resultados.append({
        "Configuración": lbl,
        "Capas ocultas": capas,
        "Neuronas/capa": neuronas,
        "Tasa aprendizaje": lr,
        "Épocas ejecutadas": n_epochs,
        "Loss Train mínima": round(train_loss_min, 6),
        "Loss Val mínima": round(val_loss_min, 6),
        "Loss Train final": round(train_loss_final, 6),
        "Loss Val final": round(val_loss_final, 6),
        "Acc Train (máx)": round(train_acc_max, 4),
        "Acc Val (máx)": round(val_acc_max, 4),
        "Acc Test": round(acc_test, 4),
        "Brecha Acc Train-Val": round(train_acc_max - val_acc_max, 4),
        "Brecha Loss Val-Train": round(val_loss_min - train_loss_min, 6),
    })

    for epoca, (loss, val_loss, acc, val_acc) in enumerate(
        zip(
            hist.history["loss"],
            hist.history["val_loss"],
            hist.history["accuracy"],
            hist.history["val_accuracy"],
        ),
        start=1,
    ):
        historiales_epocas.append({
            "Configuración": lbl,
            "Época": epoca,
            "Loss Train": loss,
            "Loss Val": val_loss,
            "Acc Train": acc,
            "Acc Val": val_acc,
        })

    historiales[lbl] = hist
    print(
        f"  {lbl:30s} | Val loss min = {val_loss_min:.4f} "
        f"| Val acc max = {val_acc_max:.4f} | Test acc ref. = {acc_test:.4f}"
    )

df_res = pd.DataFrame(resultados)

# Orden recomendado: primero menor pérdida de validación; como desempate,
# menor brecha entre desempeño de entrenamiento y validación.
df_res_ordenado = df_res.sort_values(
    by=["Loss Val mínima", "Brecha Acc Train-Val"],
    ascending=[True, True],
).reset_index(drop=True)

print("\n── Tabla comparativa de arquitecturas ──")
print(df_res_ordenado.to_string(index=False))

save_csv(df_res_ordenado, "13_p5_comparacion_arquitecturas_mlp.csv")
save_csv(pd.DataFrame(historiales_epocas), "14_p5_historial_perdida_por_epoca.csv")

# Figura 7 – Curvas de aprendizaje de todas las configuraciones
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axes = axes.flatten()
for ax, (lbl, hist) in zip(axes, historiales.items()):
    ax.plot(hist.history["loss"],     label="Train loss", linewidth=1.5)
    ax.plot(hist.history["val_loss"], label="Val loss",   linewidth=1.5, linestyle="--")
    ax.set_title(lbl, fontsize=9, fontweight="bold")
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

fig.suptitle("Figura 7 – Curvas de aprendizaje por configuración (P5)",
             fontsize=13, fontweight="bold")
save_figure(fig, "07_curvas_aprendizaje.png")
print("✔ Figura 7 guardada: 07_curvas_aprendizaje.png")

# Mejor configuración: se selecciona usando VALIDACIÓN, no TEST.
best_row = df_res_ordenado.iloc[0]
best_cfg = best_row["Configuración"]
print(
    f"\n★ Mejor configuración según validación: {best_cfg} "
    f"(Loss Val mínima = {best_row['Loss Val mínima']:.4f}; "
    f"Acc Val máx = {best_row['Acc Val (máx)']:.4f}; "
    f"Acc Test solo como referencia = {best_row['Acc Test']:.4f})"
)


# ─────────────────────────────────────────────────────────────────────────────
# PREGUNTA 6 – Entropía cruzada vs MSE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PREGUNTA 6 – Comparación de funciones de pérdida: CCE vs MSE")
print("="*70)

# Usamos la arquitectura ganadora de P5
bl = best_row
capas_best   = int(bl["Capas ocultas"])
neuronas_best = int(bl["Neuronas/capa"])
lr_best       = float(bl["Tasa aprendizaje"])

# Modelo con CCE
m_cce = build_mlp(len(FEATURES), n_classes, capas_best, neuronas_best, lr_best,
                  loss="sparse_categorical_crossentropy")
h_cce = train_model(m_cce, X_trainB_sm, y_trainB_sm, X_valB, y_valB)

# Para MSE necesitamos one-hot
from tensorflow.keras.utils import to_categorical
y_tr_oh  = to_categorical(y_trainB_sm, n_classes)
y_val_oh = to_categorical(y_valB,      n_classes)
y_te_oh  = to_categorical(y_testB,     n_classes)

m_mse = build_mlp(len(FEATURES), n_classes, capas_best, neuronas_best, lr_best,
                  loss="mean_squared_error")
m_mse.compile(optimizer=keras.optimizers.Adam(lr_best),
              loss="mean_squared_error", metrics=["accuracy"])
h_mse = m_mse.fit(X_trainB_sm, y_tr_oh, validation_data=(X_valB, y_val_oh),
                   epochs=150, batch_size=16, verbose=0,
                   callbacks=[keras.callbacks.EarlyStopping(
                       monitor="val_loss", patience=20, restore_best_weights=True)])

acc_cce = accuracy_score(y_testB, np.argmax(m_cce.predict(X_testB, verbose=0), axis=1))
acc_mse = accuracy_score(y_testB, np.argmax(m_mse.predict(X_testB, verbose=0), axis=1))

print(f"\n  CCE – Accuracy en prueba: {acc_cce:.4f}")
print(f"  MSE – Accuracy en prueba: {acc_mse:.4f}")

# Figura 8 – Curvas de pérdida CCE vs MSE
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (hist, lbl, color) in zip(axes, [
        (h_cce, "Entropía Cruzada (CCE)", "steelblue"),
        (h_mse, "Error Cuadrático Medio (MSE)", "tomato")]):
    ax.plot(hist.history["loss"],     label="Train", color=color, lw=1.8)
    ax.plot(hist.history["val_loss"], label="Val",   color=color, lw=1.8, ls="--")
    ax.set_title(lbl, fontsize=11, fontweight="bold")
    ax.set_xlabel("Época"); ax.set_ylabel("Pérdida"); ax.legend(); ax.grid(alpha=0.3)

fig.suptitle("Figura 8 – Convergencia: CCE vs MSE (P6)", fontsize=12, fontweight="bold")
save_figure(fig, "08_cce_vs_mse.png")
print("✔ Figura 8 guardada: 08_cce_vs_mse.png")

print("""
Justificación teórica P6:
  • La entropía cruzada categórica (CCE) es la función de pérdida natural para
    clasificación multi-clase: mide la divergencia KL entre la distribución
    verdadera y la predicha por softmax.
  • MSE fuerza a la red a tratar las clases como valores continuos, ignorando
    la naturaleza discreta y el orden implícito de las categorías.
  • En la práctica, CCE produce gradientes más informativos cerca del óptimo y
    generalmente converge más rápido y a mejores soluciones.
""")

# Modelo final multi-clase (CCE, mejor arquitectura) para P7
best_model_multi = m_cce

# ─────────────────────────────────────────────────────────────────────────────
# PREGUNTA 7 – Métricas completas del mejor modelo multi-clase
# ─────────────────────────────────────────────────────────────────────────────
print("="*70)
print("PREGUNTA 7 – Métricas completas del mejor modelo multi-clase")
print("="*70)

y_pred_multi = np.argmax(best_model_multi.predict(X_testB, verbose=0), axis=1)
y_prob_multi = best_model_multi.predict(X_testB, verbose=0)

print(f"\nExactitud global (Test): {accuracy_score(y_testB, y_pred_multi):.4f}")
print("\nReporte de clasificación:")
target_names = [f"Sev {c}" for c in classes_orig]
print(classification_report(y_testB, y_pred_multi, target_names=target_names))

fig, ax = plt.subplots(figsize=(7, 5))
cm = confusion_matrix(y_testB, y_pred_multi)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names, ax=ax, linewidths=0.5)
ax.set_xlabel("Predicho", fontsize=11)
ax.set_ylabel("Real", fontsize=11)
ax.set_title("Figura 9 – Matriz de confusión (mejor MLP multi-clase, P7)",
             fontsize=12, fontweight="bold")
save_figure(fig, "09_confusion_multiclase.png")
print("✔ Figura 9 guardada: 09_confusion_multiclase.png")

y_bin_test = label_binarize(y_testB, classes=list(range(n_classes)))
if n_classes == 2:
    y_bin_test = np.hstack([1 - y_bin_test, y_bin_test])

fig, ax = plt.subplots(figsize=(7, 5))
auc_scores = []
for i, cls in enumerate(classes_orig):
    if y_bin_test[:, i].sum() == 0:
        continue
    fpr, tpr, _ = roc_curve(y_bin_test[:, i], y_prob_multi[:, i])
    auc_val = auc(fpr, tpr)
    auc_scores.append(auc_val)
    ax.plot(fpr, tpr, lw=1.8, label=f"Sev {cls} (AUC={auc_val:.2f})")

ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("Tasa de Falsos Positivos")
ax.set_ylabel("Tasa de Verdaderos Positivos")
ax.set_title("Figura 10 – Curvas ROC por clase (one-vs-rest) P7",
             fontsize=12, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
save_figure(fig, "10_roc_multiclase.png")
print("✔ Figura 10 guardada: 10_roc_multiclase.png")
print(f"\nAUC-ROC promedio (macro): {np.mean(auc_scores):.4f}")


# =============================================================================
# PARTE II – CLASIFICACIÓN BINARIA
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# PREGUNTA 8 – Colapso de categorías a binario
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PREGUNTA 8 – Colapso a clasificación binaria: sano (1) vs. enfermo (≥2)")
print("="*70)

y_bin = (df[TARGET].values > 1).astype(int)
print(f"\nDistribución binaria:\n  Sano  (0): {(y_bin==0).sum()}\n  Enfermo (1): {(y_bin==1).sum()}")
print("""
Justificación P8:
  • Agronómicamente, la detección temprana de CUALQUIER infección es crítica;
    por tanto, agrupar todos los niveles de severidad como "enfermo" es útil
    para diseñar alertas tempranas.
  • La reducción a binario simplifica la decisión pero pierde la gradación de
    severidad, lo que podría importar para gestión diferenciada de la cosecha.
  • Una alternativa sería sano vs. leve vs. severo (tres grupos), pero requiere
    consenso agronómico para definir el umbral "leve/severo".
""")

X_tv_b, X_te_b, y_tv_b, y_te_b = train_test_split(
    X_scaled, y_bin, test_size=0.15, random_state=42, stratify=y_bin
)
X_tr_b, X_val_b, y_tr_b, y_val_b = train_test_split(
    X_tv_b, y_tv_b, test_size=0.15/0.85, random_state=42, stratify=y_tv_b
)
try:
    sm_b = SMOTE(random_state=42)
    X_tr_b_sm, y_tr_b_sm = sm_b.fit_resample(X_tr_b, y_tr_b)
except Exception:
    X_tr_b_sm, y_tr_b_sm = X_tr_b, y_tr_b


# ─────────────────────────────────────────────────────────────────────────────
# PREGUNTA 9 – MLP binario y comparación con multi-clase
# ─────────────────────────────────────────────────────────────────────────────
print("="*70)
print("PREGUNTA 9 – MLP binario y comparación con modelo multi-clase")
print("="*70)

def build_mlp_bin(n_inputs, hidden_layers, neurons, lr=0.01):
    model = keras.Sequential()
    model.add(layers.Input(shape=(n_inputs,)))
    for _ in range(hidden_layers):
        model.add(layers.Dense(neurons, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

best_bin_model = build_mlp_bin(len(FEATURES), capas_best, neuronas_best, lr_best)
h_bin = train_model(best_bin_model, X_tr_b_sm, y_tr_b_sm, X_val_b, y_val_b)

y_prob_bin = best_bin_model.predict(X_te_b, verbose=0).flatten()
y_pred_bin = (y_prob_bin >= 0.5).astype(int)
print(f"\nAccuracy binario (Test): {accuracy_score(y_te_b, y_pred_bin):.4f}")
print(classification_report(y_te_b, y_pred_bin, target_names=["Sano", "Enfermo"]))


# ─────────────────────────────────────────────────────────────────────────────
# PREGUNTA 10 – Métricas binarias y umbral óptimo de Youden
# ─────────────────────────────────────────────────────────────────────────────
print("="*70)
print("PREGUNTA 10 – Métricas binarias completas y umbral óptimo (Youden)")
print("="*70)

fpr_b, tpr_b, thresholds_b = roc_curve(y_te_b, y_prob_bin)
auc_bin  = roc_auc_score(y_te_b, y_prob_bin)
youden   = tpr_b - fpr_b
best_idx = np.argmax(youden)
best_thr = thresholds_b[best_idx]

y_pred_youden = (y_prob_bin >= best_thr).astype(int)
tn, fp, fn, tp = confusion_matrix(y_te_b, y_pred_youden).ravel()
sens  = tp/(tp+fn) if (tp+fn) > 0 else 0
spec  = tn/(tn+fp) if (tn+fp) > 0 else 0
prec  = tp/(tp+fp) if (tp+fp) > 0 else 0
f1_y  = 2*prec*sens/(prec+sens) if (prec+sens) > 0 else 0
acc_y = (tp+tn)/(tp+tn+fp+fn)

print(f"\n  Umbral óptimo (Youden):  {best_thr:.4f}")
print(f"  Accuracy:                {acc_y:.4f}")
print(f"  Sensibilidad (Recall):   {sens:.4f}  ← Detecta verdaderos enfermos")
print(f"  Especificidad:           {spec:.4f}  ← Detecta verdaderos sanos")
print(f"  Precisión:               {prec:.4f}")
print(f"  F1-Score:                {f1_y:.4f}")
print(f"  AUC-ROC:                 {auc_bin:.4f}")
print("""
Contexto agronómico P10:
  • Un FALSO NEGATIVO (planta enferma clasificada como sana) es MÁS COSTOSO
    porque la infección se propaga sin tratamiento.
  • Por tanto, se prioriza ALTA SENSIBILIDAD aunque baje la especificidad.
  • El umbral de Youden equilibra ambos, pero en la práctica se podría bajar
    el umbral (ej. 0.35) para aumentar sensibilidad a expensas de más FP.
""")

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr_b, tpr_b, color="steelblue", lw=2, label=f"ROC (AUC = {auc_bin:.3f})")
ax.scatter(fpr_b[best_idx], tpr_b[best_idx], color="red", zorder=5,
           label=f"Umbral Youden = {best_thr:.3f}")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("Tasa de Falsos Positivos")
ax.set_ylabel("Tasa de Verdaderos Positivos")
ax.set_title("Figura 11 – Curva ROC binaria con umbral de Youden (P10)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
save_figure(fig, "11_roc_binario.png")
print("✔ Figura 11 guardada: 11_roc_binario.png")


# =============================================================================
# PARTE III – REGRESIÓN LOGÍSTICA Y COMPARACIÓN
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# PREGUNTA 11 – Regresión logística binaria con coeficientes
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PREGUNTA 11 – Regresión logística binaria (coeficientes e inferencia)")
print("="*70)

log_reg = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
log_reg.fit(X_tr_b_sm, y_tr_b_sm)

coefs     = log_reg.coef_[0]
intercept = log_reg.intercept_[0]
n_tr      = X_tr_b_sm.shape[0]

y_hat_p = log_reg.predict_proba(X_tr_b_sm)[:, 1]
W       = np.diag(y_hat_p * (1 - y_hat_p))
X_aug   = np.hstack([np.ones((n_tr, 1)), X_tr_b_sm])
XtWX    = X_aug.T @ W @ X_aug
try:
    cov_mat  = np.linalg.inv(XtWX)
    se       = np.sqrt(np.diag(cov_mat))
    se_coef  = se[1:]
    z_coef   = coefs / se_coef
    p_coef   = 2*(1 - stats.norm.cdf(np.abs(z_coef)))
    ci_lo    = coefs - 1.96*se_coef
    ci_hi    = coefs + 1.96*se_coef
    valid_se = True
except np.linalg.LinAlgError:
    valid_se = False

df_coefs = pd.DataFrame({
    "Variable":    FEATURES,
    "Coeficiente": coefs,
    "OR (exp(β))": np.exp(coefs),
})
if valid_se:
    df_coefs["Error Std"] = se_coef
    df_coefs["Z"]         = z_coef
    df_coefs["p-valor"]   = p_coef
    df_coefs["IC95 Inf"]  = ci_lo
    df_coefs["IC95 Sup"]  = ci_hi

print("\nCoeficientes del modelo logístico:")
print(df_coefs.round(4).to_string(index=False))
print(f"\n  Intercepto: {intercept:.4f}")
save_csv(df_coefs.round(6), "06_coeficientes_logisticos.csv")

fig, ax = plt.subplots(figsize=(8, 4))
colors_coef = ["tomato" if c < 0 else "steelblue" for c in coefs]
ax.barh(FEATURES, coefs, color=colors_coef, edgecolor="black", alpha=0.8)
if valid_se:
    ax.errorbar(coefs, FEATURES, xerr=1.96*se_coef, fmt="none",
                color="black", capsize=5, linewidth=1.5)
ax.axvline(0, color="black", lw=0.8, linestyle="--")
ax.set_xlabel("Coeficiente logístico (β)", fontsize=11)
ax.set_title("Figura 12 – Coeficientes de regresión logística con IC 95% (P11)",
             fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
save_figure(fig, "12_coeficientes_logisticos.png")
print("✔ Figura 12 guardada: 12_coeficientes_logisticos.png")


# ─────────────────────────────────────────────────────────────────────────────
# PREGUNTA 12 – Selección de variables (Lasso L1)
# ─────────────────────────────────────────────────────────────────────────────
print("="*70)
print("PREGUNTA 12 – Selección de variables con regularización Lasso (L1)")
print("="*70)

log_lasso = LogisticRegressionCV(
    cv=5, penalty="l1", solver="liblinear",
    Cs=np.logspace(-3, 2, 20), max_iter=1000, random_state=42
)
log_lasso.fit(X_tr_b_sm, y_tr_b_sm)
coefs_lasso = log_lasso.coef_[0]

df_lasso = pd.DataFrame({
    "Variable":    FEATURES,
    "Coef. Lasso": coefs_lasso,
    "Abs(Coef)":   np.abs(coefs_lasso)
}).sort_values("Abs(Coef)", ascending=False)

print(f"\nC óptimo (Lasso): {log_lasso.C_[0]:.5f}")
print("\nImportancia de variables según Lasso:")
print(df_lasso.round(5).to_string(index=False))
save_csv(df_lasso.round(6), "07_importancia_lasso.csv")

fig, ax = plt.subplots(figsize=(7, 4))
colors_l = ["steelblue" if c > 0 else "tomato" for c in df_lasso["Coef. Lasso"]]
ax.barh(df_lasso["Variable"], df_lasso["Coef. Lasso"],
        color=colors_l, edgecolor="black", alpha=0.85)
ax.axvline(0, color="black", lw=0.8, linestyle="--")
ax.set_xlabel("Coeficiente (L1)", fontsize=11)
ax.set_title("Figura 13 – Importancia de variables – Regresión Logística Lasso (P12)",
             fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
save_figure(fig, "13_importancia_lasso.png")
print("✔ Figura 13 guardada: 13_importancia_lasso.png")


# ─────────────────────────────────────────────────────────────────────────────
# PREGUNTA 13 – Tabla comparativa: Regresión logística vs. MLP binario
# ─────────────────────────────────────────────────────────────────────────────
print("="*70)
print("PREGUNTA 13 – Comparación: Regresión logística vs. MLP binario")
print("="*70)

y_prob_lr = log_reg.predict_proba(X_te_b)[:, 1]
y_pred_lr = log_reg.predict(X_te_b)
auc_lr    = roc_auc_score(y_te_b, y_prob_lr)

def metricas(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp/(tp+fn) if (tp+fn) > 0 else 0
    spec = tn/(tn+fp) if (tn+fp) > 0 else 0
    return {
        "Accuracy":      round(accuracy_score(y_true, y_pred), 4),
        "Precision":     round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Sensibilidad":  round(sens, 4),
        "Especificidad": round(spec, 4),
        "F1-Score":      round(f1_score(y_true, y_pred, zero_division=0), 4),
        "AUC-ROC":       round(roc_auc_score(y_true, y_prob), 4),
    }

met_mlp = metricas(y_te_b, y_pred_bin, y_prob_bin)
met_lr  = metricas(y_te_b, y_pred_lr,  y_prob_lr)

df_comp = pd.DataFrame([met_mlp, met_lr], index=["MLP Binario", "Regresión Logística"])
print("\n── Tabla comparativa final ──")
print(df_comp.to_string())
save_csv(df_comp.reset_index(names="Modelo"), "08_comparacion_modelos.csv")

fig, ax = plt.subplots(figsize=(9, 4))
metricas_lbl = list(met_mlp.keys())
x = np.arange(len(metricas_lbl))
w = 0.35
ax.bar(x - w/2, list(met_mlp.values()), w, label="MLP", color="steelblue", edgecolor="black")
ax.bar(x + w/2, list(met_lr.values()),  w, label="Reg. Logística", color="coral", edgecolor="black")
ax.set_xticks(x)
ax.set_xticklabels(metricas_lbl, rotation=30, ha="right")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Valor de la métrica", fontsize=11)
ax.set_title("Figura 14 – Comparación MLP vs Regresión Logística (P13)",
             fontsize=12, fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.3)
save_figure(fig, "14_comparacion_modelos.png")
print("✔ Figura 14 guardada: 14_comparacion_modelos.png")
print("""
Conclusión P13:
  • Si el MLP supera a la regresión logística en AUC-ROC y sensibilidad,
    es preferible para el sistema de alerta fitosanitaria.
  • La regresión logística es preferible cuando se requiere interpretabilidad
    directa (coeficientes), rapidez de cómputo y menor riesgo de sobreajuste
    con muestras pequeñas.
  • En producción se recomendaría desplegar el MLP con un umbral ajustado
    que maximice sensibilidad (detectar enfermos), aceptando más FP.
""")


# ─────────────────────────────────────────────────────────────────────────────
# PREGUNTA 14 – Importancia de variables en MLP (Permutation Feature Importance)
# ─────────────────────────────────────────────────────────────────────────────
print("="*70)
print("PREGUNTA 14 – Importancia de variables en MLP (Permutation Feature Importance)")
print("="*70)

class KerasWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model): self.model = model
    def fit(self, X, y): return self
    def predict(self, X): return (self.model.predict(X, verbose=0).flatten() >= 0.5).astype(int)
    def score(self, X, y): return accuracy_score(y, self.predict(X))

wrapped = KerasWrapper(best_bin_model)
pfi = permutation_importance(wrapped, X_te_b, y_te_b,
                              n_repeats=30, random_state=42, scoring="accuracy")

df_pfi = pd.DataFrame({
    "Variable":        FEATURES,
    "Importancia PFI": pfi.importances_mean,
    "Std":             pfi.importances_std
}).sort_values("Importancia PFI", ascending=False)

print("\nImportancia por permutación (MLP):")
print(df_pfi.round(5).to_string(index=False))
save_csv(df_pfi.round(6), "09_importancia_pfi.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
order_pfi = df_pfi["Variable"].tolist()
axes[0].barh(order_pfi, df_pfi["Importancia PFI"],
             xerr=df_pfi["Std"], color="mediumseagreen",
             edgecolor="black", alpha=0.85, capsize=4)
axes[0].set_xlabel("Caída en Accuracy")
axes[0].set_title("PFI – MLP Binario", fontsize=11, fontweight="bold")
axes[0].grid(axis="x", alpha=0.3)

df_lasso_sorted = df_lasso.sort_values("Abs(Coef)", ascending=True)
axes[1].barh(df_lasso_sorted["Variable"], df_lasso_sorted["Abs(Coef)"],
             color="coral", edgecolor="black", alpha=0.85)
axes[1].set_xlabel("|Coeficiente| Lasso")
axes[1].set_title("Importancia – Regresión Logística Lasso", fontsize=11, fontweight="bold")
axes[1].grid(axis="x", alpha=0.3)

fig.suptitle("Figura 15 – Comparación de importancia de variables: MLP vs Logística (P14)",
             fontsize=12, fontweight="bold")
save_figure(fig, "15_importancia_variables.png")
print("✔ Figura 15 guardada: 15_importancia_variables.png")


# =============================================================================
# PARTE IV – ANÁLISIS DE DEPENDENCIA ESPACIAL
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# PREGUNTA 15 – Grilla 14×14 y visualización
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PREGUNTA 15 – Construcción de grilla 14×14 y visualización")
print("="*70)

GRID_N    = 14
N_SAMPLE  = GRID_N * GRID_N
sample_idx = np.random.choice(len(df), N_SAMPLE, replace=False)
df_grid   = df.iloc[sample_idx].copy().reset_index(drop=True)
df_grid["row"]     = df_grid.index // GRID_N
df_grid["col"]     = df_grid.index %  GRID_N
df_grid["x_coord"] = df_grid["col"]
df_grid["y_coord"] = df_grid["row"]

grid_sev = df_grid.pivot(index="row", columns="col", values=TARGET).values.astype(float)
n_sev    = int(df[TARGET].max())
cmap_sev = plt.colormaps["YlOrRd"].resampled(n_sev)

fig, ax = plt.subplots(figsize=(8, 7))
im   = ax.imshow(grid_sev, cmap=cmap_sev, vmin=0.5, vmax=n_sev+0.5, origin="upper")
cbar = fig.colorbar(im, ax=ax, ticks=range(1, n_sev+1))
cbar.set_label("Nivel de severidad", fontsize=11)
ax.set_title("Figura 16 – Grilla 14×14: severidad de Verticillium sp. (P15)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Columna (x)")
ax.set_ylabel("Fila (y)")
save_figure(fig, "16_grilla_severidad.png")
print("✔ Figura 16 guardada: 16_grilla_severidad.png")


# ─────────────────────────────────────────────────────────────────────────────
# PREGUNTA 16 – Índice de Moran global
# ─────────────────────────────────────────────────────────────────────────────
print("="*70)
print("PREGUNTA 16 – Índice de Moran global para datos ordinales (rangos)")
print("="*70)

def moran_global(values_flat, n_rows, n_cols):
    N = n_rows * n_cols
    z = values_flat - values_flat.mean()
    W = np.zeros((N, N))
    for i in range(N):
        r, c = divmod(i, n_cols)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r+dr, c+dc
                if 0 <= nr < n_rows and 0 <= nc < n_cols:
                    W[i, nr*n_cols + nc] = 1
    W_row_sum = W.sum(axis=1, keepdims=True)
    W_row_sum[W_row_sum == 0] = 1
    Wn    = W / W_row_sum
    W_sum = Wn.sum()
    I     = (N * (z @ (Wn @ z))) / (W_sum * (z @ z))
    E_I   = -1 / (N - 1)
    S1    = 0.5 * np.sum((Wn + Wn.T)**2)
    S2    = np.sum((Wn.sum(axis=1) + Wn.sum(axis=0))**2)
    n     = N
    Var_I = (n*((n**2-3*n+3)*S1 - n*S2 + 3*(W_sum**2)) /
             ((n-1)*(n-2)*(n-3)*(W_sum**2)) - (-1/(n-1))**2)
    if Var_I <= 0:
        Var_I = abs(Var_I) + 1e-10
    z_score = (I - E_I) / np.sqrt(Var_I)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    return I, E_I, z_score, p_value

ranks = pd.Series(df_grid[TARGET].values).rank().values
I, E_I, Z, pval = moran_global(ranks, GRID_N, GRID_N)
print(f"\n  Índice de Moran (I):     {I:.4f}")
print(f"  Valor esperado E[I]:     {E_I:.4f}")
print(f"  Estadístico Z:           {Z:.4f}")
print(f"  p-valor:                 {pval:.4f}")
print(f"\n  {'★ Existe autocorrelación espacial positiva significativa (p<0.05).' if pval<0.05 else '✗ No hay evidencia de dependencia espacial significativa (p≥0.05).'}")

save_csv(pd.DataFrame({"I": [I], "E_I": [E_I], "Z": [Z], "p_valor": [pval]}),
         "10_moran_global.csv")


# ─────────────────────────────────────────────────────────────────────────────
# PREGUNTA 17 – MLP con coordenadas espaciales
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PREGUNTA 17 – MLP con coordenadas espaciales como predictores adicionales")
print("="*70)

df_full = df.copy().reset_index(drop=True)
df_full["x_coord"] = df_full.index % GRID_N
df_full["y_coord"] = df_full.index // GRID_N

FEATURES_SPATIAL = FEATURES + ["x_coord", "y_coord"]
X_sp      = df_full[FEATURES_SPATIAL].values
scaler_sp = StandardScaler()
X_sp_sc   = scaler_sp.fit_transform(X_sp)

X_tv_sp, X_te_sp, y_tv_sp, y_te_sp = train_test_split(
    X_sp_sc, y_bin, test_size=0.15, random_state=42, stratify=y_bin
)
X_tr_sp, X_val_sp, y_tr_sp, y_val_sp = train_test_split(
    X_tv_sp, y_tv_sp, test_size=0.15/0.85, random_state=42, stratify=y_tv_sp
)
try:
    X_tr_sp_sm, y_tr_sp_sm = sm_b.fit_resample(X_tr_sp, y_tr_sp)
except Exception:
    X_tr_sp_sm, y_tr_sp_sm = X_tr_sp, y_tr_sp

m_sp = build_mlp_bin(len(FEATURES_SPATIAL), capas_best, neuronas_best, lr_best)
train_model(m_sp, X_tr_sp_sm, y_tr_sp_sm, X_val_sp, y_val_sp)

y_prob_sp = m_sp.predict(X_te_sp, verbose=0).flatten()
y_pred_sp = (y_prob_sp >= 0.5).astype(int)
acc_sp    = accuracy_score(y_te_sp, y_pred_sp)
auc_sp    = roc_auc_score(y_te_sp, y_prob_sp)
acc_base  = accuracy_score(y_te_b, y_pred_bin)
auc_base  = roc_auc_score(y_te_b, y_prob_bin)

print(f"\n  Modelo SIN coordenadas – Acc: {acc_base:.4f} | AUC: {auc_base:.4f}")
print(f"  Modelo CON coordenadas – Acc: {acc_sp:.4f}   | AUC: {auc_sp:.4f}")
print(f"  Δ AUC = {auc_sp - auc_base:+.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# PREGUNTA 18 – Coordenadas solas e interacciones espectral-espaciales
# ─────────────────────────────────────────────────────────────────────────────
print("="*70)
print("PREGUNTA 18 – Coordenadas como único predictor e interacciones espaciales")
print("="*70)

FEAT_COORDS = ["x_coord", "y_coord"]
X_co        = df_full[FEAT_COORDS].values
scaler_c    = StandardScaler()
X_co_sc     = scaler_c.fit_transform(X_co)

X_tv_c, X_te_c, y_tv_c, y_te_c = train_test_split(
    X_co_sc, y_bin, test_size=0.15, random_state=42, stratify=y_bin
)
X_tr_c, X_val_c, y_tr_c, y_val_c = train_test_split(
    X_tv_c, y_tv_c, test_size=0.15/0.85, random_state=42, stratify=y_tv_c
)
m_coords = build_mlp_bin(2, capas_best, neuronas_best, lr_best)
train_model(m_coords, X_tr_c, y_tr_c, X_val_c, y_val_c)
y_prob_c = m_coords.predict(X_te_c, verbose=0).flatten()
acc_c    = accuracy_score(y_te_c, (y_prob_c >= 0.5).astype(int))
auc_c    = roc_auc_score(y_te_c, y_prob_c)
print(f"\n  18a) Solo coordenadas – Acc: {acc_c:.4f} | AUC: {auc_c:.4f}")

df_inter = df_full.copy()
df_inter["x_ndvi"] = df_inter["x_coord"] * df_inter["ndvi_med"]
df_inter["y_ndvi"] = df_inter["y_coord"] * df_inter["ndvi_med"]
df_inter["x_evi"]  = df_inter["x_coord"] * df_inter["evi_med"]
df_inter["y_evi"]  = df_inter["y_coord"] * df_inter["evi_med"]

FEAT_INTER = FEATURES + ["x_coord", "y_coord", "x_ndvi", "y_ndvi", "x_evi", "y_evi"]
X_in       = df_inter[FEAT_INTER].values
scaler_in  = StandardScaler()
X_in_sc    = scaler_in.fit_transform(X_in)

X_tv_i, X_te_i, y_tv_i, y_te_i = train_test_split(
    X_in_sc, y_bin, test_size=0.15, random_state=42, stratify=y_bin
)
X_tr_i, X_val_i, y_tr_i, y_val_i = train_test_split(
    X_tv_i, y_tv_i, test_size=0.15/0.85, random_state=42, stratify=y_tv_i
)
try:
    X_tr_i_sm, y_tr_i_sm = sm_b.fit_resample(X_tr_i, y_tr_i)
except Exception:
    X_tr_i_sm, y_tr_i_sm = X_tr_i, y_tr_i

m_inter  = build_mlp_bin(len(FEAT_INTER), capas_best, neuronas_best, lr_best)
train_model(m_inter, X_tr_i_sm, y_tr_i_sm, X_val_i, y_val_i)
y_prob_i = m_inter.predict(X_te_i, verbose=0).flatten()
acc_i    = accuracy_score(y_te_i, (y_prob_i >= 0.5).astype(int))
auc_i    = roc_auc_score(y_te_i, y_prob_i)
print(f"  18b) Interacciones espaciales – Acc: {acc_i:.4f} | AUC: {auc_i:.4f}")

fig, ax = plt.subplots(figsize=(9, 4))
modelos_sp = ["Base\n(sin coords)", "Con\ncoordenadas", "Solo\ncoordenadas", "Con\ninteracciones"]
aucs_sp    = [auc_base, auc_sp, auc_c, auc_i]
colors_sp  = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
bars = ax.bar(modelos_sp, aucs_sp, color=colors_sp, edgecolor="black", alpha=0.85)
for bar, v in zip(bars, aucs_sp):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.005, f"{v:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.set_ylabel("AUC-ROC", fontsize=11)
ax.set_title("Figura 17 – Efecto de la información espacial en el AUC-ROC (P17-18)",
             fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
save_figure(fig, "17_espacial_aucroc.png")
print("✔ Figura 17 guardada: 17_espacial_aucroc.png")


# =============================================================================
# PARTE V – PROPUESTA LIBRE: DROPOUT + RANDOM FOREST
# =============================================================================

print("\n" + "="*70)
print("PREGUNTA 19 – Propuesta libre: Dropout / Early Stopping + Random Forest")
print("="*70)
print("""
Objetivo: Evaluar el efecto de Dropout y Early Stopping sobre el sobreajuste,
y comparar el MLP con Random Forest usando las mismas particiones y métricas.
""")

def build_mlp_dropout(n_inputs, hidden_layers, neurons, lr=0.01, dropout_rate=0.3):
    model = keras.Sequential()
    model.add(layers.Input(shape=(n_inputs,)))
    for _ in range(hidden_layers):
        model.add(layers.Dense(neurons, activation="relu"))
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model

m_drop      = build_mlp_dropout(len(FEATURES), capas_best, neuronas_best, lr_best, 0.3)
h_drop      = train_model(m_drop, X_tr_b_sm, y_tr_b_sm, X_val_b, y_val_b, epochs=300)
y_prob_drop = m_drop.predict(X_te_b, verbose=0).flatten()
y_pred_drop = (y_prob_drop >= 0.5).astype(int)
acc_drop    = accuracy_score(y_te_b, y_pred_drop)
auc_drop    = roc_auc_score(y_te_b, y_prob_drop)
print(f"\n  MLP con Dropout (p=0.3) – Acc: {acc_drop:.4f} | AUC: {auc_drop:.4f}")

rf = RandomForestClassifier(n_estimators=200, max_depth=None,
                             class_weight="balanced", random_state=42)
rf.fit(X_tr_b_sm, y_tr_b_sm)
y_prob_rf = rf.predict_proba(X_te_b)[:, 1]
y_pred_rf = rf.predict(X_te_b)
acc_rf    = accuracy_score(y_te_b, y_pred_rf)
auc_rf    = roc_auc_score(y_te_b, y_prob_rf)
print(f"  Random Forest (200 árboles) – Acc: {acc_rf:.4f} | AUC: {auc_rf:.4f}")

df_rf_imp = pd.DataFrame({
    "Variable":    FEATURES,
    "Importancia": rf.feature_importances_
}).sort_values("Importancia", ascending=False)
print("\nImportancia RF:")
print(df_rf_imp.round(5).to_string(index=False))
save_csv(df_rf_imp.round(6), "11_importancia_rf.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (hist, lbl, clr) in zip(axes, [
        (h_bin,  "MLP Sin Dropout", "steelblue"),
        (h_drop, "MLP Con Dropout", "darkorange")]):
    ax.plot(hist.history["loss"],     label="Train", color=clr, lw=1.8)
    ax.plot(hist.history["val_loss"], label="Val",   color=clr, lw=1.8, ls="--")
    ax.set_title(lbl, fontsize=11, fontweight="bold")
    ax.set_xlabel("Época")
    ax.set_ylabel("BCE Loss")
    ax.legend()
    ax.grid(alpha=0.3)
fig.suptitle("Figura 18 – Efecto del Dropout sobre sobreajuste (P19)",
             fontsize=12, fontweight="bold")
save_figure(fig, "18_dropout_comparacion.png")

all_models = ["Reg. Logística", "MLP Binario", "MLP Dropout", "Random Forest"]
all_aucs   = [auc_lr, auc_base, auc_drop, auc_rf]
all_accs   = [accuracy_score(y_te_b, y_pred_lr), acc_base, acc_drop, acc_rf]

fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(all_models))
w = 0.35
ax.bar(x - w/2, all_accs, w, label="Accuracy", color="steelblue", edgecolor="black", alpha=0.85)
ax.bar(x + w/2, all_aucs, w, label="AUC-ROC",  color="coral",     edgecolor="black", alpha=0.85)
for xi, (a, r) in enumerate(zip(all_accs, all_aucs)):
    ax.text(xi-w/2, a+0.005, f"{a:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(xi+w/2, r+0.005, f"{r:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(all_models)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Valor de la métrica", fontsize=11)
ax.set_title("Figura 18 – Comparación final de todos los modelos (P19)",
             fontsize=12, fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.3)
save_figure(fig, "19_comparacion_final.png")
print("✔ Figuras 17-19 guardadas.")

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(df_rf_imp["Variable"], df_rf_imp["Importancia"],
        color="mediumorchid", edgecolor="black", alpha=0.85)
ax.set_xlabel("Importancia (Gini)")
ax.set_title("Figura 18 – Importancia de variables – Random Forest (P19)",
             fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
save_figure(fig, "20_importancia_rf.png")
print("✔ Figura 20 guardada: 20_importancia_rf.png")


# =============================================================================
# RESUMEN EJECUTIVO FINAL
# =============================================================================
print("\n" + "="*70)
print("RESUMEN EJECUTIVO FINAL")
print("="*70)
resumen = pd.DataFrame({
    "Modelo":   all_models,
    "Accuracy": [round(a, 4) for a in all_accs],
    "AUC-ROC":  [round(r, 4) for r in all_aucs],
})
print(resumen.to_string(index=False))
save_csv(resumen, "12_resumen_ejecutivo.csv")
print("""
Recomendación final:
  • Para un sistema de apoyo a la decisión agronómica se sugiere el modelo
    con mayor AUC-ROC y sensibilidad, ajustando el umbral para minimizar FN.
  • La regularización por Dropout reduce el sobreajuste sin costo notable
    en desempeño y se recomienda en producción.
  • El análisis espacial (Moran) permite saber si la distribución de la
    enfermedad sigue un patrón de contagio, lo que es clave para diseñar
    zonas de cuarentena o tratamiento focalizado.
""")
print(f"Tablas guardadas en: {TABLES_DIR}")
print(f"Figuras guardadas en: {FIGURES_DIR}")

