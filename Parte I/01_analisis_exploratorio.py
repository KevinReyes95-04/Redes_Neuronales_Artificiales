"""Pregunta 1: analisis exploratorio de los datos originales."""

# %%
import os
from pathlib import Path
from tempfile import gettempdir

os.environ.setdefault("MPLCONFIGDIR", str(Path(gettempdir()) / "rna_taller_matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


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
