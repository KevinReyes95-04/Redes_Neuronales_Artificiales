"""Parte I: clasificacion multiple de severidad con perceptron multicapa."""

from _bootstrap import load_config, save_intermediate

load_config(__file__, globals())

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
# Funciones auxiliares para construir y entrenar modelos Keras (multiclase)
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
    """Entrena el modelo con early stopping sobre perdida de validación."""
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

balance_smote_train = pd.DataFrame({
    "severidad": classes_orig,
    "entrenamiento_antes_smote": [
        int(np.sum(y_trainB == label_map[severity])) for severity in classes_orig
    ],
    "entrenamiento_despues_smote": [
        int(np.sum(y_trainB_sm == label_map[severity])) for severity in classes_orig
    ],
})
save_csv(balance_smote_train, "06_p2_balance_smote_entrenamiento.csv")
print("\nBalance del entrenamiento antes y despues de SMOTE:")
print(balance_smote_train.to_string(index=False))

# Entrenamiento comparativo con una misma arquitectura inicial
p3_hidden_layers = 1
p3_neurons = 16
p3_lr = 0.001
p3_epochs = 150
p3_batch_size = 16

keras.backend.clear_session()
tf.random.set_seed(42)
mlp_p3_a = build_mlp(len(FEATURES), n_classes, p3_hidden_layers, p3_neurons, p3_lr)
h_p3_a = mlp_p3_a.fit(
    X_trainA_sm, y_trainA_sm,
    epochs=p3_epochs,
    batch_size=p3_batch_size,
    verbose=0,
)
loss_train_a, acc_train_a = mlp_p3_a.evaluate(X_trainA_sm, y_trainA_sm, verbose=0)
loss_test_a, acc_test_a = mlp_p3_a.evaluate(X_testA, y_testA, verbose=0)

keras.backend.clear_session()
tf.random.set_seed(42)
mlp_p3_b = build_mlp(len(FEATURES), n_classes, p3_hidden_layers, p3_neurons, p3_lr)
h_p3_b = train_model(
    mlp_p3_b,
    X_trainB_sm, y_trainB_sm,
    X_valB, y_valB,
    epochs=p3_epochs,
    batch_size=p3_batch_size,
)
loss_train_b, acc_train_b = mlp_p3_b.evaluate(X_trainB_sm, y_trainB_sm, verbose=0)
loss_val_b, acc_val_b = mlp_p3_b.evaluate(X_valB, y_valB, verbose=0)
loss_test_b, acc_test_b = mlp_p3_b.evaluate(X_testB, y_testB, verbose=0)

p3_comparison = pd.DataFrame([
    {
        "Esquema": "80/20",
        "Arquitectura inicial": "1 capa oculta, 16 neuronas, lr=0.001",
        "Entrenamiento": int(len(y_trainA_sm)),
        "Validacion": 0,
        "Prueba": int(len(y_testA)),
        "Epocas ejecutadas": len(h_p3_a.history["loss"]),
        "Accuracy entrenamiento": round(float(acc_train_a), 4),
        "Accuracy validacion": np.nan,
        "Accuracy prueba": round(float(acc_test_a), 4),
        "Loss prueba": round(float(loss_test_a), 6),
        "Uso metodologico": "No permite seleccionar hiperparametros sin tocar prueba",
    },
    {
        "Esquema": "70/15/15",
        "Arquitectura inicial": "1 capa oculta, 16 neuronas, lr=0.001",
        "Entrenamiento": int(len(y_trainB_sm)),
        "Validacion": int(len(y_valB)),
        "Prueba": int(len(y_testB)),
        "Epocas ejecutadas": len(h_p3_b.history["loss"]),
        "Accuracy entrenamiento": round(float(acc_train_b), 4),
        "Accuracy validacion": round(float(acc_val_b), 4),
        "Accuracy prueba": round(float(acc_test_b), 4),
        "Loss prueba": round(float(loss_test_b), 6),
        "Uso metodologico": "Permite ajustar hiperparametros con validacion independiente",
    },
])
save_csv(p3_comparison, "p3_comparacion_esquemas_particion.csv")
print("\nComparacion experimental de esquemas con MLP inicial:")
print(p3_comparison.to_string(index=False))

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

p3_b_pred = np.argmax(mlp_p3_b.predict(X_testB, verbose=0), axis=1)
p3_b_f1_macro = f1_score(y_testB, p3_b_pred, average="macro", zero_division=0)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
p4_fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_enc), start=1):
    X_train_fold_raw, X_val_fold_raw = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y_enc[train_idx], y_enc[val_idx]

    fold_scaler = StandardScaler()
    X_train_fold = fold_scaler.fit_transform(X_train_fold_raw)
    X_val_fold = fold_scaler.transform(X_val_fold_raw)

    try:
        sm_fold = SMOTE(random_state=42)
        X_train_fold_sm, y_train_fold_sm = sm_fold.fit_resample(
            X_train_fold, y_train_fold
        )
        smote_status = "Si"
    except Exception:
        X_train_fold_sm, y_train_fold_sm = X_train_fold, y_train_fold
        smote_status = "No"

    keras.backend.clear_session()
    keras.utils.set_random_seed(42 + fold)
    mlp_fold = build_mlp(
        len(FEATURES), n_classes, p3_hidden_layers, p3_neurons, p3_lr
    )
    hist_fold = train_model(
        mlp_fold,
        X_train_fold_sm,
        y_train_fold_sm,
        X_val_fold,
        y_val_fold,
        epochs=p3_epochs,
        batch_size=p3_batch_size,
    )

    y_pred_fold = np.argmax(mlp_fold.predict(X_val_fold, verbose=0), axis=1)
    loss_train_fold, acc_train_fold = mlp_fold.evaluate(
        X_train_fold_sm, y_train_fold_sm, verbose=0
    )
    loss_val_fold, acc_val_fold = mlp_fold.evaluate(
        X_val_fold, y_val_fold, verbose=0
    )
    f1_macro_fold = f1_score(
        y_val_fold, y_pred_fold, average="macro", zero_division=0
    )

    p4_fold_results.append({
        "Fold": fold,
        "Entrenamiento original": int(len(y_train_fold)),
        "Entrenamiento usado": int(len(y_train_fold_sm)),
        "Validacion": int(len(y_val_fold)),
        "SMOTE aplicado": smote_status,
        "Epocas ejecutadas": len(hist_fold.history["loss"]),
        "Accuracy entrenamiento": round(float(acc_train_fold), 4),
        "Accuracy validacion": round(float(acc_val_fold), 4),
        "F1 macro validacion": round(float(f1_macro_fold), 4),
        "Loss validacion": round(float(loss_val_fold), 6),
    })

p4_kfold = pd.DataFrame(p4_fold_results)
save_csv(p4_kfold, "p4_kfold_mlp_por_fold.csv")

p4_comparison = pd.DataFrame([
    {
        "Estrategia": "Particion fija 70/15/15",
        "Modelo": "MLP inicial",
        "Conjunto evaluado": "Prueba independiente",
        "Accuracy medio": round(float(acc_test_b), 4),
        "Desviacion accuracy": np.nan,
        "F1 macro medio": round(float(p3_b_f1_macro), 4),
        "Desviacion F1 macro": np.nan,
        "Uso recomendado": "Evaluacion final trazable",
    },
    {
        "Estrategia": "K-fold estratificado (k=5)",
        "Modelo": "MLP inicial",
        "Conjunto evaluado": "Validaciones internas",
        "Accuracy medio": round(float(p4_kfold["Accuracy validacion"].mean()), 4),
        "Desviacion accuracy": round(float(p4_kfold["Accuracy validacion"].std(ddof=1)), 4),
        "F1 macro medio": round(float(p4_kfold["F1 macro validacion"].mean()), 4),
        "Desviacion F1 macro": round(float(p4_kfold["F1 macro validacion"].std(ddof=1)), 4),
        "Uso recomendado": "Estimacion de variabilidad y apoyo a hiperparametros",
    },
])
save_csv(p4_comparison, "p4_comparacion_kfold_particion_fija.csv")

print("\nResultados k-fold del MLP inicial por fold:")
print(p4_kfold.to_string(index=False))
print("\nComparacion entre particion fija y k-fold:")
print(p4_comparison.to_string(index=False))
print("""
Conclusión P4:
  • Con n ≈ 200 la varianza de la estimación con una sola partición puede ser
    alta. K-fold (k=5) permite medir esa variabilidad con el mismo tipo de
    modelo usado en el taller.
  • El escalador y SMOTE se ajustan dentro de cada fold usando solo el
    entrenamiento correspondiente, evitando fuga de información.
  • Se recomienda usar k-fold como complemento para seleccionar
    hiperparámetros y conservar la partición fija para la evaluación final.
""")


# ─────────────────────────────────────────────────────────────────────────────
# PREGUNTA 5 – Comparación de arquitecturas
# ─────────────────────────────────────────────────────────────────────────────
print("="*70)
print("PREGUNTA 5 – Comparación de arquitecturas del MLP")
print("="*70)

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

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

y_pred_cce_p6 = np.argmax(m_cce.predict(X_testB, verbose=0), axis=1)
y_pred_mse_p6 = np.argmax(m_mse.predict(X_testB, verbose=0), axis=1)
acc_cce = accuracy_score(y_testB, y_pred_cce_p6)
acc_mse = accuracy_score(y_testB, y_pred_mse_p6)

p6_comparison = pd.DataFrame([
    {
        "Funcion perdida": "Entropia cruzada categorica",
        "Implementacion": "sparse_categorical_crossentropy",
        "Arquitectura": best_cfg,
        "Formato etiquetas": "Enteras",
        "Epocas ejecutadas": len(h_cce.history["loss"]),
        "Loss Train minima": round(float(min(h_cce.history["loss"])), 6),
        "Loss Val minima": round(float(min(h_cce.history["val_loss"])), 6),
        "Acc Train max": round(float(max(h_cce.history["accuracy"])), 4),
        "Acc Val max": round(float(max(h_cce.history["val_accuracy"])), 4),
        "Accuracy prueba": round(float(acc_cce), 4),
        "Aciertos prueba": int(np.sum(y_pred_cce_p6 == y_testB)),
        "Total prueba": int(len(y_testB)),
        "Decision metodologica": "Recomendada para clasificacion multiclase con softmax",
    },
    {
        "Funcion perdida": "Error cuadratico medio",
        "Implementacion": "mean_squared_error",
        "Arquitectura": best_cfg,
        "Formato etiquetas": "One-hot",
        "Epocas ejecutadas": len(h_mse.history["loss"]),
        "Loss Train minima": round(float(min(h_mse.history["loss"])), 6),
        "Loss Val minima": round(float(min(h_mse.history["val_loss"])), 6),
        "Acc Train max": round(float(max(h_mse.history["accuracy"])), 4),
        "Acc Val max": round(float(max(h_mse.history["val_accuracy"])), 4),
        "Accuracy prueba": round(float(acc_mse), 4),
        "Aciertos prueba": int(np.sum(y_pred_mse_p6 == y_testB)),
        "Total prueba": int(len(y_testB)),
        "Decision metodologica": "Comparativa; no recomendada como criterio principal",
    },
])
save_csv(p6_comparison, "p6_comparacion_funciones_perdida.csv")

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

target_names = [f"Sev {c}" for c in classes_orig]
acc_multi = accuracy_score(y_testB, y_pred_multi)
report_multi = classification_report(
    y_testB,
    y_pred_multi,
    target_names=target_names,
    output_dict=True,
    zero_division=0,
)

p7_metrics_rows = []
for class_name in target_names:
    p7_metrics_rows.append({
        "Clase": class_name,
        "Precision": round(float(report_multi[class_name]["precision"]), 4),
        "Sensibilidad": round(float(report_multi[class_name]["recall"]), 4),
        "F1-score": round(float(report_multi[class_name]["f1-score"]), 4),
        "Soporte": int(report_multi[class_name]["support"]),
    })
for avg_name, label in [
    ("macro avg", "Macro-promedio"),
    ("weighted avg", "Promedio ponderado"),
]:
    p7_metrics_rows.append({
        "Clase": label,
        "Precision": round(float(report_multi[avg_name]["precision"]), 4),
        "Sensibilidad": round(float(report_multi[avg_name]["recall"]), 4),
        "F1-score": round(float(report_multi[avg_name]["f1-score"]), 4),
        "Soporte": int(report_multi[avg_name]["support"]),
    })
p7_metrics = pd.DataFrame(p7_metrics_rows)
save_csv(p7_metrics, "p7_metricas_clasificacion_multiclase.csv")

print(f"\nExactitud global (Test): {acc_multi:.4f}")
print("\nReporte de clasificación:")
print(classification_report(y_testB, y_pred_multi, target_names=target_names, zero_division=0))

fig, ax = plt.subplots(figsize=(7, 5))
cm = confusion_matrix(y_testB, y_pred_multi)
cm_table = pd.DataFrame(cm, columns=target_names)
cm_table.insert(0, "Real\\Predicho", target_names)
save_csv(cm_table, "p7_matriz_confusion_multiclase.csv")
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
auc_rows = []
for i, cls in enumerate(classes_orig):
    if y_bin_test[:, i].sum() == 0:
        continue
    fpr, tpr, _ = roc_curve(y_bin_test[:, i], y_prob_multi[:, i])
    auc_val = auc(fpr, tpr)
    auc_scores.append(auc_val)
    auc_rows.append({
        "Clase": f"Sev {cls}",
        "AUC-ROC": round(float(auc_val), 4),
        "Esquema": "One-vs-rest",
    })
    ax.plot(fpr, tpr, lw=1.8, label=f"Sev {cls} (AUC={auc_val:.2f})")

p7_auc = pd.DataFrame(auc_rows)
save_csv(p7_auc, "p7_auc_roc_ovr_multiclase.csv")
p7_summary = pd.DataFrame([{
    "Modelo": best_cfg,
    "Accuracy": round(float(acc_multi), 4),
    "Aciertos": int(np.sum(y_pred_multi == y_testB)),
    "Total prueba": int(len(y_testB)),
    "Precision macro": round(float(report_multi["macro avg"]["precision"]), 4),
    "Sensibilidad macro": round(float(report_multi["macro avg"]["recall"]), 4),
    "F1 macro": round(float(report_multi["macro avg"]["f1-score"]), 4),
    "Precision ponderada": round(float(report_multi["weighted avg"]["precision"]), 4),
    "Sensibilidad ponderada": round(float(report_multi["weighted avg"]["recall"]), 4),
    "F1 ponderado": round(float(report_multi["weighted avg"]["f1-score"]), 4),
    "AUC-ROC macro": round(float(np.mean(auc_scores)), 4),
}])
save_csv(p7_summary, "p7_resumen_metricas_multiclase.csv")

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

save_intermediate(
    globals(),
    "parte_i",
    [
        "df",
        "X_scaled",
        "capas_best",
        "neuronas_best",
        "lr_best",
        "classes_orig",
        "n_classes",
        "scaler",
        "X_testB",
        "y_testB",
        "y_pred_multi",
        "y_prob_multi",
    ],
)
