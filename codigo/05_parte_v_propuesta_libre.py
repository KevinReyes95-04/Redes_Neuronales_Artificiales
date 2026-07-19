"""Parte V: propuesta libre del estudiante."""

from _bootstrap import load_config, load_intermediate, save_intermediate

load_config(__file__, globals())
required_from_part_ii = [
    "X_tr_b_sm", "y_tr_b_sm", "X_val_b", "y_val_b", "X_te_b", "y_te_b",
    "capas_best", "neuronas_best", "lr_best", "h_bin", "y_prob_bin",
]
if not all(name in globals() for name in required_from_part_ii):
    load_intermediate(
        globals(),
        "parte_ii",
        required_keys=required_from_part_ii,
        producer="python codigo/02_parte_ii_clasificacion_binaria.py",
    )

required_from_part_iii = ["auc_lr", "y_pred_lr", "y_prob_lr"]
if not all(name in globals() for name in required_from_part_iii):
    load_intermediate(
        globals(),
        "parte_iii",
        required_keys=required_from_part_iii,
        producer="python codigo/03_parte_iii_regresion_logistica.py",
    )

required_from_part_iv = ["acc_base", "auc_base"]
if not all(name in globals() for name in required_from_part_iv):
    load_intermediate(
        globals(),
        "parte_iv",
        required_keys=required_from_part_iv,
        producer="python codigo/04_parte_iv_analisis_espacial.py",
    )

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
save_csv(df_rf_imp.round(6), "p19_importancia_random_forest.csv")

save_csv(
    pd.DataFrame([
        {
            "Extension": "Regularizacion por Dropout",
            "Objetivo": "Evaluar si dropout reduce sobreajuste y mejora generalizacion del MLP binario",
            "Implementacion": "Dropout 0.3 despues de cada capa oculta y early stopping",
        },
        {
            "Extension": "Comparacion con Random Forest",
            "Objetivo": "Contrastar el MLP con un algoritmo no neuronal usando la misma particion y metricas",
            "Implementacion": "Random Forest con 200 arboles y class_weight='balanced'",
        },
    ]),
    "p19_configuracion_propuesta.csv",
)

def metricas_youden(modelo, y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best_idx = int(np.argmax(tpr - fpr))
    threshold = float(thresholds[best_idx])
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp/(tp+fn) if (tp+fn) > 0 else 0
    spec = tn/(tn+fp) if (tn+fp) > 0 else 0
    return {
        "Modelo": modelo,
        "Criterio umbral": "Youden",
        "Umbral": round(threshold, 4),
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Sensibilidad": round(float(sens), 4),
        "Especificidad": round(float(spec), 4),
        "F1-score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "AUC-ROC": round(float(roc_auc_score(y_true, y_prob)), 4),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

df_modelos_p19 = pd.DataFrame([
    metricas_youden("Regresion logistica", y_te_b, y_prob_lr),
    metricas_youden("MLP binario base", y_te_b, y_prob_bin),
    metricas_youden("MLP con Dropout", y_te_b, y_prob_drop),
    metricas_youden("Random Forest", y_te_b, y_prob_rf),
])
save_csv(df_modelos_p19, "p19_comparacion_modelos.csv")

def resumen_historial(hist, modelo):
    loss = hist.history["loss"]
    val_loss = hist.history["val_loss"]
    min_val_idx = int(np.argmin(val_loss))
    return {
        "Modelo": modelo,
        "Epocas entrenadas": len(loss),
        "Loss train final": round(float(loss[-1]), 6),
        "Loss val final": round(float(val_loss[-1]), 6),
        "Loss val minima": round(float(val_loss[min_val_idx]), 6),
        "Epoca mejor val_loss": min_val_idx + 1,
    }

save_csv(
    pd.DataFrame([
        resumen_historial(h_bin, "MLP binario base"),
        resumen_historial(h_drop, "MLP con Dropout"),
    ]),
    "p19_resumen_dropout.csv",
)

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

all_models = df_modelos_p19["Modelo"].tolist()
all_aucs   = df_modelos_p19["AUC-ROC"].tolist()
all_accs   = df_modelos_p19["Accuracy"].tolist()

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
ax.set_title("Figura 19 – Comparación final de todos los modelos (P19)",
             fontsize=12, fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.3)
save_figure(fig, "19_comparacion_final.png")
print("✔ Figuras 18-19 guardadas.")

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(df_rf_imp["Variable"], df_rf_imp["Importancia"],
        color="mediumorchid", edgecolor="black", alpha=0.85)
ax.set_xlabel("Importancia (Gini)")
ax.set_title("Figura 20 – Importancia de variables – Random Forest (P19)",
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
resumen = df_modelos_p19[["Modelo", "Accuracy", "AUC-ROC"]].copy()
print(resumen.to_string(index=False))
save_csv(resumen, "12_resumen_ejecutivo.csv")
save_intermediate(
    globals(),
    "parte_v",
    ["resumen", "acc_drop", "auc_drop", "acc_rf", "auc_rf", "df_rf_imp"],
)
print("""
Recomendación final:
  • Para un sistema de apoyo a la decisión agronómica se sugiere priorizar
    el modelo con mayor AUC-ROC y sensibilidad, ajustando el umbral para
    minimizar falsos negativos.
  • Dropout debe evaluarse empiricamente: en esta configuracion solo seria
    recomendable si mejora las metricas frente al MLP base.
  • Random Forest funciona como contraste no neuronal e insumo adicional
    para interpretar la importancia de variables.
""")
print(f"Tablas guardadas en: {TABLES_DIR}")
print(f"Figuras guardadas en: {FIGURES_DIR}")
