"""Parte II: clasificacion binaria de severidad."""

from _bootstrap import load_config, load_intermediate, save_intermediate, save_keras_model

load_config(__file__, globals())
required_from_part_i = ["df", "X_scaled", "capas_best", "neuronas_best", "lr_best"]
if not all(name in globals() for name in required_from_part_i):
    load_intermediate(
        globals(),
        "parte_i",
        required_keys=required_from_part_i,
        producer="python codigo/01_parte_i_clasificacion_multiple.py",
    )

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

p8_distribution = pd.DataFrame([
    {
        "Grupo": "Sano",
        "Codigo": 0,
        "Severidades originales": "1",
        "Observaciones": int(np.sum(y_bin == 0)),
        "Porcentaje": round(float(np.mean(y_bin == 0) * 100), 2),
    },
    {
        "Grupo": "Enfermo",
        "Codigo": 1,
        "Severidades originales": "2-6",
        "Observaciones": int(np.sum(y_bin == 1)),
        "Porcentaje": round(float(np.mean(y_bin == 1) * 100), 2),
    },
])
save_csv(p8_distribution, "p8_distribucion_binaria.csv")

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

p9_partition = pd.DataFrame([
    {
        "Conjunto": "Entrenamiento original",
        "Total": int(len(y_tr_b)),
        "Sano": int(np.sum(y_tr_b == 0)),
        "Enfermo": int(np.sum(y_tr_b == 1)),
    },
    {
        "Conjunto": "Entrenamiento con SMOTE",
        "Total": int(len(y_tr_b_sm)),
        "Sano": int(np.sum(y_tr_b_sm == 0)),
        "Enfermo": int(np.sum(y_tr_b_sm == 1)),
    },
    {
        "Conjunto": "Validacion",
        "Total": int(len(y_val_b)),
        "Sano": int(np.sum(y_val_b == 0)),
        "Enfermo": int(np.sum(y_val_b == 1)),
    },
    {
        "Conjunto": "Prueba",
        "Total": int(len(y_te_b)),
        "Sano": int(np.sum(y_te_b == 0)),
        "Enfermo": int(np.sum(y_te_b == 1)),
    },
])
save_csv(p9_partition, "p9_particion_binaria.csv")


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

print(
    f"Arquitectura binaria: {capas_best} capas ocultas, "
    f"{neuronas_best} neuronas/capa, lr={lr_best}"
)
best_bin_model = build_mlp_bin(len(FEATURES), capas_best, neuronas_best, lr_best)
print("Entrenando MLP binario con particion 70/15/15 y SMOTE solo en entrenamiento...")
h_bin = train_model(best_bin_model, X_tr_b_sm, y_tr_b_sm, X_val_b, y_val_b)

y_prob_bin = best_bin_model.predict(X_te_b, verbose=0).flatten()
y_pred_bin = (y_prob_bin >= 0.5).astype(int)
report_bin = classification_report(
    y_te_b,
    y_pred_bin,
    target_names=["Sano", "Enfermo"],
    output_dict=True,
    zero_division=0,
)

p9_metrics_rows = []
for class_name in ["Sano", "Enfermo"]:
    p9_metrics_rows.append({
        "Clase": class_name,
        "Precision": round(float(report_bin[class_name]["precision"]), 4),
        "Sensibilidad": round(float(report_bin[class_name]["recall"]), 4),
        "F1-score": round(float(report_bin[class_name]["f1-score"]), 4),
        "Soporte": int(report_bin[class_name]["support"]),
    })
for avg_name, label in [
    ("macro avg", "Macro-promedio"),
    ("weighted avg", "Promedio ponderado"),
]:
    p9_metrics_rows.append({
        "Clase": label,
        "Precision": round(float(report_bin[avg_name]["precision"]), 4),
        "Sensibilidad": round(float(report_bin[avg_name]["recall"]), 4),
        "F1-score": round(float(report_bin[avg_name]["f1-score"]), 4),
        "Soporte": int(report_bin[avg_name]["support"]),
    })
p9_metrics = pd.DataFrame(p9_metrics_rows)
save_csv(p9_metrics, "p9_metricas_binarias_umbral_05.csv")

tn_05, fp_05, fn_05, tp_05 = confusion_matrix(y_te_b, y_pred_bin).ravel()
p9_summary = pd.DataFrame([{
    "Modelo": f"MLP binario ({capas_best}L-{neuronas_best}N-lr{lr_best})",
    "Umbral": 0.5,
    "Accuracy": round(float(accuracy_score(y_te_b, y_pred_bin)), 4),
    "Aciertos": int(np.sum(y_pred_bin == y_te_b)),
    "Total prueba": int(len(y_te_b)),
    "TN": int(tn_05),
    "FP": int(fp_05),
    "FN": int(fn_05),
    "TP": int(tp_05),
    "Precision macro": round(float(report_bin["macro avg"]["precision"]), 4),
    "Sensibilidad macro": round(float(report_bin["macro avg"]["recall"]), 4),
    "F1 macro": round(float(report_bin["macro avg"]["f1-score"]), 4),
    "Precision ponderada": round(float(report_bin["weighted avg"]["precision"]), 4),
    "Sensibilidad ponderada": round(float(report_bin["weighted avg"]["recall"]), 4),
    "F1 ponderado": round(float(report_bin["weighted avg"]["f1-score"]), 4),
}])
save_csv(p9_summary, "p9_resumen_mlp_binario_umbral_05.csv")

if all(name in globals() for name in ["y_testB", "y_pred_multi"]):
    report_multi_common = classification_report(
        y_testB,
        y_pred_multi,
        output_dict=True,
        zero_division=0,
    )
    p9_comparison = pd.DataFrame([
        {
            "Modelo": "MLP multiclase",
            "Clases": 6,
            "Accuracy": round(float(accuracy_score(y_testB, y_pred_multi)), 4),
            "Precision macro": round(float(report_multi_common["macro avg"]["precision"]), 4),
            "Sensibilidad macro": round(float(report_multi_common["macro avg"]["recall"]), 4),
            "F1 macro": round(float(report_multi_common["macro avg"]["f1-score"]), 4),
            "Precision ponderada": round(float(report_multi_common["weighted avg"]["precision"]), 4),
            "Sensibilidad ponderada": round(float(report_multi_common["weighted avg"]["recall"]), 4),
            "F1 ponderado": round(float(report_multi_common["weighted avg"]["f1-score"]), 4),
        },
        {
            "Modelo": "MLP binario",
            "Clases": 2,
            "Accuracy": round(float(accuracy_score(y_te_b, y_pred_bin)), 4),
            "Precision macro": round(float(report_bin["macro avg"]["precision"]), 4),
            "Sensibilidad macro": round(float(report_bin["macro avg"]["recall"]), 4),
            "F1 macro": round(float(report_bin["macro avg"]["f1-score"]), 4),
            "Precision ponderada": round(float(report_bin["weighted avg"]["precision"]), 4),
            "Sensibilidad ponderada": round(float(report_bin["weighted avg"]["recall"]), 4),
            "F1 ponderado": round(float(report_bin["weighted avg"]["f1-score"]), 4),
        },
    ])
    save_csv(p9_comparison, "p9_comparacion_mlp_multiclase_binario.csv")

print(f"\nAccuracy binario (Test): {accuracy_score(y_te_b, y_pred_bin):.4f}")
print(classification_report(y_te_b, y_pred_bin, target_names=["Sano", "Enfermo"], zero_division=0))


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

p10_metrics = pd.DataFrame([{
    "Criterio umbral": "Youden",
    "Umbral": round(float(best_thr), 4),
    "Accuracy": round(float(acc_y), 4),
    "Precision": round(float(prec), 4),
    "Sensibilidad": round(float(sens), 4),
    "Especificidad": round(float(spec), 4),
    "F1-score": round(float(f1_y), 4),
    "AUC-ROC": round(float(auc_bin), 4),
    "TN": int(tn),
    "FP": int(fp),
    "FN": int(fn),
    "TP": int(tp),
}])
save_csv(p10_metrics, "p10_metricas_umbral_youden.csv")

spec_05 = tn_05 / (tn_05 + fp_05) if (tn_05 + fp_05) > 0 else 0
p10_threshold_comparison = pd.DataFrame([
    {
        "Umbral": 0.5,
        "Criterio": "Por defecto",
        "Accuracy": round(float(accuracy_score(y_te_b, y_pred_bin)), 4),
        "Sensibilidad": round(float(report_bin["Enfermo"]["recall"]), 4),
        "Especificidad": round(float(spec_05), 4),
        "F1 Enfermo": round(float(report_bin["Enfermo"]["f1-score"]), 4),
        "TN": int(tn_05),
        "FP": int(fp_05),
        "FN": int(fn_05),
        "TP": int(tp_05),
    },
    {
        "Umbral": round(float(best_thr), 4),
        "Criterio": "Youden",
        "Accuracy": round(float(acc_y), 4),
        "Sensibilidad": round(float(sens), 4),
        "Especificidad": round(float(spec), 4),
        "F1 Enfermo": round(float(f1_y), 4),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    },
])
save_csv(p10_threshold_comparison, "p10_comparacion_umbrales.csv")

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

h_bin_history = h_bin.history
save_keras_model(globals(), best_bin_model, "best_bin_model.keras")
save_intermediate(
    globals(),
    "parte_ii",
    [
        "df",
        "X_scaled",
        "capas_best",
        "neuronas_best",
        "lr_best",
        "y_bin",
        "X_tv_b",
        "X_te_b",
        "y_tv_b",
        "y_te_b",
        "X_tr_b",
        "X_val_b",
        "y_tr_b",
        "y_val_b",
        "X_tr_b_sm",
        "y_tr_b_sm",
        "y_prob_bin",
        "y_pred_bin",
        "auc_bin",
        "best_thr",
        "h_bin_history",
    ],
)
