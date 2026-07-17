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
