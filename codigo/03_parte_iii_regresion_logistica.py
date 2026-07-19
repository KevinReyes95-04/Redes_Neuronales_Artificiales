"""Parte III: regresion logistica y comparacion de modelos."""

from _bootstrap import load_config, load_intermediate, load_keras_model, save_intermediate

load_config(__file__, globals())
required_from_part_ii = [
    "X_tr_b_sm", "y_tr_b_sm", "X_te_b", "y_te_b", "y_prob_bin", "y_pred_bin"
]
if not all(name in globals() for name in required_from_part_ii):
    load_intermediate(
        globals(),
        "parte_ii",
        required_keys=required_from_part_ii,
        producer="python codigo/02_parte_ii_clasificacion_binaria.py",
    )
load_keras_model(globals(), "best_bin_model", "best_bin_model.keras")

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
save_csv(
    pd.DataFrame([{
        "Parametro": "Intercepto",
        "Coeficiente": intercept,
    }]).round(6),
    "p11_intercepto_logistico.csv",
)

if valid_se:
    variables_espectrales = {"ndvi_med", "evi_med", "ndre_med", "gli_med"}

    def evaluar_signo_esperado(row):
        variable = row["Variable"]
        coeficiente = row["Coeficiente"]
        if variable in variables_espectrales:
            return "Si" if coeficiente < 0 else "No"
        if variable == "height_med":
            return "No claro" if coeficiente > 0 else "Si"
        return "No evaluado"

    df_p11 = df_coefs.copy()
    df_p11["Significativa (alpha=0.05)"] = np.where(
        df_p11["p-valor"] < 0.05,
        "Si",
        "No",
    )
    df_p11["Signo esperado"] = df_p11.apply(evaluar_signo_esperado, axis=1)
    save_csv(df_p11.round(6), "p11_coeficientes_logisticos_inferencia.csv")

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
save_csv(df_lasso.round(6), "p12_importancia_lasso.csv")
save_csv(
    pd.DataFrame([{
        "Metodo": "Regresion logistica Lasso (L1)",
        "Validacion cruzada": 5,
        "C optimo": log_lasso.C_[0],
        "Variables retenidas": int((df_lasso["Abs(Coef)"] > 0).sum()),
    }]).round(6),
    "p12_parametros_lasso.csv",
)

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

def prediccion_youden(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best_idx = int(np.argmax(tpr - fpr))
    best_thr = float(thresholds[best_idx])
    return best_thr, (y_prob >= best_thr).astype(int)

def metricas(modelo, criterio, umbral, y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp/(tp+fn) if (tp+fn) > 0 else 0
    spec = tn/(tn+fp) if (tn+fp) > 0 else 0
    return {
        "Modelo": modelo,
        "Criterio umbral": criterio,
        "Umbral": round(float(umbral), 4),
        "Accuracy":      round(accuracy_score(y_true, y_pred), 4),
        "Precision":     round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Sensibilidad":  round(sens, 4),
        "Especificidad": round(spec, 4),
        "F1-score":      round(f1_score(y_true, y_pred, zero_division=0), 4),
        "AUC-ROC":       round(roc_auc_score(y_true, y_prob), 4),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

thr_mlp_youden, y_pred_mlp_youden = prediccion_youden(y_te_b, y_prob_bin)
thr_lr_youden, y_pred_lr_youden = prediccion_youden(y_te_b, y_prob_lr)

df_comp_05 = pd.DataFrame([
    metricas("MLP Binario", "Por defecto", 0.5, y_te_b, y_pred_bin, y_prob_bin),
    metricas("Regresión Logística", "Por defecto", 0.5, y_te_b, y_pred_lr, y_prob_lr),
])
df_comp_youden = pd.DataFrame([
    metricas("MLP Binario", "Youden", thr_mlp_youden, y_te_b, y_pred_mlp_youden, y_prob_bin),
    metricas("Regresión Logística", "Youden", thr_lr_youden, y_te_b, y_pred_lr_youden, y_prob_lr),
])
df_comp = df_comp_youden

print("\n── Tabla comparativa final ──")
print(df_comp.to_string(index=False))
save_csv(df_comp_05, "p13_comparacion_modelos_umbral_05.csv")
save_csv(df_comp_youden, "p13_comparacion_modelos_youden.csv")
save_csv(df_comp_youden, "08_comparacion_modelos.csv")

fig, ax = plt.subplots(figsize=(9, 4))
metricas_lbl = ["Accuracy", "Sensibilidad", "F1-score", "AUC-ROC"]
x = np.arange(len(metricas_lbl))
w = 0.35
mlp_y = df_comp_youden.loc[df_comp_youden["Modelo"] == "MLP Binario", metricas_lbl].iloc[0]
lr_y = df_comp_youden.loc[df_comp_youden["Modelo"] == "Regresión Logística", metricas_lbl].iloc[0]
ax.bar(x - w/2, mlp_y.values, w, label="MLP", color="steelblue", edgecolor="black")
ax.bar(x + w/2, lr_y.values,  w, label="Reg. Logística", color="coral", edgecolor="black")
ax.set_xticks(x)
ax.set_xticklabels(metricas_lbl, rotation=30, ha="right")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Valor de la métrica", fontsize=11)
ax.set_title("Figura 14 – Comparación MLP vs Regresión Logística con umbral de Youden (P13)",
             fontsize=12, fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.3)
save_figure(fig, "14_comparacion_modelos.png")
print("✔ Figura 14 guardada: 14_comparacion_modelos.png")
print("""
Conclusión P13:
  • Con umbral 0.5, el MLP supera a la regresión logística en sensibilidad
    y F1-score, aunque la regresión logística conserva mayor AUC-ROC.
  • Con umbral de Youden para ambos modelos, la regresión logística obtiene
    mayor sensibilidad, F1-score y AUC-ROC.
  • Por desempeño ajustado por umbral, interpretabilidad y bajo costo
    computacional, la regresión logística es una alternativa preferible
    para un sistema de apoyo a la decisión agronómica.
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

df_lasso_rank = df_lasso.copy()
df_lasso_rank["Ranking Lasso"] = range(1, len(df_lasso_rank) + 1)
df_pfi_rank = df_pfi.copy()
df_pfi_rank["Ranking MLP PFI"] = range(1, len(df_pfi_rank) + 1)
df_importancia_comp = (
    df_lasso_rank.merge(df_pfi_rank, on="Variable", how="inner")
    [[
        "Variable",
        "Ranking Lasso",
        "Coef. Lasso",
        "Abs(Coef)",
        "Ranking MLP PFI",
        "Importancia PFI",
        "Std",
    ]]
    .sort_values("Ranking Lasso")
)
save_csv(df_importancia_comp.round(6), "p12_comparacion_lasso_mlp_pfi.csv")
save_csv(df_importancia_comp.round(6), "p14_comparacion_importancia_variables.csv")

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

save_intermediate(
    globals(),
    "parte_iii",
    [
        "y_prob_lr",
        "y_pred_lr",
        "y_pred_lr_youden",
        "auc_lr",
        "df_coefs",
        "df_lasso",
        "df_comp",
        "df_comp_05",
        "df_comp_youden",
        "df_pfi",
    ],
)
