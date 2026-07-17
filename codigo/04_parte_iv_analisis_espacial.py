"""Parte IV: analisis de dependencia espacial."""

from _bootstrap import load_config, load_intermediate, save_intermediate

load_config(__file__, globals())
required_from_part_ii = [
    "df", "y_bin", "capas_best", "neuronas_best", "lr_best",
    "y_pred_bin", "y_prob_bin", "X_te_b", "y_te_b",
]
if not all(name in globals() for name in required_from_part_ii):
    load_intermediate(
        globals(),
        "parte_ii",
        required_keys=required_from_part_ii,
        producer="python codigo/02_parte_ii_clasificacion_binaria.py",
    )

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
    X_tr_sp_sm, y_tr_sp_sm = SMOTE(random_state=42).fit_resample(X_tr_sp, y_tr_sp)
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
    X_tr_i_sm, y_tr_i_sm = SMOTE(random_state=42).fit_resample(X_tr_i, y_tr_i)
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

save_intermediate(
    globals(),
    "parte_iv",
    [
        "acc_base",
        "auc_base",
        "acc_sp",
        "auc_sp",
        "acc_c",
        "auc_c",
        "acc_i",
        "auc_i",
    ],
)
