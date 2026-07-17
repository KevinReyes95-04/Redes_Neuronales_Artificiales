"""Configuracion comun del taller de perceptron multicapa.

Este archivo se ejecuta primero desde 99_ejecutar_todo.py.
"""

# %%
import os
import warnings
from pathlib import Path
from tempfile import gettempdir

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLCONFIGDIR", str(Path(gettempdir()) / "rna_taller_matplotlib"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

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
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT
RESULTS_DIR = PROJECT_ROOT / "resultados"
INTERMEDIATE_DIR = RESULTS_DIR / "intermedios"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ntrain_median_values.csv"
TABLES_DIR = RESULTS_DIR / "tablas"
FIGURES_DIR = RESULTS_DIR / "figuras"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

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


def train_model(model, X_tr, y_tr, X_val, y_val, epochs=150, batch_size=16):
    """Entrena un modelo Keras con early stopping."""
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=20,
                                         restore_best_weights=True)]
    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        verbose=0, callbacks=cb
    )
    return hist


def build_mlp_bin(n_inputs, hidden_layers, neurons, lr=0.01):
    """Construye un MLP binario con salida sigmoide."""
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
