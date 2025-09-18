#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pipeline para BAF (usando Base.csv):
- Split: train (month 0–5), val (6), test (7)
- Preprocesamiento: relleno (mediana/moda) + OneHot + escalado (fit en train)
- Modelo: LightGBM con peso de clase
- Selección de umbral en val. para FPR=0.05 
- Evaluación en test: recall y FPR usando ese umbral 
- Además: métrica de reporte --> recall en TEST en 5% FPR
- Fairness (Predictive Equality): FPR por edad (<50 vs >=50) y ratio
Salidas (exporta):
- metrics_baf_base.json / metrics_baf_base.csv
- val_roc_base.png
- test_fpr_by_age.png
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb  # modelo de gradient boosting para tabulares
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer  # para relleno de missing values
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ctes
FPR_OBJ: float = 0.05 # objetivo de FPR para el umbral (como pide el reto)
AGE_SPLIT: int = 50 # corte de edad para fairness
DEFAULT_SEED: int = 123 

# utilidades
def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)  # semilla random python
    np.random.seed(seed)  # semilla Numpy

def false_positive_rate(y_true: np.ndarray, y_pred_bin: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin, labels=[0, 1]).ravel()  # extrayendo TN, FP, FN, TP
    denom = tn + fp  # negativos reales
    return (fp / denom) if denom > 0 else 0.0  # FPR = FP / (TN+FP), protege contra división entre cero

def recall(y_true: np.ndarray, y_pred_bin: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin, labels=[0, 1]).ravel()
    denom = tp + fn  # positivos reales
    return (tp / denom) if denom > 0 else 0.0  # recall = TP / (TP+FN)

def ratio_min_max(a: float, b: float) -> float:
    x, y = (a, b) if a <= b else (b, a)  # ordena (min, max)
    return (x / y) if y > 0 else 0.0  # razón min/max; 0 si el máximo es 0

# datos/split
def split_mes(csv_path: str, month_col: str = "month", target: str = "fraud_bool"
                        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    train = df[df[month_col].isin([0, 1, 2, 3, 4, 5])].copy()  # 0-5 train
    val   = df[df[month_col] == 6].copy()  # 6 val
    test  = df[df[month_col] == 7].copy()  # 7 test
    return train, val, test

# preprocesamiento
NEG_A_NAN = [  # columnas donde un valor negativo = faltante
    "prev_address_months_count",
    "current_address_months_count",
    "intended_balcon_amount",
    "session_length_in_minutes",
    "bank_months_count",
    "device_distinct_emails_8w",
]
NEG_A_NAN_FALL = ["device_distinct_emails", "device_distinct_emails_8w"]  # nombre alterno (fallback)

# reemplazar negs por NaN (antes de relleno)
def neg_a_nan(df: pd.DataFrame, cols: list, alt_names: list) -> None:
    for c in cols + alt_names:
        if c in df.columns:
            df.loc[df[c] < 0, c] = np.nan

def preprocesador(df_train: pd.DataFrame, target: str) -> Tuple[ColumnTransformer, list, list]:
    num_cols = df_train.select_dtypes(include=["int64", "float64"]).columns.tolist()  # numéricas por dtype
    cat_cols = df_train.select_dtypes(include=["object"]).columns.tolist()  # categóricas por dtype
    # excluyendo target
    num_cols = [c for c in num_cols if c != target]
    cat_cols = [c for c in cat_cols if c != target]
    # pipeline var numéricas
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),  # imput. numéricas con mediana
        ("sc", StandardScaler())  # escala numéricas a media 0, varianza 1
    ])
    # pipeline var categóricas
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),  # imput. categóricas con moda
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop")  # columnas que no necesitamos

    return pre, num_cols, cat_cols

# modelo
def train_lgbm(Xtr, ytr, Xva, yva, random_state: int = DEFAULT_SEED) -> lgb.LGBMClassifier:
    n_neg = int((ytr == 0).sum())  # negs en train
    n_pos = int((ytr == 1).sum())  # pos en train
    w_pos = float(n_neg / max(n_pos, 1))  # balanceo (peso para la clase positiva)
    # para LightGBM
    clf = lgb.LGBMClassifier(
        verbose=-1,
        n_estimators=1800,  # num. árboles
        learning_rate=0.03,
        num_leaves=128,
        min_child_samples=80,  # control de overfit
        subsample=0.8,  # bagging (filas por árbol)
        colsample_bytree=0.8,  # feature subsampling (col por árbol)
        reg_lambda=1.0,  # l2
        objective="binary",
        class_weight={0: 1.0, 1: w_pos},  # pesos de clase para quitar desbalance
        n_jobs=-1,
        random_state=random_state,
    )
    clf.fit(
        Xtr, ytr,  # ajustando con datos transformados de train
        eval_set=[(Xva, yva)],  # conjunto de val para early stopping y métrica
        eval_metric="auc",  # optimizando AUC en validación reference metric
        # si no mejora en 200 iters, detener
        callbacks=[lgb.early_stopping(200),
                   lgb.log_evaluation(period=0)], # consola verbosa
    )
    return clf  # modelo entrenado

# eval
@dataclass
class Metrics:
    auc_val: float  # AUC en val
    apval: float  # AP (PR-AUC) en val
    threshold: float  # umbral operativo elegido en val (FPR=0.05)
    fpr_val: float  # FPR en val usando threshold
    recall_val: float  # recall en validación usando threshold
    auc_test: float  # AUC en test
    apttest: float  # AP en test
    fpr_test: float  # FPR en test con threshold de val
    recall_test: float  # recall en test con threshold de validación
    recall_test_curve_at_5: float  # recall en test a FPR = 0.05 leído de la curva (no cambia threshold)
    fpr_test_age_mayor50: float  # FPR de clientes con edad >=50
    fpr_test_age_menor50: float  # FPR de clientes con edad <50
    fpr_ratio_age_groups: float  # min/max de FPR entre grupos de edad (predictive equality)

def evaluate(csv_path: str,
             out_dir: str = ".",
             seed: int = DEFAULT_SEED,
             age_split: int = AGE_SPLIT,
             make_plots: bool = True) -> Metrics:
    set_seed(seed)
    target = "fraud_bool"
    train, val, test = split_mes(csv_path, target=target)

    # limpiar los codified missing antes
    for part in (train, val, test):
        neg_a_nan(part, NEG_A_NAN, NEG_A_NAN_FALL)

    # separar X/y, features y etiquetas
    Xtr, ytr = train.drop(columns=[target]), train[target].values
    Xva, yva = val.drop(columns=[target]),   val[target].values
    Xte, yte = test.drop(columns=[target]),  test[target].values

    # preprocesamiento sin fugas (fit solo en train)
    pre, _, _ = preprocesador(train, target)  # transformamos cols
    X_train = pre.fit_transform(Xtr)  # ajustamos prepro con train y transformamos train
    X_val = pre.transform(Xva)  # trans. validación con el mismo prepro
    X_test = pre.transform(Xte)  # trans. test con el mismo prepro

    # train LightGBM con val
    clf = train_lgbm(X_train, ytr, X_val, yva, random_state=seed)

    # validación
    proba_val = clf.predict_proba(X_val)[:, 1]  # odds clase positiva en val
    auc_val = float(roc_auc_score(yva, proba_val))  # AUC en val
    apval  = float(average_precision_score(yva, proba_val))  # AP (PR-AUC) en val

    # umbral operativo (FPR=FPR_OBJ en val)
    neg_scores_val = proba_val[yva == 0]  # extraemos scores de la clase negativa
    threshold = float(np.quantile(neg_scores_val, 1.0 - FPR_OBJ))  # deja FPR_OBJ de negativos por arriba
    yhat_va = (proba_val >= threshold).astype(int)  # preds con el umbral
    fpr_val = false_positive_rate(yva, yhat_va)  # FPR resultante en val
    recall_val = recall(yva, yhat_va)  # recall en val

    # test (mismo umbral)
    # probs, AUC y AP en test
    proba_test = clf.predict_proba(X_test)[:, 1]
    auc_test = float(roc_auc_score(yte, proba_test))
    apttest  = float(average_precision_score(yte, proba_test))
    # predicciones binarias en test con threshold de val, FPR, y recall en test con ese umbral
    yhat_te = (proba_test >= threshold).astype(int)
    fpr_test = float(false_positive_rate(yte, yhat_te))
    recall_test = float(recall(yte, yhat_te))

    # also: recall en test a FPR = FPR_OBJ usando la curva (no cambia el umbral)
    fpr_curve, tpr_curve, _ = roc_curve(yte, proba_test)  # curva ROC (FPR, TPR) en test
    recall_test_curve_at_5 = float(np.interp(FPR_OBJ, fpr_curve, tpr_curve))  # interpolando TPR en FPR objetivo

    # fairness por edad (predictive equality)
    ages = test["customer_age"].values
    arriba_50 = ages >= age_split
    # FPR grupo mayor y menor, y su razón
    fpr_age_ge50 = float(false_positive_rate(yte[arriba_50], yhat_te[arriba_50])) if arriba_50.any() else 0.0
    fpr_age_lt50 = float(false_positive_rate(yte[~arriba_50], yhat_te[~arriba_50])) if (~arriba_50).any() else 0.0
    fpr_ratio_age = ratio_min_max(fpr_age_ge50, fpr_age_lt50)

    # exportar las métricas
    os.makedirs(out_dir, exist_ok=True)
    metrics = {
    "auc_val": auc_val,
    "ap_val": apval,
    "umbral_val_fpr5": threshold,
    "fpr_val": fpr_val,
    "recall_val": recall_val,
    "auc_test": auc_test,
    "ap_test": apttest,
    "fpr_test_umbral_val": fpr_test,
    "recall_test_umbral_val": recall_test,
    "recall_test_fpr5_curve": recall_test_curve_at_5,
    "fpr_test_age_menor50": fpr_age_lt50, 
    "fpr_test_age_mayor50": fpr_age_ge50, 
    "fpr_ratio_age": fpr_ratio_age,  # entre grupos
    }

    # métricas a archivo JSON
    with open(os.path.join(out_dir, "metrics_baf_base.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame([metrics]).to_csv(
        os.path.join(out_dir, "metrics_baf_base.csv"), index=False  # a csv
    )

    # plots
    if make_plots:
        # ROC (val) con punto operativo
        fpr_v, tpr_v, _ = roc_curve(yva, proba_val)  # curva ROC en val
        idx = int(np.argmin((fpr_v - fpr_val) ** 2 + (tpr_v - recall_val) ** 2))  # idx del punto más cercano al operativo
        plt.figure()
        plt.plot(fpr_v, tpr_v, label="ROC (Validación)")
        plt.scatter([fpr_v[idx]], [tpr_v[idx]], label=f"Punto operativo (FPR={FPR_OBJ:.0%}, val)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate (Recall)")
        plt.title("Validación ROC (Base)")
        plt.legend()
        plt.savefig(os.path.join(out_dir, "val_roc_base.png"), bbox_inches="tight", dpi=150)
        plt.close()
        # FPR por grupo de edad (test)
        plt.figure()
        plt.bar([f"edad<{age_split}", f"edad>={age_split}"], [fpr_age_lt50, fpr_age_ge50])
        plt.ylabel("FPR")
        plt.title("FPR por grupo de edad (Test)")
        plt.savefig(os.path.join(out_dir, "test_fpr_by_age.png"), bbox_inches="tight", dpi=150)
        plt.close()

    return Metrics(
        auc_val=auc_val,
        apval=apval,
        threshold=threshold,
        fpr_val=fpr_val,
        recall_val=recall_val,
        auc_test=auc_test,
        apttest=apttest,
        fpr_test=fpr_test,
        recall_test=recall_test,
        recall_test_curve_at_5=recall_test_curve_at_5,
        fpr_test_age_mayor50=fpr_age_ge50,
        fpr_test_age_menor50=fpr_age_lt50, 
        fpr_ratio_age_groups=fpr_ratio_age, 
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Ruta a Base.csv (p.ej., Datasets/Base.csv)")
    parser.add_argument("--out_dir", default="outputs", help="Carpeta de salida")
    args = parser.parse_args()
    bundle = evaluate(
        csv_path=args.csv,
        out_dir=args.out_dir,
        seed=DEFAULT_SEED,
        age_split=AGE_SPLIT,
        make_plots=True,
    )
    print("\nValidación")
    print(f"AUC: {bundle.auc_val:.3f}")
    print(f"AP (PR-AUC): {bundle.apval:.3f}")
    print(f"Umbral operativo (FPR 5%): {bundle.threshold:.3f}")
    print(f"FPR: {bundle.fpr_val:.3f}")
    print(f"Recall: {bundle.recall_val:.3f}")
    print("\nTest (usando umbral de validación)")
    print(f"AUC: {bundle.auc_test:.3f}")
    print(f"AP (PR-AUC): {bundle.apttest:.3f}")
    print(f"FPR (umbral val): {bundle.fpr_test:.3f}")
    print(f"Recall (umbral val): {bundle.recall_test:.3f}")
    print(f"Recall (curve, FPR=5%): {bundle.recall_test_curve_at_5:.3f}")
    print("\nFairness: Predictive Equality (edad)")
    print(f"FPR <{AGE_SPLIT} años: {bundle.fpr_test_age_menor50:.3f}")
    print(f"FPR ≥{AGE_SPLIT} años: {bundle.fpr_test_age_mayor50:.3f}")
    print(f"Ratio FPR (min/max): {bundle.fpr_ratio_age_groups:.3f}")

if __name__ == "__main__":
    main()