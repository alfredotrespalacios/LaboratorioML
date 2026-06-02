import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from statsmodels.stats.stattools import durbin_watson
from utils.estadisticas import calcular_acf_pacf

def metricas_regresion(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {"R2": float(r2), "MAE": float(mae), "MSE": float(mse), "RMSE": float(rmse)}

def metricas_clasificacion(y_true, y_pred, y_prob=None):
    salida = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            salida["AUC"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            salida["AUC"] = None
    return salida

def matriz_confusion_df(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"Real {x}" for x in (labels if labels is not None else range(cm.shape[0]))],
                        columns=[f"Pred {x}" for x in (labels if labels is not None else range(cm.shape[1]))])

def curva_roc_df(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})

def diagnostico_residuales(residuales, max_lags=20):
    resid = pd.Series(residuales).dropna()
    if len(resid) == 0:
        return {}
    jb = jarque_bera(resid)
    acfs = calcular_acf_pacf(resid, max_lags=max_lags)
    return {
        "media": float(resid.mean()),
        "varianza": float(resid.var(ddof=1)) if len(resid) > 1 else np.nan,
        "desviacion_estandar": float(resid.std(ddof=1)) if len(resid) > 1 else np.nan,
        "autocorrelacion_orden_1": float(resid.autocorr(lag=1)) if len(resid) > 2 else np.nan,
        "durbin_watson": float(durbin_watson(resid)),
        "jarque_bera": float(jb.statistic),
        "p_valor_jarque_bera": float(jb.pvalue),
        "acf": acfs.get("acf", []),
        "pacf": acfs.get("pacf", []),
    }
