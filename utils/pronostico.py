import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODO_EVALUACION = "Evaluar capacidad predictiva del modelo"
MODO_VALORES_NO_OBSERVADOS = "Generar pronóstico de valores no observados"

def descripcion_modo_pronostico(modo: str) -> dict:
    if modo == MODO_EVALUACION:
        return {
            "modo_pronostico": modo,
            "uso_modo_pronostico": (
                "Se reservan los últimos N datos con variable objetivo conocida para evaluar qué tan bien "
                "el modelo pronostica observaciones que no usó durante el entrenamiento."
            ),
            "implicacion_modo_pronostico": (
                "Este modo permite calcular errores de pronóstico, como MAE, RMSE y MAPE. "
                "Sirve para validar desempeño fuera de muestra; no debe confundirse con un pronóstico empresarial futuro."
            ),
        }
    return {
        "modo_pronostico": modo,
        "uso_modo_pronostico": (
            "El modelo se entrena con filas donde la variable objetivo tiene datos observados y luego estima "
            "la variable objetivo en filas donde Y está vacía o pendiente de observar."
        ),
        "implicacion_modo_pronostico": (
            "Este modo entrega valores pronosticados, pero no permite medir el error real de pronóstico porque "
            "todavía no existe Y observada. Debe interpretarse como una estimación pedagógica o preliminar."
        ),
    }

def metricas_pronostico(y_real, y_pred) -> dict:
    y_real = pd.Series(y_real).astype(float)
    y_pred = pd.Series(y_pred).astype(float)
    mask = y_real.notna() & y_pred.notna()
    y_real = y_real[mask]
    y_pred = y_pred[mask]

    if len(y_real) == 0:
        return {}

    mae = mean_absolute_error(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    rmse = float(np.sqrt(mse))

    mask_mape = y_real != 0
    mape = None
    if mask_mape.any():
        mape = float((np.abs((y_real[mask_mape] - y_pred[mask_mape]) / y_real[mask_mape]).mean()) * 100)

    return {
        "MAE_pronostico": float(mae),
        "MSE_pronostico": float(mse),
        "RMSE_pronostico": float(rmse),
        "MAPE_pronostico_porcentaje": mape,
        "U_Theil_pronostico": theil_u(y_real, y_pred),
        "n_pronostico_evaluable": int(len(y_real)),
    }

def agregar_errores_pronostico(df: pd.DataFrame, real_col: str, pred_col: str) -> pd.DataFrame:
    work = df.copy()
    if real_col in work.columns and pred_col in work.columns:
        work["error_pronostico"] = work[real_col] - work[pred_col]
        work["error_absoluto"] = work["error_pronostico"].abs()
        work["error_porcentual"] = np.where(
            work[real_col].notna() & (work[real_col] != 0),
            work["error_pronostico"] / work[real_col] * 100,
            np.nan,
        )
    return work


def theil_u(y_real, y_pred) -> float | None:
    """
    U de Theil tipo U2. Compara el RMSE del modelo contra un pronóstico ingenuo y_t_hat = y_{t-1}.
    Valores menores que 1 sugieren que el modelo supera al pronóstico ingenuo.
    """
    y_real = pd.Series(y_real).astype(float).reset_index(drop=True)
    y_pred = pd.Series(y_pred).astype(float).reset_index(drop=True)
    mask = y_real.notna() & y_pred.notna()
    y_real = y_real[mask].reset_index(drop=True)
    y_pred = y_pred[mask].reset_index(drop=True)
    if len(y_real) < 2:
        return None
    rmse_modelo = float(np.sqrt(np.mean((y_real - y_pred) ** 2)))
    rmse_ingenuo = float(np.sqrt(np.mean((y_real.iloc[1:].values - y_real.iloc[:-1].values) ** 2)))
    if rmse_ingenuo == 0:
        return None
    return float(rmse_modelo / rmse_ingenuo)
