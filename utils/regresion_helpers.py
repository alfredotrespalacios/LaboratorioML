import numpy as np
import pandas as pd

from utils.pronostico import metricas_pronostico, theil_u


def preparar_tabla_doble_escala(df_pred: pd.DataFrame, usar_log_y: bool, tipo: str = "prediccion") -> pd.DataFrame:
    """
    If Y was modeled in logs, adds original-level columns by exponential back-transformation.
    tipo='prediccion' expects y_real/y_predicho.
    tipo='pronostico' expects y_real/y_pronosticada.
    """
    work = df_pred.copy()
    if not usar_log_y:
        return work

    if "y_real" in work.columns:
        work["y_real_nivel"] = np.exp(work["y_real"])

    pred_col = "y_predicho" if tipo == "prediccion" else "y_pronosticada"
    if pred_col in work.columns:
        work[f"{pred_col}_nivel"] = np.exp(work[pred_col])

    if tipo == "prediccion" and "y_real_nivel" in work.columns and "y_predicho_nivel" in work.columns:
        work["residual_nivel"] = work["y_real_nivel"] - work["y_predicho_nivel"]

    if tipo == "pronostico" and "y_real_nivel" in work.columns and "y_pronosticada_nivel" in work.columns:
        work["error_pronostico_nivel"] = work["y_real_nivel"] - work["y_pronosticada_nivel"]
        work["error_absoluto_nivel"] = work["error_pronostico_nivel"].abs()
        work["error_porcentual_nivel"] = np.where(
            work["y_real_nivel"].notna() & (work["y_real_nivel"] != 0),
            work["error_pronostico_nivel"] / work["y_real_nivel"] * 100,
            np.nan,
        )

    return work


def metricas_regresion_ext(y_true, y_pred) -> dict:
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred).astype(float)
    mask = y_true.notna() & y_pred.notna()
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {}

    err = y_true - y_pred
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = None if ss_tot == 0 else float(1 - ss_res / ss_tot)

    mask_mape = y_true != 0
    mape = None
    if mask_mape.any():
        mape = float((np.abs(err[mask_mape] / y_true[mask_mape]).mean()) * 100)

    return {
        "R2": r2,
        "Error_medio": float(err.mean()),
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE_porcentaje": mape,
        "U_Theil": theil_u(y_true, y_pred),
        "n_observaciones": int(len(y_true)),
    }


def metricas_pronostico_ext(y_real, y_pred) -> dict:
    base = metricas_pronostico(y_real, y_pred)
    return base


def tabla_modo_pronostico(res: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "modo_pronostico": res.get("modo_pronostico"),
        "uso_modo_pronostico": res.get("uso_modo_pronostico"),
        "implicacion_modo_pronostico": res.get("implicacion_modo_pronostico"),
    }])


def interpretacion_u_theil(valor):
    if valor is None or pd.isna(valor):
        return "No disponible."
    if valor < 1:
        return "U de Theil menor que 1: el modelo supera al pronóstico ingenuo."
    if valor > 1:
        return "U de Theil mayor que 1: el modelo tiene peor desempeño que el pronóstico ingenuo."
    return "U de Theil igual a 1: desempeño similar al pronóstico ingenuo."


def tabla_interpretacion_error_medio(valor, nombre="Error medio"):
    if valor is None or pd.isna(valor):
        interp = "No disponible."
    elif valor > 0:
        interp = "Valor positivo: el modelo subestima en promedio, porque Y real supera a Y pronosticada."
    elif valor < 0:
        interp = "Valor negativo: el modelo sobrestima en promedio, porque Y pronosticada supera a Y real."
    else:
        interp = "Valor cercano a cero: no se observa sesgo promedio en el pronóstico."
    return pd.DataFrame([{"indicador": nombre, "valor": valor, "interpretacion": interp}])
