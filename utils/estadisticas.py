import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from statsmodels.tsa.stattools import adfuller, acf, pacf

PERCENTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]

def interpretar_p_valor_normalidad(p_valor: float, alpha: float = 0.05) -> str:
    if np.isnan(p_valor):
        return "No disponible"
    if p_valor < alpha:
        return f"Se rechaza normalidad al {alpha:.0%}."
    return f"No se rechaza normalidad al {alpha:.0%}."

def interpretar_p_valor_adf(p_valor: float, alpha: float = 0.05) -> str:
    if np.isnan(p_valor):
        return "No disponible"
    if p_valor < alpha:
        return f"Se rechaza raíz unitaria al {alpha:.0%}; hay evidencia de estacionariedad."
    return f"No se rechaza raíz unitaria al {alpha:.0%}; la serie podría ser no estacionaria."

def prueba_jarque_bera(serie: pd.Series) -> dict:
    x = pd.to_numeric(serie, errors="coerce").dropna()

    if len(x) < 3:
        return {
            "jarque_bera": np.nan,
            "p_valor_jarque_bera": np.nan,
            "conclusion_normalidad": "No disponible: se requieren más observaciones.",
        }

    resultado = jarque_bera(x)
    estadistico = float(resultado.statistic)
    p_valor = float(resultado.pvalue)

    return {
        "jarque_bera": estadistico,
        "p_valor_jarque_bera": p_valor,
        "conclusion_normalidad": interpretar_p_valor_normalidad(p_valor),
    }

def prueba_adf(serie: pd.Series) -> dict:
    x = pd.to_numeric(serie, errors="coerce").dropna()

    if len(x) < 12 or x.nunique() <= 1:
        return {
            "adf": np.nan,
            "p_valor_adf": np.nan,
            "valores_criticos_adf": {},
            "conclusion_adf": "No disponible: se requieren más observaciones y variabilidad.",
        }

    try:
        resultado = adfuller(x, autolag="AIC")
        estadistico = float(resultado[0])
        p_valor = float(resultado[1])
        valores_criticos = {k: float(v) for k, v in resultado[4].items()}

        return {
            "adf": estadistico,
            "p_valor_adf": p_valor,
            "valores_criticos_adf": valores_criticos,
            "conclusion_adf": interpretar_p_valor_adf(p_valor),
        }
    except Exception as exc:
        return {
            "adf": np.nan,
            "p_valor_adf": np.nan,
            "valores_criticos_adf": {},
            "conclusion_adf": f"No disponible: {exc}",
        }

def estadisticos_principales(serie: pd.Series) -> dict:
    x = pd.to_numeric(serie, errors="coerce").dropna()

    if len(x) == 0:
        return {
            "n_observaciones": 0, "media": np.nan, "mediana": np.nan,
            "varianza": np.nan, "desviacion_estandar": np.nan,
            "minimo": np.nan, "maximo": np.nan, "asimetria": np.nan, "curtosis": np.nan,
        }

    return {
        "n_observaciones": int(x.shape[0]),
        "media": float(x.mean()),
        "mediana": float(x.median()),
        "varianza": float(x.var(ddof=1)) if len(x) > 1 else np.nan,
        "desviacion_estandar": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
        "minimo": float(x.min()),
        "maximo": float(x.max()),
        "asimetria": float(x.skew()) if len(x) > 2 else np.nan,
        "curtosis": float(x.kurtosis()) if len(x) > 3 else np.nan,
    }

def percentiles_serie(serie: pd.Series) -> dict:
    x = pd.to_numeric(serie, errors="coerce").dropna()

    if len(x) == 0:
        return {"p1": np.nan, "p5": np.nan, "p25": np.nan, "p50": np.nan, "p75": np.nan, "p95": np.nan, "p99": np.nan}

    cuantiles = x.quantile(PERCENTILES)

    return {
        "p1": float(cuantiles.loc[0.01]),
        "p5": float(cuantiles.loc[0.05]),
        "p25": float(cuantiles.loc[0.25]),
        "p50": float(cuantiles.loc[0.50]),
        "p75": float(cuantiles.loc[0.75]),
        "p95": float(cuantiles.loc[0.95]),
        "p99": float(cuantiles.loc[0.99]),
    }

def pruebas_estadisticas(serie: pd.Series) -> dict:
    salida = {}
    salida.update(prueba_jarque_bera(serie))
    salida.update(prueba_adf(serie))
    return salida

def calcular_acf_pacf(serie: pd.Series, max_lags: int = 20) -> dict:
    x = pd.to_numeric(serie, errors="coerce").dropna()

    if len(x) < 10 or x.nunique() <= 1:
        return {"acf": [], "pacf": [], "max_lags_usado": 0, "mensaje": "No disponible: se requieren más observaciones y variabilidad."}

    max_lags_ajustado = min(max_lags, max(1, len(x) // 2 - 1))

    try:
        acf_vals = acf(x, nlags=max_lags_ajustado, fft=False)
        pacf_vals = pacf(x, nlags=max_lags_ajustado, method="ywm")
        return {
            "acf": [{"rezago": int(i), "valor": float(v)} for i, v in enumerate(acf_vals)],
            "pacf": [{"rezago": int(i), "valor": float(v)} for i, v in enumerate(pacf_vals)],
            "max_lags_usado": int(max_lags_ajustado),
            "mensaje": "OK",
        }
    except Exception as exc:
        return {"acf": [], "pacf": [], "max_lags_usado": 0, "mensaje": f"No disponible: {exc}"}

def construir_tablas_resultados(resultados: dict):
    filas_est, filas_pct, filas_pruebas, filas_acf, filas_pacf = [], [], [], [], []

    for variable, bloques in resultados.items():
        for tipo_serie, valores in bloques.items():
            base = {"variable": variable, "tipo_serie": tipo_serie}
            filas_est.append({**base, **valores.get("estadisticos", {})})
            filas_pct.append({**base, **valores.get("percentiles", {})})

            pruebas = valores.get("pruebas", {}).copy()
            if "valores_criticos_adf" in pruebas:
                pruebas["valores_criticos_adf"] = str(pruebas["valores_criticos_adf"])
            filas_pruebas.append({**base, **pruebas})

            for item in valores.get("acf", []):
                filas_acf.append({**base, **item})
            for item in valores.get("pacf", []):
                filas_pacf.append({**base, **item})

    return pd.DataFrame(filas_est), pd.DataFrame(filas_pct), pd.DataFrame(filas_pruebas), pd.DataFrame(filas_acf), pd.DataFrame(filas_pacf)
