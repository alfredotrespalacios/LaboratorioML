import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def ordenar_por_fecha(df: pd.DataFrame, columna_fecha: str | None) -> pd.DataFrame:
    if columna_fecha is None or columna_fecha == "Ninguna":
        return df.copy()

    df_ordenado = df.copy()
    df_ordenado[columna_fecha] = pd.to_datetime(df_ordenado[columna_fecha], errors="coerce")
    df_ordenado = df_ordenado.dropna(subset=[columna_fecha])
    df_ordenado = df_ordenado.sort_values(columna_fecha)
    return df_ordenado

def rendimiento_logaritmico(serie: pd.Series) -> pd.Series:
    serie_limpia = pd.to_numeric(serie, errors="coerce")
    serie_limpia = serie_limpia.where(serie_limpia > 0)
    return np.log(serie_limpia / serie_limpia.shift(1))

def aplicar_log_si_corresponde(df: pd.DataFrame, columna: str, usar_log: bool):
    serie = pd.to_numeric(df[columna], errors="coerce")
    nombre = f"ln_{columna}" if usar_log else columna

    if usar_log:
        serie = serie.where(serie > 0)
        return np.log(serie), nombre

    return serie, nombre

def preparar_xy(df: pd.DataFrame, y_col: str, x_cols: list[str], test_size: float, random_state: int = 42):
    datos = df[[y_col] + x_cols].copy().dropna()
    X = datos[x_cols]
    y = datos[y_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def escalar_train_test(X_train, X_test, aplicar: bool):
    if not aplicar:
        return X_train, X_test, None
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler

def codificar_y(y):
    le = LabelEncoder()
    y_cod = le.fit_transform(y.astype(str))
    return y_cod, le
