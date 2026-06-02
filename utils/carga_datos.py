from pathlib import Path

import pandas as pd
import streamlit as st

def cargar_datos_modulo(ruta_defecto: str, nombre_modulo: str):
    """
    Carga datos desde un Excel por defecto en data/ o desde un Excel subido por el usuario.
    El archivo subido se usa temporalmente en memoria y no se guarda permanentemente.
    """
    st.sidebar.header("1. Datos")

    opcion = st.sidebar.radio(
        "Fuente de datos",
        ["Usar datos de entrada por defecto", "Subir archivo Excel propio"],
        key=f"fuente_datos_{nombre_modulo}",
    )

    if opcion == "Usar datos de entrada por defecto":
        ruta = Path(ruta_defecto)
        if not ruta.exists():
            st.error(f"No se encontró el archivo por defecto: {ruta_defecto}")
            st.stop()

        df = pd.read_excel(ruta)
        nombre_archivo = str(ruta)
        fuente_datos = "Datos de entrada por defecto"
        return df, nombre_archivo, fuente_datos

    archivo = st.sidebar.file_uploader(
        "Suba un archivo Excel",
        type=["xlsx", "xls"],
        key=f"archivo_{nombre_modulo}",
    )

    if archivo is None:
        st.info("Suba un archivo Excel o seleccione los datos de entrada por defecto.")
        st.stop()

    try:
        df = pd.read_excel(archivo)
        nombre_archivo = archivo.name
        fuente_datos = "Archivo subido por el usuario"
        return df, nombre_archivo, fuente_datos
    except Exception as exc:
        st.error(f"No fue posible cargar el archivo: {exc}")
        st.stop()

def detectar_columnas_fecha(df: pd.DataFrame):
    columnas_fecha = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            columnas_fecha.append(col)
        else:
            try:
                pd.to_datetime(df[col], errors="raise")
                columnas_fecha.append(col)
            except Exception:
                pass
    return columnas_fecha

def detectar_columnas_numericas(df: pd.DataFrame):
    return df.select_dtypes(include=["number"]).columns.tolist()

def detectar_columnas_categoricas(df: pd.DataFrame):
    return df.select_dtypes(exclude=["number"]).columns.tolist()
