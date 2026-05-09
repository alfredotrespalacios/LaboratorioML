from io import BytesIO
import json
import re

import pandas as pd

# Excel XML does not allow several ASCII control characters.
_ILLEGAL_XML_CHARS = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")
_MAX_EXCEL_CELL_CHARS = 32000

# Heavy fields should not go to the metadata sheet.
_EXCLUDED_METADATA_KEYS = {
    "reporte_statsmodels",
    "prompt_resumen_ejecutivo",
}


def limpiar_valor_excel(valor):
    """
    Limpia valores antes de escribirlos en Excel para evitar avisos de recuperación
    por caracteres XML no válidos o celdas excesivamente largas.
    """
    if valor is None:
        return ""

    try:
        if pd.isna(valor):
            return ""
    except Exception:
        pass

    if isinstance(valor, (int, float, bool)):
        return valor

    texto = str(valor)
    texto = _ILLEGAL_XML_CHARS.sub("", texto)

    if len(texto) > _MAX_EXCEL_CELL_CHARS:
        texto = texto[:_MAX_EXCEL_CELL_CHARS] + "..."

    return texto


def limpiar_dataframe_excel(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()

    limpio = df.copy()

    for col in limpio.columns:
        if limpio[col].dtype == "object":
            limpio[col] = limpio[col].map(limpiar_valor_excel)

    return limpio


def preparar_json_descarga(resultados: dict) -> str:
    return json.dumps(resultados, ensure_ascii=False, indent=2, default=str)


def _write_meta(writer, resultados: dict):
    metadatos = {}

    for k, v in resultados.items():
        if k in _EXCLUDED_METADATA_KEYS:
            continue

        # Lists and dicts are exported in dedicated sheets when needed.
        if isinstance(v, (list, dict)):
            continue

        metadatos[k] = limpiar_valor_excel(v)

    pd.DataFrame(
        [{"campo": k, "valor": v} for k, v in metadatos.items()]
    ).to_excel(writer, sheet_name="Metadatos", index=False)


def preparar_excel_generico(resultados: dict, hojas: dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        _write_meta(writer, resultados)

        for nombre, df in hojas.items():
            if df is not None:
                safe = nombre[:31]
                limpiar_dataframe_excel(df).to_excel(writer, sheet_name=safe, index=False)

    return output.getvalue()


def preparar_excel_descarga_descriptiva(
    resultados: dict,
    tabla_estadisticos: pd.DataFrame,
    tabla_percentiles: pd.DataFrame,
    tabla_pruebas: pd.DataFrame,
    tabla_acf: pd.DataFrame | None = None,
    tabla_pacf: pd.DataFrame | None = None,
) -> bytes:
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        metadatos = {
            "tipo_analisis": resultados.get("tipo_analisis"),
            "fecha_ejecucion": resultados.get("fecha_ejecucion"),
            "fecha_guardado": resultados.get("fecha_guardado"),
            "archivo_usado": resultados.get("archivo_usado"),
            "fuente_datos": resultados.get("fuente_datos"),
            "filas_originales": resultados.get("filas_originales"),
            "filas_utilizadas": resultados.get("filas_utilizadas"),
            "columna_fecha": resultados.get("columna_fecha"),
            "formula_rendimiento_logaritmico": resultados.get("formula_rendimiento_logaritmico"),
        }

        pd.DataFrame(
            [{"campo": k, "valor": limpiar_valor_excel(v)} for k, v in metadatos.items()]
        ).to_excel(writer, sheet_name="Metadatos", index=False)

        pd.DataFrame(
            {"variables_analizadas": resultados.get("variables_analizadas", [])}
        ).to_excel(writer, sheet_name="Variables", index=False)

        if resultados.get("descripcion_variables"):
            pd.DataFrame(
                [{"variable": k, "descripcion": limpiar_valor_excel(v)} for k, v in resultados.get("descripcion_variables", {}).items()]
            ).to_excel(writer, sheet_name="Descripcion_variables", index=False)

        limpiar_dataframe_excel(tabla_estadisticos).to_excel(writer, sheet_name="Estadisticos", index=False)
        limpiar_dataframe_excel(tabla_percentiles).to_excel(writer, sheet_name="Percentiles", index=False)
        limpiar_dataframe_excel(tabla_pruebas).to_excel(writer, sheet_name="Pruebas", index=False)

        if tabla_acf is not None and not tabla_acf.empty:
            limpiar_dataframe_excel(tabla_acf).to_excel(writer, sheet_name="ACF", index=False)

        if tabla_pacf is not None and not tabla_pacf.empty:
            limpiar_dataframe_excel(tabla_pacf).to_excel(writer, sheet_name="PACF", index=False)

    return output.getvalue()
