from io import BytesIO
import json
import pandas as pd

def preparar_json_descarga(resultados: dict) -> str:
    return json.dumps(resultados, ensure_ascii=False, indent=2, default=str)

def _write_meta(writer, resultados: dict):
    metadatos = {k: v for k, v in resultados.items() if not isinstance(v, (list, dict))}
    pd.DataFrame([{"campo": k, "valor": str(v)} for k, v in metadatos.items()]).to_excel(writer, sheet_name="Metadatos", index=False)

def preparar_excel_generico(resultados: dict, hojas: dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        _write_meta(writer, resultados)
        for nombre, df in hojas.items():
            if df is not None:
                safe = nombre[:31]
                df.to_excel(writer, sheet_name=safe, index=False)
    return output.getvalue()

def preparar_excel_descarga_descriptiva(
    resultados: dict,
    tabla_estadisticos: pd.DataFrame,
    tabla_percentiles: pd.DataFrame,
    tabla_pruebas: pd.DataFrame,
    tabla_acf: pd.DataFrame | None = None,
    tabla_pacf: pd.DataFrame | None = None,
) -> bytes:
    hojas = {
        "Estadisticos": tabla_estadisticos,
        "Percentiles": tabla_percentiles,
        "Pruebas": tabla_pruebas,
        "ACF": tabla_acf,
        "PACF": tabla_pacf,
    }
    return preparar_excel_generico(resultados, hojas)
