from datetime import datetime

import pandas as pd
import streamlit as st

from utils.carga_datos import cargar_datos_modulo, detectar_columnas_fecha, detectar_columnas_numericas
from utils.estadisticas import estadisticos_principales, percentiles_serie, pruebas_estadisticas, calcular_acf_pacf, construir_tablas_resultados
from utils.exportacion import preparar_excel_descarga_descriptiva, preparar_json_descarga
from utils.pdf_reportes import crear_pdf_modulo
from utils.graficos import grafico_acf_pacf_interactivo, grafico_boxplot_interactivo, grafico_histograma_interactivo, grafico_matriz_correlacion_interactiva, grafico_serie_interactivo
from utils.transformaciones import ordenar_por_fecha, rendimiento_logaritmico
from utils.estilos import mostrar_encabezado, caja_pedagogica

st.set_page_config(page_title="Estadística descriptiva", page_icon="📈", layout="wide")
mostrar_encabezado("📈 Estadística descriptiva", "Analiza una o varias variables en nivel y sus rendimientos logarítmicos.")

caja_pedagogica(
    "Resultados separados en tres tablas: estadísticos principales, percentiles y pruebas estadísticas. "
    "Jarque-Bera evalúa normalidad; ADF evalúa raíz unitaria. ACF y PACF se muestran como figuras."
)

df, nombre_archivo, fuente_datos = cargar_datos_modulo("data/datos_descriptiva.xlsx", "estadistica_descriptiva")
filas_originales = len(df)

st.subheader("Vista previa de los datos")
st.dataframe(df.head(20), use_container_width=True)

st.sidebar.header("2. Configuración")
columnas_fecha = detectar_columnas_fecha(df)
columna_fecha = st.sidebar.selectbox("Columna de fecha para ordenar la serie", ["Ninguna"] + columnas_fecha)
max_lags = st.sidebar.slider("Número máximo de rezagos para ACF/PACF", 5, 40, 20, 1)

df_ordenado = ordenar_por_fecha(df, columna_fecha)
columnas_numericas = detectar_columnas_numericas(df_ordenado)
if not columnas_numericas:
    st.error("No se encontraron columnas numéricas para analizar.")
    st.stop()

variables = st.sidebar.multiselect("Seleccione una o varias variables numéricas", columnas_numericas, default=columnas_numericas[: min(2, len(columnas_numericas))])
if not variables:
    st.warning("Seleccione al menos una variable.")
    st.stop()

st.write("### Variables seleccionadas")
st.write(", ".join(variables))

if "resultados_descriptiva_actuales" not in st.session_state:
    st.session_state["resultados_descriptiva_actuales"] = None

if st.button("Ejecutar análisis descriptivo"):
    resultados = {}

    for variable in variables:
        serie_original = pd.to_numeric(df_ordenado[variable], errors="coerce")
        serie_rendimiento = rendimiento_logaritmico(serie_original)
        acf_pacf_original = calcular_acf_pacf(serie_original, max_lags=max_lags)
        acf_pacf_rend = calcular_acf_pacf(serie_rendimiento, max_lags=max_lags)

        resultados[variable] = {
            "variable_original": {
                "estadisticos": estadisticos_principales(serie_original),
                "percentiles": percentiles_serie(serie_original),
                "pruebas": pruebas_estadisticas(serie_original),
                "acf": acf_pacf_original["acf"],
                "pacf": acf_pacf_original["pacf"],
            },
            "rendimiento_logaritmico": {
                "estadisticos": estadisticos_principales(serie_rendimiento),
                "percentiles": percentiles_serie(serie_rendimiento),
                "pruebas": pruebas_estadisticas(serie_rendimiento),
                "acf": acf_pacf_rend["acf"],
                "pacf": acf_pacf_rend["pacf"],
            },
        }

    tabla_est, tabla_pct, tabla_pruebas, tabla_acf, tabla_pacf = construir_tablas_resultados(resultados)

    st.session_state["resultados_descriptiva_actuales"] = {
        "tipo_analisis": "Estadística descriptiva",
        "fecha_ejecucion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "archivo_usado": nombre_archivo,
        "fuente_datos": fuente_datos,
        "filas_originales": filas_originales,
        "filas_utilizadas": len(df_ordenado),
        "columna_fecha": columna_fecha,
        "variables_analizadas": variables,
        "formula_rendimiento_logaritmico": "r_t = ln(X_t / X_{t-1})",
        "max_lags_acf_pacf": max_lags,
        "tabla_estadisticos": tabla_est.to_dict(orient="records"),
        "tabla_percentiles": tabla_pct.to_dict(orient="records"),
        "tabla_pruebas": tabla_pruebas.to_dict(orient="records"),
        "tabla_acf": tabla_acf.to_dict(orient="records"),
        "tabla_pacf": tabla_pacf.to_dict(orient="records"),
    }

if st.session_state["resultados_descriptiva_actuales"] is not None:
    resultados_actuales = st.session_state["resultados_descriptiva_actuales"]
    tabla_est = pd.DataFrame(resultados_actuales["tabla_estadisticos"])
    tabla_pct = pd.DataFrame(resultados_actuales["tabla_percentiles"])
    tabla_pruebas = pd.DataFrame(resultados_actuales["tabla_pruebas"])
    tabla_acf = pd.DataFrame(resultados_actuales["tabla_acf"])
    tabla_pacf = pd.DataFrame(resultados_actuales["tabla_pacf"])

    st.subheader("Tabla 1: Estadísticos principales")
    st.dataframe(tabla_est, use_container_width=True)

    st.subheader("Tabla 2: Percentiles")
    st.dataframe(tabla_pct, use_container_width=True)

    st.subheader("Tabla 3: Pruebas estadísticas")
    st.caption("Jarque-Bera evalúa normalidad. ADF evalúa raíz unitaria o estacionariedad.")
    st.dataframe(tabla_pruebas, use_container_width=True)

    st.subheader("Figuras interactivas por variable")
    x_col = columna_fecha if columna_fecha != "Ninguna" else None

    for variable in variables:
        with st.expander(f"Figuras de {variable}", expanded=False):
            serie_original = pd.to_numeric(df_ordenado[variable], errors="coerce")
            serie_rend = rendimiento_logaritmico(serie_original)
            df_graf = df_ordenado.copy()
            df_graf[f"r_log_{variable}"] = serie_rend

            acf_o = tabla_acf[(tabla_acf["variable"] == variable) & (tabla_acf["tipo_serie"] == "variable_original")]
            pacf_o = tabla_pacf[(tabla_pacf["variable"] == variable) & (tabla_pacf["tipo_serie"] == "variable_original")]
            acf_r = tabla_acf[(tabla_acf["variable"] == variable) & (tabla_acf["tipo_serie"] == "rendimiento_logaritmico")]
            pacf_r = tabla_pacf[(tabla_pacf["variable"] == variable) & (tabla_pacf["tipo_serie"] == "rendimiento_logaritmico")]

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Variable original**")
                st.plotly_chart(grafico_serie_interactivo(df_graf, variable, f"Serie original: {variable}", x_col), use_container_width=True)
                st.plotly_chart(grafico_histograma_interactivo(serie_original, f"Histograma: {variable}"), use_container_width=True)
                st.plotly_chart(grafico_boxplot_interactivo(serie_original, f"Boxplot: {variable}"), use_container_width=True)
                st.plotly_chart(grafico_acf_pacf_interactivo(acf_o[["rezago","valor"]].to_dict(orient="records"), f"ACF original: {variable}", len(serie_original.dropna())), use_container_width=True)
                st.plotly_chart(grafico_acf_pacf_interactivo(pacf_o[["rezago","valor"]].to_dict(orient="records"), f"PACF original: {variable}", len(serie_original.dropna())), use_container_width=True)
            with col2:
                st.write("**Rendimiento logarítmico**")
                st.plotly_chart(grafico_serie_interactivo(df_graf, f"r_log_{variable}", f"Rendimiento logarítmico: {variable}", x_col), use_container_width=True)
                st.plotly_chart(grafico_histograma_interactivo(serie_rend, f"Histograma rendimiento: {variable}"), use_container_width=True)
                st.plotly_chart(grafico_boxplot_interactivo(serie_rend, f"Boxplot rendimiento: {variable}"), use_container_width=True)
                st.plotly_chart(grafico_acf_pacf_interactivo(acf_r[["rezago","valor"]].to_dict(orient="records"), f"ACF rendimiento: {variable}", len(serie_rend.dropna())), use_container_width=True)
                st.plotly_chart(grafico_acf_pacf_interactivo(pacf_r[["rezago","valor"]].to_dict(orient="records"), f"PACF rendimiento: {variable}", len(serie_rend.dropna())), use_container_width=True)

    if len(variables) > 1:
        st.subheader("Matrices de correlación interactivas")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(grafico_matriz_correlacion_interactiva(df_ordenado[variables], "Correlación de variables originales"), use_container_width=True)
        with c2:
            rendimientos = pd.DataFrame({f"r_log_{v}": rendimiento_logaritmico(pd.to_numeric(df_ordenado[v], errors="coerce")) for v in variables})
            st.plotly_chart(grafico_matriz_correlacion_interactiva(rendimientos, "Correlación de rendimientos logarítmicos"), use_container_width=True)

    st.subheader("Interpretación preliminar")
    st.write("Revise por separado los estadísticos, percentiles y pruebas. La ACF/PACF ayuda a explorar dependencia temporal.")

    st.markdown("---")
    st.subheader("Gestión de resultados")
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        if st.button("Guardar resultados para resumen ejecutivo"):
            st.session_state["resultados_estadistica_descriptiva"] = {**resultados_actuales, "fecha_guardado": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            st.success("Resultados guardados.")

    with c2:
        excel = preparar_excel_descarga_descriptiva(resultados_actuales, tabla_est, tabla_pct, tabla_pruebas, tabla_acf, tabla_pacf)
        st.download_button("Descargar Excel", data=excel, file_name="resultados_estadistica_descriptiva.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with c3:
        st.download_button("Descargar JSON", data=preparar_json_descarga(resultados_actuales), file_name="resultados_estadistica_descriptiva.json", mime="application/json")

    with c4:
        figuras_pdf = []
        for variable in variables:
            serie_original_pdf = pd.to_numeric(df_ordenado[variable], errors="coerce")
            serie_rend_pdf = rendimiento_logaritmico(serie_original_pdf)
            df_pdf = df_ordenado.copy()
            df_pdf[f"r_log_{variable}"] = serie_rend_pdf
            acf_o_pdf = tabla_acf[(tabla_acf["variable"] == variable) & (tabla_acf["tipo_serie"] == "variable_original")]
            pacf_o_pdf = tabla_pacf[(tabla_pacf["variable"] == variable) & (tabla_pacf["tipo_serie"] == "variable_original")]
            acf_r_pdf = tabla_acf[(tabla_acf["variable"] == variable) & (tabla_acf["tipo_serie"] == "rendimiento_logaritmico")]
            pacf_r_pdf = tabla_pacf[(tabla_pacf["variable"] == variable) & (tabla_pacf["tipo_serie"] == "rendimiento_logaritmico")]
            figuras_pdf.extend([
                (f"Serie original - {variable}", grafico_serie_interactivo(df_pdf, variable, f"Serie original: {variable}", x_col), "Muestra la evolución de la variable en niveles."),
                (f"Histograma original - {variable}", grafico_histograma_interactivo(serie_original_pdf, f"Histograma: {variable}"), "Permite observar la forma de la distribución y posibles valores extremos."),
                (f"ACF original - {variable}", grafico_acf_pacf_interactivo(acf_o_pdf[["rezago","valor"]].to_dict(orient="records"), f"ACF original: {variable}", len(serie_original_pdf.dropna())), "La ACF muestra dependencia temporal por rezagos."),
                (f"PACF original - {variable}", grafico_acf_pacf_interactivo(pacf_o_pdf[["rezago","valor"]].to_dict(orient="records"), f"PACF original: {variable}", len(serie_original_pdf.dropna())), "La PACF muestra la dependencia parcial por rezagos."),
                (f"Serie rendimiento logarítmico - {variable}", grafico_serie_interactivo(df_pdf, f"r_log_{variable}", f"Rendimiento logarítmico: {variable}", x_col), "Muestra la variación logarítmica entre observaciones consecutivas."),
                (f"Histograma rendimiento - {variable}", grafico_histograma_interactivo(serie_rend_pdf, f"Histograma rendimiento: {variable}"), "Permite revisar la distribución de los rendimientos."),
                (f"ACF rendimiento - {variable}", grafico_acf_pacf_interactivo(acf_r_pdf[["rezago","valor"]].to_dict(orient="records"), f"ACF rendimiento: {variable}", len(serie_rend_pdf.dropna())), "Ayuda a revisar autocorrelación en los rendimientos."),
                (f"PACF rendimiento - {variable}", grafico_acf_pacf_interactivo(pacf_r_pdf[["rezago","valor"]].to_dict(orient="records"), f"PACF rendimiento: {variable}", len(serie_rend_pdf.dropna())), "Ayuda a revisar autocorrelación parcial en los rendimientos."),
            ])
        if len(variables) > 1:
            figuras_pdf.append(("Correlación variables originales", grafico_matriz_correlacion_interactiva(df_ordenado[variables], "Correlación de variables originales"), "Matriz de correlación entre variables en niveles."))
        pdf_bytes = crear_pdf_modulo(
            "Informe del módulo: Estadística descriptiva",
            resultados_actuales,
            tablas=[("Estadísticos principales", tabla_est), ("Percentiles", tabla_pct), ("Pruebas estadísticas", tabla_pruebas), ("ACF", tabla_acf), ("PACF", tabla_pacf)],
            figuras=figuras_pdf,
            notas=["Jarque-Bera evalúa normalidad.", "ADF evalúa raíz unitaria o estacionariedad.", "ACF y PACF se presentan como figuras para interpretar dependencia temporal."]
        )
        st.download_button("Descargar PDF", pdf_bytes, "informe_estadistica_descriptiva.pdf", "application/pdf")

    with c5:
        if st.button("Eliminar resultados guardados"):
            st.session_state.pop("resultados_estadistica_descriptiva", None)
            st.success("Resultados eliminados.")
else:
    st.info("Configure el análisis y presione **Ejecutar análisis descriptivo**.")
