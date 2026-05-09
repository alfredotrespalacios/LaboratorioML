from datetime import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st

from utils.carga_datos import cargar_datos_modulo, detectar_columnas_fecha, detectar_columnas_numericas
from utils.diagnosticos import diagnostico_residuales
from utils.exportacion import preparar_excel_generico, preparar_json_descarga
from utils.graficos import (
    grafico_acf_pacf_interactivo,
    grafico_coeficientes,
    grafico_histograma_interactivo,
    grafico_lineas_modelo,
    grafico_qq,
    grafico_real_vs_predicho,
    grafico_residuales,
    grafico_residuales_vs_ajustados,
)
from utils.pdf_reportes import crear_pdf_modulo
from utils.pronostico import (
    MODO_EVALUACION,
    MODO_VALORES_NO_OBSERVADOS,
    agregar_errores_pronostico,
    descripcion_modo_pronostico,
    metricas_pronostico,
)
from utils.transformaciones import aplicar_log_si_corresponde, ordenar_por_fecha
from utils.variables import capturar_descripcion_variables, descripcion_variables_df
from utils.estilos import mostrar_encabezado, caja_pedagogica

st.set_page_config(page_title="Regresión lineal", page_icon="📉", layout="wide")
mostrar_encabezado("📉 Regresión lineal", "OLS con statsmodels, especificación del modelo, calibración y pronóstico.")

caja_pedagogica(
    "El modo de pronóstico permite diferenciar entre evaluar capacidad predictiva con datos conocidos "
    "y generar pronósticos para valores no observados de la variable objetivo."
)

df, nombre_archivo, fuente_datos = cargar_datos_modulo("data/datos_regresion_lineal.xlsx", "regresion_lineal")
filas_originales = len(df)

st.subheader("Vista previa de los datos seleccionados")
st.caption("Esta vista corresponde al archivo realmente cargado: datos por defecto o archivo subido por el usuario.")
st.dataframe(df.head(20), use_container_width=True)

num_cols = detectar_columnas_numericas(df)
if len(num_cols) < 2:
    st.error("Se requieren al menos dos columnas numéricas.")
    st.stop()

st.sidebar.header("2. Variables")
y_col = st.sidebar.selectbox("Variable dependiente Y", num_cols, index=0)
x_cols = st.sidebar.multiselect("Variables explicativas X", [c for c in num_cols if c != y_col], default=[c for c in num_cols if c != y_col][:3])
if not x_cols:
    st.warning("Seleccione al menos una variable explicativa.")
    st.stop()

st.sidebar.header("3. Orden y modo de pronóstico")
columnas_fecha = detectar_columnas_fecha(df)
columna_fecha = st.sidebar.selectbox("Columna de fecha u orden", ["Ninguna"] + columnas_fecha)

modo_pronostico = st.sidebar.radio(
    "Modo de pronóstico",
    [MODO_EVALUACION, MODO_VALORES_NO_OBSERVADOS],
    help="Seleccione si desea evaluar el desempeño fuera de muestra o generar pronósticos para filas donde Y está vacía."
)
info_modo = descripcion_modo_pronostico(modo_pronostico)

if modo_pronostico == MODO_EVALUACION:
    max_pronostico = max(0, len(df) - 10)
    n_pronostico = st.sidebar.number_input(
        "Número de datos finales para evaluación de pronóstico",
        min_value=0,
        max_value=max_pronostico,
        value=min(20, max_pronostico),
        step=1,
    )
else:
    n_pronostico = 0
    st.sidebar.info("Este modo usa filas donde la variable Y está vacía y las X están disponibles.")

st.info(f"**Modo seleccionado:** {info_modo['modo_pronostico']}\n\n**Uso:** {info_modo['uso_modo_pronostico']}\n\n**Implicación:** {info_modo['implicacion_modo_pronostico']}")

st.sidebar.header("4. Transformaciones")
usar_log_y = st.sidebar.radio(f"Transformación de {y_col}", ["Nivel", "ln(Y)"], horizontal=True) == "ln(Y)"
transformaciones_x = {}
for x in x_cols:
    transformaciones_x[x] = st.sidebar.radio(f"Transformación de {x}", ["Nivel", f"ln({x})"], horizontal=True, key=f"log_{x}") != "Nivel"

incluir_intercepto = st.sidebar.checkbox("Incluir intercepto", value=True)
max_lags = st.sidebar.slider("Rezagos ACF/PACF residuales", 5, 40, 20)

descripcion_variables = capturar_descripcion_variables([y_col] + x_cols, "regresion_lineal")

y_trans_name = f"ln({y_col})" if usar_log_y else y_col
x_trans_names = [f"ln({x})" if transformaciones_x.get(x, False) else x for x in x_cols]
rhs_latex = " + ".join([fr"\beta_{{{i+1}}}{name}" for i, name in enumerate(x_trans_names)])
ecuacion_latex = f"{y_trans_name} = " + (r"\beta_{0} + " if incluir_intercepto else "") + rhs_latex + r" + \varepsilon"
ecuacion_texto = f"{y_trans_name} = " + ("β₀ + " if incluir_intercepto else "") + " + ".join([f"β{i+1}·{name}" for i, name in enumerate(x_trans_names)]) + " + ε"

st.subheader("Especificación del modelo")
st.latex(ecuacion_latex)

if "resultados_regresion_lineal_actuales" not in st.session_state:
    st.session_state["resultados_regresion_lineal_actuales"] = None

if st.button("Estimar regresión lineal"):
    df_ordenado = ordenar_por_fecha(df, columna_fecha)
    datos = pd.DataFrame(index=df_ordenado.index)

    y_trans, y_nombre = aplicar_log_si_corresponde(df_ordenado, y_col, usar_log_y)
    datos[y_nombre] = y_trans
    x_nombres = []
    for x in x_cols:
        x_trans, x_nombre = aplicar_log_si_corresponde(df_ordenado, x, transformaciones_x[x])
        datos[x_nombre] = x_trans
        x_nombres.append(x_nombre)

    datos["_eje_x"] = pd.to_datetime(df_ordenado[columna_fecha], errors="coerce").astype(str) if columna_fecha != "Ninguna" else list(range(len(datos)))
    filas_antes = len(datos)
    datos = datos.replace([np.inf, -np.inf], np.nan)

    if modo_pronostico == MODO_EVALUACION:
        datos_validos = datos.dropna(subset=[y_nombre] + x_nombres).copy()
        if n_pronostico >= len(datos_validos) - len(x_nombres) - 3:
            st.error("El número de datos para pronóstico es demasiado alto para la cantidad de observaciones válidas.")
            st.stop()
        datos_cal = datos_validos.iloc[:-int(n_pronostico)].copy() if int(n_pronostico) > 0 else datos_validos.copy()
        datos_fore = datos_validos.iloc[-int(n_pronostico):].copy() if int(n_pronostico) > 0 else pd.DataFrame(columns=datos.columns)
    else:
        datos_cal = datos.dropna(subset=[y_nombre] + x_nombres).copy()
        datos_fore = datos[datos[y_nombre].isna()].dropna(subset=x_nombres).copy()
        if datos_fore.empty:
            st.warning("No se encontraron filas con Y vacía y variables X completas para pronosticar valores no observados.")

    if len(datos_cal) < len(x_nombres) + 3:
        st.error("No hay suficientes observaciones para calibrar el modelo.")
        st.stop()

    y_cal = datos_cal[y_nombre]
    X_cal = datos_cal[x_nombres]
    if incluir_intercepto:
        X_cal = sm.add_constant(X_cal, has_constant="add")

    modelo = sm.OLS(y_cal, X_cal).fit()
    y_hat_cal = modelo.fittedvalues
    resid = modelo.resid

    pronostico_df = pd.DataFrame()
    if not datos_fore.empty:
        X_fore = datos_fore[x_nombres]
        if incluir_intercepto:
            X_fore = sm.add_constant(X_fore, has_constant="add")
        y_fore_hat = modelo.predict(X_fore)

        pronostico_df = pd.DataFrame({
            "eje_x": datos_fore["_eje_x"].values,
            "y_real": datos_fore[y_nombre].values if y_nombre in datos_fore.columns else np.nan,
            "y_pronosticada": y_fore_hat.values,
            "periodo": "evaluacion_pronostico" if modo_pronostico == MODO_EVALUACION else "valor_no_observado",
        })
        if modo_pronostico == MODO_EVALUACION:
            pronostico_df = agregar_errores_pronostico(pronostico_df, "y_real", "y_pronosticada")

    metricas_fore = metricas_pronostico(pronostico_df.get("y_real", []), pronostico_df.get("y_pronosticada", [])) if modo_pronostico == MODO_EVALUACION and not pronostico_df.empty else {}

    coef_table = pd.DataFrame({
        "variable": modelo.params.index,
        "coeficiente": modelo.params.values,
        "error_estandar": modelo.bse.values,
        "estadistico_t": modelo.tvalues.values,
        "p_valor": modelo.pvalues.values,
        "ic_inf": modelo.conf_int()[0].values,
        "ic_sup": modelo.conf_int()[1].values,
    })

    diag = diagnostico_residuales(resid, max_lags=max_lags)
    diag_resumen = {k: v for k, v in diag.items() if k not in ["acf", "pacf"]}

    p_jb = diag_resumen.get("p_valor_jarque_bera")
    conclusion_jb = "No disponible"
    if p_jb is not None:
        conclusion_jb = "Se rechaza normalidad al 5%" if p_jb < 0.05 else "No se rechaza normalidad al 5%"

    resumen_modelo = {
        "variable_dependiente": y_nombre,
        "n_observaciones_calibracion": int(modelo.nobs),
        "n_observaciones_pronostico": int(len(datos_fore)),
        "r2": float(modelo.rsquared),
        "r2_ajustado": float(modelo.rsquared_adj),
        "f_statistic": float(modelo.fvalue) if modelo.fvalue is not None else None,
        "prob_f_statistic": float(modelo.f_pvalue) if modelo.f_pvalue is not None else None,
        "aic": float(modelo.aic),
        "bic": float(modelo.bic),
    }

    resumen_visual_df = pd.DataFrame([
        {"indicador": "Variable dependiente", "valor": y_nombre, "interpretacion": "Variable explicada del modelo."},
        {"indicador": "Modo de pronóstico", "valor": info_modo["modo_pronostico"], "interpretacion": info_modo["uso_modo_pronostico"]},
        {"indicador": "Implicación del modo", "valor": info_modo["implicacion_modo_pronostico"], "interpretacion": "Alcance metodológico del pronóstico."},
        {"indicador": "Observaciones de calibración", "valor": int(modelo.nobs), "interpretacion": "Datos usados para estimar los coeficientes."},
        {"indicador": "Observaciones de pronóstico", "valor": int(len(datos_fore)), "interpretacion": "Datos usados para evaluar o generar pronóstico."},
        {"indicador": "R²", "valor": float(modelo.rsquared), "interpretacion": "Proporción de variabilidad explicada."},
        {"indicador": "R² ajustado", "valor": float(modelo.rsquared_adj), "interpretacion": "R² ajustado por número de variables."},
        {"indicador": "Prob(F-statistic)", "valor": float(modelo.f_pvalue) if modelo.f_pvalue is not None else None, "interpretacion": "Significancia global del modelo."},
    ])

    diagnostico_visual_df = pd.DataFrame([
        {"indicador": "Jarque-Bera", "valor": diag_resumen.get("jarque_bera"), "interpretacion": "Prueba de normalidad de residuales."},
        {"indicador": "p-valor Jarque-Bera", "valor": diag_resumen.get("p_valor_jarque_bera"), "interpretacion": conclusion_jb},
        {"indicador": "Durbin-Watson", "valor": diag_resumen.get("durbin_watson"), "interpretacion": "Valores cercanos a 2 sugieren baja autocorrelación."},
        {"indicador": "Media residuales", "valor": diag_resumen.get("media"), "interpretacion": "Promedio de errores."},
        {"indicador": "Varianza residuales", "valor": diag_resumen.get("varianza"), "interpretacion": "Dispersión de errores."},
    ])

    calibracion_df = pd.DataFrame({
        "eje_x": datos_cal["_eje_x"].values,
        "y_observada": y_cal.values,
        "y_estimada": y_hat_cal.values,
        "residual": resid.values,
        "periodo": "calibracion",
    })

    serie_modelo_df = calibracion_df[["eje_x", "y_observada", "y_estimada"]].copy()
    serie_modelo_df["y_pronosticada"] = np.nan
    if not pronostico_df.empty:
        tmp = pd.DataFrame({
            "eje_x": pronostico_df["eje_x"],
            "y_observada": pronostico_df["y_real"] if "y_real" in pronostico_df.columns else np.nan,
            "y_estimada": np.nan,
            "y_pronosticada": pronostico_df["y_pronosticada"],
        })
        serie_modelo_df = pd.concat([serie_modelo_df, tmp], ignore_index=True)

    st.session_state["resultados_regresion_lineal_actuales"] = {
        "tipo_analisis": "Regresión lineal",
        "fecha_ejecucion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "archivo_usado": nombre_archivo,
        "fuente_datos": fuente_datos,
        "filas_originales": filas_originales,
        "filas_validas": int(datos.dropna(subset=x_nombres).shape[0]),
        "filas_calibracion": len(datos_cal),
        "filas_pronostico": len(datos_fore),
        "columna_fecha": columna_fecha,
        "variable_dependiente_original": y_col,
        "variable_dependiente_usada": y_nombre,
        "variables_explicativas_originales": x_cols,
        "variables_explicativas_usadas": x_nombres,
        "descripcion_variables": descripcion_variables,
        "ecuacion_especificacion": ecuacion_texto,
        **info_modo,
        "n_pronostico": int(n_pronostico) if modo_pronostico == MODO_EVALUACION else None,
        "metricas_pronostico": metricas_fore,
        "resumen_modelo": resumen_modelo,
        "resumen_visual": resumen_visual_df.to_dict(orient="records"),
        "coeficientes": coef_table.to_dict(orient="records"),
        "diagnostico_residuales": diag_resumen,
        "diagnostico_visual": diagnostico_visual_df.to_dict(orient="records"),
        "acf_residuales": diag.get("acf", []),
        "pacf_residuales": diag.get("pacf", []),
        "reporte_statsmodels": modelo.summary().as_text(),
        "calibracion": calibracion_df.to_dict(orient="records"),
        "pronostico": pronostico_df.to_dict(orient="records"),
        "serie_modelo": serie_modelo_df.to_dict(orient="records"),
    }

res = st.session_state["resultados_regresion_lineal_actuales"]
if res:
    st.subheader("Modo de pronóstico usado")
    st.info(f"**{res.get('modo_pronostico')}**\n\n**Uso:** {res.get('uso_modo_pronostico')}\n\n**Implicación:** {res.get('implicacion_modo_pronostico')}")

    st.subheader("Especificación estimada")
    st.write(res["ecuacion_especificacion"])

    resumen_df = pd.DataFrame([res["resumen_modelo"]])
    resumen_visual_df = pd.DataFrame(res.get("resumen_visual", []))
    coef_df = pd.DataFrame(res["coeficientes"])
    diag_df = pd.DataFrame([res["diagnostico_residuales"]])
    diagnostico_visual_df = pd.DataFrame(res.get("diagnostico_visual", []))
    metricas_pronostico_df = pd.DataFrame([res.get("metricas_pronostico", {})])
    cal_df = pd.DataFrame(res["calibracion"])
    pron_df = pd.DataFrame(res["pronostico"])
    serie_df = pd.DataFrame(res["serie_modelo"])
    desc_df = descripcion_variables_df(res.get("descripcion_variables", {}))

    st.header("Reporte visual organizado del modelo lineal")
    st.subheader("Resumen del modelo")
    st.dataframe(resumen_visual_df, use_container_width=True)

    cols = st.columns(4)
    cols[0].metric("R²", f"{res['resumen_modelo'].get('r2', 0):.4f}")
    cols[1].metric("R² ajustado", f"{res['resumen_modelo'].get('r2_ajustado', 0):.4f}")
    cols[2].metric("Observ. calibración", f"{res.get('filas_calibracion', 0)}")
    cols[3].metric("Observ. pronóstico", f"{res.get('filas_pronostico', 0)}")

    st.subheader("Coeficientes estimados")
    st.dataframe(coef_df, use_container_width=True)
    st.plotly_chart(grafico_coeficientes(coef_df), use_container_width=True)

    st.subheader("Diagnóstico de residuales")
    st.dataframe(diagnostico_visual_df, use_container_width=True)

    p_jb = res["diagnostico_residuales"].get("p_valor_jarque_bera")
    if p_jb is not None:
        if p_jb < 0.05:
            st.warning("Conclusión: se rechaza normalidad de residuales al 5% según Jarque-Bera.")
        else:
            st.success("Conclusión: no se rechaza normalidad de residuales al 5% según Jarque-Bera.")

    st.header("Reporte completo de statsmodels")
    st.caption("Se conserva el reporte original de statsmodels como anexo técnico. También se exporta al Excel en la hoja Reporte_OLS.")
    with st.expander("Ver reporte completo OLS en formato original", expanded=True):
        st.code(res["reporte_statsmodels"], language="text")

    st.subheader("Variable explicada histórica y resultado del modelo en calibración")
    st.plotly_chart(grafico_lineas_modelo(cal_df, "eje_x", ["y_observada", "y_estimada"], "Variable explicada observada vs. estimada en calibración"), use_container_width=True)

    st.subheader("Pronóstico")
    if pron_df.empty:
        st.info("No hay filas disponibles para pronóstico según el modo seleccionado.")
    else:
        st.dataframe(pron_df, use_container_width=True)
        if res.get("metricas_pronostico"):
            st.write("**Métricas de evaluación de pronóstico**")
            st.dataframe(metricas_pronostico_df, use_container_width=True)
        st.plotly_chart(grafico_lineas_modelo(serie_df, "eje_x", ["y_observada", "y_estimada", "y_pronosticada"], "Calibración y pronóstico de la variable explicada"), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(grafico_acf_pacf_interactivo(res["acf_residuales"], "ACF de residuales", len(cal_df)), use_container_width=True)
    with c4:
        st.plotly_chart(grafico_acf_pacf_interactivo(res["pacf_residuales"], "PACF de residuales", len(cal_df)), use_container_width=True)

    st.subheader("Gráficos adicionales de diagnóstico")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(grafico_real_vs_predicho(cal_df["y_observada"], cal_df["y_estimada"], "Y observada vs. Y estimada"), use_container_width=True)
        st.plotly_chart(grafico_residuales(cal_df["residual"], "Residuales"), use_container_width=True)
        st.plotly_chart(grafico_histograma_interactivo(cal_df["residual"], "Histograma de residuales"), use_container_width=True)
    with c2:
        st.plotly_chart(grafico_residuales_vs_ajustados(cal_df["y_estimada"], cal_df["residual"]), use_container_width=True)
        st.plotly_chart(grafico_qq(cal_df["residual"]), use_container_width=True)

    if not desc_df.empty:
        st.subheader("Descripción de variables")
        st.dataframe(desc_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Gestión de resultados")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        if st.button("Guardar resultados para resumen ejecutivo"):
            st.session_state["resultados_regresion_lineal"] = {**res, "fecha_guardado": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            st.success("Resultados guardados.")
    with c2:
        excel = preparar_excel_generico(res, {
            "Descripcion_variables": desc_df,
            "Resumen_visual": resumen_visual_df,
            "Resumen_modelo": resumen_df,
            "Modo_pronostico": pd.DataFrame([{
                "modo_pronostico": res.get("modo_pronostico"),
                "uso_modo_pronostico": res.get("uso_modo_pronostico"),
                "implicacion_modo_pronostico": res.get("implicacion_modo_pronostico"),
            }]),
            "Metricas_pronostico": metricas_pronostico_df,
            "Coeficientes": coef_df,
            "Diagnostico_visual": diagnostico_visual_df,
            "Diagnostico": diag_df,
            "Calibracion": cal_df,
            "Pronostico": pron_df,
            "Serie_modelo": serie_df,
            "ACF_residuales": pd.DataFrame(res["acf_residuales"]),
            "PACF_residuales": pd.DataFrame(res["pacf_residuales"]),
            "Reporte_OLS": pd.DataFrame({"reporte": res["reporte_statsmodels"].splitlines()}),
        })
        st.download_button("Descargar Excel", excel, "resultados_regresion_lineal.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with c3:
        st.download_button("Descargar JSON", preparar_json_descarga(res), "resultados_regresion_lineal.json", "application/json")
    with c4:
        pdf_bytes = crear_pdf_modulo(
            "Informe del módulo: Regresión lineal",
            res,
            tablas=[
                ("Descripción de variables", desc_df),
                ("Modo de pronóstico", pd.DataFrame([{
                    "modo_pronostico": res.get("modo_pronostico"),
                    "uso_modo_pronostico": res.get("uso_modo_pronostico"),
                    "implicacion_modo_pronostico": res.get("implicacion_modo_pronostico"),
                }])),
                ("Resumen visual", resumen_visual_df),
                ("Métricas de pronóstico", metricas_pronostico_df),
                ("Coeficientes", coef_df),
                ("Diagnóstico visual", diagnostico_visual_df),
                ("Calibración", cal_df),
                ("Pronóstico", pron_df),
            ],
            figuras=[
                ("Variable explicada observada vs. estimada", grafico_lineas_modelo(cal_df, "eje_x", ["y_observada", "y_estimada"], "Variable explicada observada vs. estimada"), "Compara la variable explicada histórica con el valor estimado por el modelo en calibración."),
                ("Calibración y pronóstico", grafico_lineas_modelo(serie_df, "eje_x", ["y_observada", "y_estimada", "y_pronosticada"], "Calibración y pronóstico"), "Muestra calibración y pronóstico según el modo seleccionado."),
            ],
            notas=[res.get("uso_modo_pronostico", ""), res.get("implicacion_modo_pronostico", "")]
        )
        st.download_button("Descargar PDF", pdf_bytes, "informe_regresion_lineal.pdf", "application/pdf")
    with c5:
        if st.button("Eliminar resultados guardados"):
            st.session_state.pop("resultados_regresion_lineal", None)
            st.success("Resultados eliminados.")
