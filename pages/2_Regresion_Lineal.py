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
from utils.transformaciones import aplicar_log_si_corresponde, ordenar_por_fecha
from utils.variables import capturar_descripcion_variables, descripcion_variables_df
from utils.estilos import mostrar_encabezado, caja_pedagogica

st.set_page_config(page_title="Regresión lineal", page_icon="📉", layout="wide")
mostrar_encabezado("📉 Regresión lineal", "OLS con statsmodels, especificación del modelo, calibración y pronóstico.")

caja_pedagogica(
    "El modelo se estima con una muestra de calibración. El estudiante puede reservar los últimos datos "
    "para pronosticar la variable explicada y comparar el pronóstico con los valores observados."
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

st.sidebar.header("3. Orden y pronóstico")
columnas_fecha = detectar_columnas_fecha(df)
columna_fecha = st.sidebar.selectbox("Columna de fecha u orden", ["Ninguna"] + columnas_fecha)
max_pronostico = max(0, len(df) - 10)
n_pronostico = st.sidebar.number_input(
    "Número de datos finales para pronóstico",
    min_value=0,
    max_value=max_pronostico,
    value=min(20, max_pronostico),
    step=1,
    help="Estos datos finales no se usan para calibrar el modelo; se reservan para pronosticar la variable explicada."
)

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
st.caption("Esta es la especificación definida con las variables y transformaciones seleccionadas. La estimación aparece después de ejecutar el modelo.")

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

    if columna_fecha != "Ninguna":
        datos["_eje_x"] = pd.to_datetime(df_ordenado[columna_fecha], errors="coerce").astype(str)
    else:
        datos["_eje_x"] = list(range(len(datos)))

    filas_antes = len(datos)
    datos = datos.replace([np.inf, -np.inf], np.nan).dropna()
    filas_validas = len(datos)

    if n_pronostico >= filas_validas - len(x_nombres) - 3:
        st.error("El número de datos para pronóstico es demasiado alto para la cantidad de observaciones válidas.")
        st.stop()

    if int(n_pronostico) > 0:
        datos_cal = datos.iloc[:-int(n_pronostico)].copy()
        datos_fore = datos.iloc[-int(n_pronostico):].copy()
    else:
        datos_cal = datos.copy()
        datos_fore = pd.DataFrame(columns=datos.columns)

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
            "y_real": datos_fore[y_nombre].values,
            "y_pronosticada": y_fore_hat.values,
            "error_pronostico": datos_fore[y_nombre].values - y_fore_hat.values,
            "periodo": "pronostico",
        })

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

    resumen_modelo = {
        "variable_dependiente": y_nombre,
        "n_observaciones_calibracion": int(modelo.nobs),
        "n_observaciones_pronostico": int(len(datos_fore)),
        "r2": float(modelo.rsquared),
        "r2_ajustado": float(modelo.rsquared_adj),
        "f_statistic": float(modelo.fvalue) if modelo.fvalue is not None else None,
        "prob_f_statistic": float(modelo.f_pvalue) if modelo.f_pvalue is not None else None,
        "log_likelihood": float(modelo.llf),
        "aic": float(modelo.aic),
        "bic": float(modelo.bic),
        "df_residuales": float(modelo.df_resid),
        "df_modelo": float(modelo.df_model),
    }

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
        tmp = pronostico_df[["eje_x", "y_real", "y_pronosticada"]].rename(columns={"y_real": "y_observada"})
        tmp["y_estimada"] = np.nan
        serie_modelo_df = pd.concat([serie_modelo_df, tmp[["eje_x", "y_observada", "y_estimada", "y_pronosticada"]]], ignore_index=True)

    st.session_state["resultados_regresion_lineal_actuales"] = {
        "tipo_analisis": "Regresión lineal",
        "fecha_ejecucion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "archivo_usado": nombre_archivo,
        "fuente_datos": fuente_datos,
        "filas_originales": filas_originales,
        "filas_validas": filas_validas,
        "filas_calibracion": len(datos_cal),
        "filas_pronostico": len(datos_fore),
        "columna_fecha": columna_fecha,
        "variable_dependiente_original": y_col,
        "variable_dependiente_usada": y_nombre,
        "variables_explicativas_originales": x_cols,
        "variables_explicativas_usadas": x_nombres,
        "transformacion_y": "logaritmo_natural" if usar_log_y else "nivel",
        "transformaciones_x": {k: ("logaritmo_natural" if v else "nivel") for k, v in transformaciones_x.items()},
        "descripcion_variables": descripcion_variables,
        "ecuacion_especificacion": ecuacion_texto,
        "observaciones_eliminadas": filas_antes - filas_validas,
        "resumen_modelo": resumen_modelo,
        "coeficientes": coef_table.to_dict(orient="records"),
        "diagnostico_residuales": diag_resumen,
        "acf_residuales": diag.get("acf", []),
        "pacf_residuales": diag.get("pacf", []),
        "reporte_statsmodels": modelo.summary().as_text(),
        "calibracion": calibracion_df.to_dict(orient="records"),
        "pronostico": pronostico_df.to_dict(orient="records"),
        "serie_modelo": serie_modelo_df.to_dict(orient="records"),
    }

res = st.session_state["resultados_regresion_lineal_actuales"]
if res:
    st.subheader("Especificación estimada")
    st.write(res["ecuacion_especificacion"])

    resumen_df = pd.DataFrame([res["resumen_modelo"]])
    coef_df = pd.DataFrame(res["coeficientes"])
    diag_df = pd.DataFrame([res["diagnostico_residuales"]])
    cal_df = pd.DataFrame(res["calibracion"])
    pron_df = pd.DataFrame(res["pronostico"])
    serie_df = pd.DataFrame(res["serie_modelo"])
    desc_df = descripcion_variables_df(res.get("descripcion_variables", {}))

    st.subheader("Resumen organizado del modelo")
    st.dataframe(resumen_df, use_container_width=True)

    st.subheader("Coeficientes y significancia")
    st.dataframe(coef_df, use_container_width=True)
    st.plotly_chart(grafico_coeficientes(coef_df), use_container_width=True)

    with st.expander("Ver reporte completo de statsmodels", expanded=False):
        st.text(res["reporte_statsmodels"])

    st.subheader("Variable explicada histórica y resultado del modelo en calibración")
    st.plotly_chart(grafico_lineas_modelo(cal_df, "eje_x", ["y_observada", "y_estimada"], "Variable explicada observada vs. estimada en calibración"), use_container_width=True)
    st.caption("Esta gráfica muestra la variable explicada histórica y el valor estimado por el modelo durante la calibración.")

    st.subheader("Pronóstico con los datos finales reservados")
    if pron_df.empty:
        st.info("No se reservaron datos finales para pronóstico.")
    else:
        st.dataframe(pron_df, use_container_width=True)
        st.plotly_chart(grafico_lineas_modelo(serie_df, "eje_x", ["y_observada", "y_estimada", "y_pronosticada"], "Calibración y pronóstico de la variable explicada"), use_container_width=True)
        st.caption("La línea pronosticada corresponde a los datos finales reservados que no se usaron para calibrar el modelo.")

    st.subheader("Diagnóstico de residuales")
    st.dataframe(diag_df, use_container_width=True)

    p_jb = res["diagnostico_residuales"].get("p_valor_jarque_bera")
    if p_jb is not None:
        if p_jb < 0.05:
            st.warning("Conclusión: se rechaza normalidad de residuales al 5% según Jarque-Bera.")
        else:
            st.success("Conclusión: no se rechaza normalidad de residuales al 5% según Jarque-Bera.")

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
            "Resumen_modelo": resumen_df,
            "Coeficientes": coef_df,
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
                ("Resumen del modelo", resumen_df),
                ("Coeficientes", coef_df),
                ("Diagnóstico de residuales", diag_df),
                ("Calibración", cal_df),
                ("Pronóstico", pron_df),
                ("ACF residuales", pd.DataFrame(res["acf_residuales"])),
                ("PACF residuales", pd.DataFrame(res["pacf_residuales"])),
            ],
            figuras=[
                ("Variable explicada observada vs. estimada", grafico_lineas_modelo(cal_df, "eje_x", ["y_observada", "y_estimada"], "Variable explicada observada vs. estimada"), "Compara la variable explicada histórica con el valor estimado por el modelo en calibración."),
                ("Calibración y pronóstico", grafico_lineas_modelo(serie_df, "eje_x", ["y_observada", "y_estimada", "y_pronosticada"], "Calibración y pronóstico"), "Muestra la calibración y los datos finales pronosticados."),
                ("Y observada vs. Y estimada", grafico_real_vs_predicho(cal_df["y_observada"], cal_df["y_estimada"], "Y observada vs. Y estimada"), "Puntos cercanos a la diagonal indican mejor ajuste."),
                ("ACF de residuales", grafico_acf_pacf_interactivo(res["acf_residuales"], "ACF de residuales", len(cal_df)), "Explora autocorrelación de residuales."),
                ("PACF de residuales", grafico_acf_pacf_interactivo(res["pacf_residuales"], "PACF de residuales", len(cal_df)), "Explora autocorrelación parcial de residuales."),
            ],
            notas=["La ecuación de especificación aparece antes de los resultados.", "Los últimos datos reservados se usan para pronóstico y no para calibración."]
        )
        st.download_button("Descargar PDF", pdf_bytes, "informe_regresion_lineal.pdf", "application/pdf")
    with c5:
        if st.button("Eliminar resultados guardados"):
            st.session_state.pop("resultados_regresion_lineal", None)
            st.success("Resultados eliminados.")
