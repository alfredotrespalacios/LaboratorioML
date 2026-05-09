from datetime import datetime
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st

from utils.carga_datos import cargar_datos_modulo, detectar_columnas_numericas
from utils.diagnosticos import diagnostico_residuales
from utils.exportacion import preparar_excel_generico, preparar_json_descarga
from utils.pdf_reportes import crear_pdf_modulo
from utils.graficos import grafico_acf_pacf_interactivo, grafico_coeficientes, grafico_histograma_interactivo, grafico_qq, grafico_real_vs_predicho, grafico_residuales, grafico_residuales_vs_ajustados
from utils.transformaciones import aplicar_log_si_corresponde
from utils.estilos import mostrar_encabezado, caja_pedagogica

st.set_page_config(page_title="Regresión lineal", page_icon="📉", layout="wide")
mostrar_encabezado("📉 Regresión lineal", "OLS con statsmodels, variables en nivel o logaritmo natural y diagnóstico de residuales.")

caja_pedagogica("Permite estimar modelos nivel-nivel, log-nivel, nivel-log, log-log y mixtos. Muestra el reporte completo de statsmodels.")

df, nombre_archivo, fuente_datos = cargar_datos_modulo("data/datos_regresion_lineal.xlsx", "regresion_lineal")
filas_originales = len(df)
st.subheader("Vista previa de datos")
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

st.sidebar.header("3. Transformaciones")
usar_log_y = st.sidebar.radio(f"Transformación de {y_col}", ["Nivel", "ln(Y)"], horizontal=True) == "ln(Y)"
transformaciones_x = {}
for x in x_cols:
    transformaciones_x[x] = st.sidebar.radio(f"Transformación de {x}", ["Nivel", f"ln({x})"], horizontal=True, key=f"log_{x}") != "Nivel"

incluir_intercepto = st.sidebar.checkbox("Incluir intercepto", value=True)
max_lags = st.sidebar.slider("Rezagos ACF/PACF residuales", 5, 40, 20)

if "resultados_regresion_lineal_actuales" not in st.session_state:
    st.session_state["resultados_regresion_lineal_actuales"] = None

if st.button("Estimar regresión lineal"):
    datos = pd.DataFrame()
    y_trans, y_nombre = aplicar_log_si_corresponde(df, y_col, usar_log_y)
    datos[y_nombre] = y_trans

    x_nombres = []
    for x in x_cols:
        x_trans, x_nombre = aplicar_log_si_corresponde(df, x, transformaciones_x[x])
        datos[x_nombre] = x_trans
        x_nombres.append(x_nombre)

    filas_antes = len(datos)
    datos = datos.replace([np.inf, -np.inf], np.nan).dropna()
    filas_usadas = len(datos)

    if filas_usadas < len(x_nombres) + 3:
        st.error("No hay suficientes observaciones válidas para estimar el modelo.")
        st.stop()

    y = datos[y_nombre]
    X = datos[x_nombres]
    if incluir_intercepto:
        X = sm.add_constant(X)

    modelo = sm.OLS(y, X).fit()
    y_hat = modelo.fittedvalues
    resid = modelo.resid

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

    metricas = {
        "r2": float(modelo.rsquared),
        "r2_ajustado": float(modelo.rsquared_adj),
        "f_statistic": float(modelo.fvalue) if modelo.fvalue is not None else None,
        "prob_f_statistic": float(modelo.f_pvalue) if modelo.f_pvalue is not None else None,
        "aic": float(modelo.aic),
        "bic": float(modelo.bic),
    }

    st.session_state["resultados_regresion_lineal_actuales"] = {
        "tipo_analisis": "Regresión lineal",
        "fecha_ejecucion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "archivo_usado": nombre_archivo,
        "fuente_datos": fuente_datos,
        "filas_originales": filas_originales,
        "filas_utilizadas": filas_usadas,
        "variable_dependiente_original": y_col,
        "variable_dependiente_usada": y_nombre,
        "transformacion_y": "logaritmo_natural" if usar_log_y else "nivel",
        "variables_explicativas_originales": x_cols,
        "variables_explicativas_usadas": x_nombres,
        "transformaciones_x": {k: ("logaritmo_natural" if v else "nivel") for k, v in transformaciones_x.items()},
        "observaciones_eliminadas": filas_antes - filas_usadas,
        "metricas": metricas,
        "coeficientes": coef_table.to_dict(orient="records"),
        "diagnostico_residuales": diag_resumen,
        "acf_residuales": diag.get("acf", []),
        "pacf_residuales": diag.get("pacf", []),
        "reporte_statsmodels": modelo.summary().as_text(),
        "predicciones": pd.DataFrame({"y_observada": y, "y_estimada": y_hat, "residual": resid}).to_dict(orient="records"),
    }

res = st.session_state["resultados_regresion_lineal_actuales"]
if res:
    st.subheader("Reporte completo OLS")
    st.text(res["reporte_statsmodels"])

    coef_df = pd.DataFrame(res["coeficientes"])
    st.subheader("Coeficientes y significancia")
    st.dataframe(coef_df, use_container_width=True)
    st.plotly_chart(grafico_coeficientes(coef_df), use_container_width=True)

    st.subheader("Métricas del modelo")
    st.json(res["metricas"])

    pred_df = pd.DataFrame(res["predicciones"])
    st.subheader("Gráficos del modelo")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(grafico_real_vs_predicho(pred_df["y_observada"], pred_df["y_estimada"], "Y observada vs. Y estimada"), use_container_width=True)
        st.plotly_chart(grafico_residuales(pred_df["residual"], "Residuales"), use_container_width=True)
        st.plotly_chart(grafico_histograma_interactivo(pred_df["residual"], "Histograma de residuales"), use_container_width=True)
    with c2:
        st.plotly_chart(grafico_residuales_vs_ajustados(pred_df["y_estimada"], pred_df["residual"]), use_container_width=True)
        st.plotly_chart(grafico_qq(pred_df["residual"]), use_container_width=True)

    st.subheader("Diagnóstico de residuales")
    st.json(res["diagnostico_residuales"])

    p_jb = res["diagnostico_residuales"].get("p_valor_jarque_bera")
    if p_jb is not None:
        if p_jb < 0.05:
            st.warning(
                "Conclusión sobre normalidad de residuales: como el p-valor de Jarque-Bera es menor que 0.05, "
                "se rechaza la hipótesis de normalidad de los residuales al 5%. Esto sugiere que los errores no "
                "siguen una distribución normal, lo cual puede afectar la inferencia estadística del modelo."
            )
        else:
            st.success(
                "Conclusión sobre normalidad de residuales: como el p-valor de Jarque-Bera es mayor o igual que 0.05, "
                "no se rechaza la hipótesis de normalidad de los residuales al 5%."
            )

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(grafico_acf_pacf_interactivo(res["acf_residuales"], "ACF de residuales", len(pred_df)), use_container_width=True)
    with c4:
        st.plotly_chart(grafico_acf_pacf_interactivo(res["pacf_residuales"], "PACF de residuales", len(pred_df)), use_container_width=True)

    st.markdown("---")
    st.subheader("Gestión de resultados")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        if st.button("Guardar resultados para resumen ejecutivo"):
            st.session_state["resultados_regresion_lineal"] = {**res, "fecha_guardado": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            st.success("Resultados guardados.")
    with c2:
        excel = preparar_excel_generico(res, {
            "Coeficientes": coef_df,
            "Metricas": pd.DataFrame([res["metricas"]]),
            "Diagnostico": pd.DataFrame([res["diagnostico_residuales"]]),
            "Predicciones": pred_df,
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
                ("Coeficientes", coef_df),
                ("Métricas", pd.DataFrame([res["metricas"]])),
                ("Diagnóstico de residuales", pd.DataFrame([res["diagnostico_residuales"]])),
                ("Predicciones y residuales", pred_df),
                ("ACF residuales", pd.DataFrame(res["acf_residuales"])),
                ("PACF residuales", pd.DataFrame(res["pacf_residuales"])),
            ],
            figuras=[
                ("Y observada vs. Y estimada", grafico_real_vs_predicho(pred_df["y_observada"], pred_df["y_estimada"], "Y observada vs. Y estimada"), "Puntos más cercanos a la diagonal indican mejor ajuste."),
                ("Residuales", grafico_residuales(pred_df["residual"], "Residuales"), "Permite observar patrones, cambios de volatilidad o valores atípicos."),
                ("Histograma de residuales", grafico_histograma_interactivo(pred_df["residual"], "Histograma de residuales"), "Ayuda a revisar la forma de la distribución de errores."),
                ("Residuales vs. ajustados", grafico_residuales_vs_ajustados(pred_df["y_estimada"], pred_df["residual"]), "Permite revisar si los errores dependen del nivel estimado."),
                ("QQ plot", grafico_qq(pred_df["residual"]), "Ayuda a evaluar visualmente normalidad de los residuales."),
                ("ACF de residuales", grafico_acf_pacf_interactivo(res["acf_residuales"], "ACF de residuales", len(pred_df)), "Explora autocorrelación de residuales."),
                ("PACF de residuales", grafico_acf_pacf_interactivo(res["pacf_residuales"], "PACF de residuales", len(pred_df)), "Explora autocorrelación parcial de residuales."),
            ],
            notas=["La conclusión sobre normalidad de residuales se basa en Jarque-Bera.", "La significancia global se evalúa con F-statistic y Prob(F-statistic)."]
        )
        st.download_button("Descargar PDF", pdf_bytes, "informe_regresion_lineal.pdf", "application/pdf")
    with c5:
        if st.button("Eliminar resultados guardados"):
            st.session_state.pop("resultados_regresion_lineal", None)
            st.success("Resultados eliminados.")
