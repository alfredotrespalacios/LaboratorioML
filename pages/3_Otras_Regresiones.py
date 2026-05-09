from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

from utils.carga_datos import cargar_datos_modulo, detectar_columnas_numericas
from utils.diagnosticos import metricas_regresion
from utils.exportacion import preparar_excel_generico, preparar_json_descarga
from utils.graficos import grafico_histograma_interactivo, grafico_importancia_variables, grafico_lineas_modelo, grafico_real_vs_predicho, grafico_residuales
from utils.modelos import obtener_regresor
from utils.pdf_reportes import crear_pdf_modulo
from utils.pronostico import (
    MODO_EVALUACION,
    MODO_VALORES_NO_OBSERVADOS,
    agregar_errores_pronostico,
    descripcion_modo_pronostico,
    metricas_pronostico,
)
from utils.transformaciones import escalar_train_test
from utils.variables import capturar_descripcion_variables, descripcion_variables_df
from utils.estilos import mostrar_encabezado, caja_pedagogica

st.set_page_config(page_title="Otras regresiones", page_icon="🌲", layout="wide")
mostrar_encabezado("🌲 Otras regresiones", "Modelos predictivos de regresión con selección de algoritmo.")

caja_pedagogica("Permite comparar algoritmos de regresión no lineal y usar dos modos de pronóstico.")

df, nombre_archivo, fuente_datos = cargar_datos_modulo("data/datos_otras_regresiones.xlsx", "otras_regresiones")
st.subheader("Vista previa de los datos seleccionados")
st.dataframe(df.head(20), use_container_width=True)

num_cols = detectar_columnas_numericas(df)
y_col = st.sidebar.selectbox("Variable objetivo Y", num_cols)
x_cols = st.sidebar.multiselect("Variables explicativas X", [c for c in num_cols if c != y_col], default=[c for c in num_cols if c != y_col][:5])
alg = st.sidebar.selectbox("Algoritmo", ["Árbol de decisión", "Random Forest", "Gradient Boosting", "KNN Regressor", "Support Vector Regression"])

modo_pronostico = st.sidebar.radio("Modo de pronóstico", [MODO_EVALUACION, MODO_VALORES_NO_OBSERVADOS])
info_modo = descripcion_modo_pronostico(modo_pronostico)

if modo_pronostico == MODO_EVALUACION:
    max_pronostico = max(0, len(df) - 10)
    n_pronostico = st.sidebar.number_input("Número de datos finales para evaluación de pronóstico", min_value=0, max_value=max_pronostico, value=min(20, max_pronostico), step=1)
else:
    n_pronostico = 0
    st.sidebar.info("Este modo usa filas donde la variable Y está vacía y las X están disponibles.")

escalar = st.sidebar.checkbox("Estandarizar variables X", value=alg in ["KNN Regressor", "Support Vector Regression"])

EXPLICACIONES_REGRESION = {
    "Árbol de decisión": "**Árbol de decisión:** divide los datos en reglas sucesivas para predecir la variable objetivo. Es fácil de interpretar, pero puede sobreajustarse.",
    "Random Forest": "**Random Forest:** construye muchos árboles y combina sus predicciones. Suele ser más estable y captura relaciones no lineales.",
    "Gradient Boosting": "**Gradient Boosting:** construye modelos secuenciales, donde cada nuevo árbol corrige errores del anterior.",
    "KNN Regressor": "**KNN Regressor:** predice usando observaciones cercanas. Es sensible a la escala de las variables.",
    "Support Vector Regression": "**Support Vector Regression:** busca una función predictiva dentro de un margen de tolerancia. Es sensible a la escala.",
}
with st.expander("Explicación del método seleccionado", expanded=True):
    st.markdown(EXPLICACIONES_REGRESION[alg])
with st.expander("Modo de pronóstico seleccionado", expanded=True):
    st.markdown(f"**Modo:** {info_modo['modo_pronostico']}\n\n**Uso:** {info_modo['uso_modo_pronostico']}\n\n**Implicación:** {info_modo['implicacion_modo_pronostico']}")

if not x_cols:
    st.warning("Seleccione variables explicativas.")
    st.stop()

descripcion_variables = capturar_descripcion_variables([y_col] + x_cols, "otras_regresiones")

if "resultados_otras_regresiones_actuales" not in st.session_state:
    st.session_state["resultados_otras_regresiones_actuales"] = None

if st.button("Entrenar modelo"):
    datos = df[[y_col] + x_cols].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)

    if modo_pronostico == MODO_EVALUACION:
        datos_validos = datos.dropna(subset=[y_col] + x_cols).copy()
        if int(n_pronostico) >= len(datos_validos) - 5:
            st.error("El número de datos para pronóstico es demasiado alto.")
            st.stop()
        train_df = datos_validos.iloc[:-int(n_pronostico)].copy() if int(n_pronostico) > 0 else datos_validos.copy()
        fore_df = datos_validos.iloc[-int(n_pronostico):].copy() if int(n_pronostico) > 0 else pd.DataFrame(columns=datos.columns)
    else:
        train_df = datos.dropna(subset=[y_col] + x_cols).copy()
        fore_df = datos[datos[y_col].isna()].dropna(subset=x_cols).copy()
        if fore_df.empty:
            st.warning("No se encontraron filas con Y vacía y variables X completas para pronosticar valores no observados.")

    X_train = train_df[x_cols]
    y_train = train_df[y_col]
    X_train_model = X_train
    scaler = None
    if escalar:
        scaler = StandardScaler()
        X_train_model = scaler.fit_transform(X_train)

    modelo = obtener_regresor(alg)
    modelo.fit(X_train_model, y_train)
    y_pred_train = modelo.predict(X_train_model)

    metricas = metricas_regresion(y_train, y_pred_train)
    pred_df = pd.DataFrame({"observacion": train_df.index, "y_real": y_train.values, "y_predicho": y_pred_train, "error": y_train.values - y_pred_train})

    pronostico_df = pd.DataFrame()
    if not fore_df.empty:
        X_fore = fore_df[x_cols]
        X_fore_model = scaler.transform(X_fore) if scaler is not None else X_fore
        y_fore = modelo.predict(X_fore_model)
        pronostico_df = pd.DataFrame({
            "observacion": fore_df.index,
            "y_real": fore_df[y_col].values,
            "y_pronosticada": y_fore,
            "periodo": "evaluacion_pronostico" if modo_pronostico == MODO_EVALUACION else "valor_no_observado",
        })
        if modo_pronostico == MODO_EVALUACION:
            pronostico_df = agregar_errores_pronostico(pronostico_df, "y_real", "y_pronosticada")

    metricas_fore = metricas_pronostico(pronostico_df.get("y_real", []), pronostico_df.get("y_pronosticada", [])) if modo_pronostico == MODO_EVALUACION and not pronostico_df.empty else {}

    serie_modelo_df = pd.DataFrame({"observacion": pred_df["observacion"], "y_observada": pred_df["y_real"], "y_estimada": pred_df["y_predicho"], "y_pronosticada": np.nan})
    if not pronostico_df.empty:
        tmp = pd.DataFrame({
            "observacion": pronostico_df["observacion"],
            "y_observada": pronostico_df["y_real"] if "y_real" in pronostico_df.columns else np.nan,
            "y_estimada": np.nan,
            "y_pronosticada": pronostico_df["y_pronosticada"],
        })
        serie_modelo_df = pd.concat([serie_modelo_df, tmp], ignore_index=True).sort_values("observacion")

    importancia_df = pd.DataFrame()
    if hasattr(modelo, "feature_importances_"):
        importancia_df = pd.DataFrame({"variable": x_cols, "importancia": modelo.feature_importances_}).sort_values("importancia")

    st.session_state["resultados_otras_regresiones_actuales"] = {
        "tipo_analisis": "Otras regresiones",
        "fecha_ejecucion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "archivo_usado": nombre_archivo,
        "fuente_datos": fuente_datos,
        "algoritmo": alg,
        "variable_objetivo": y_col,
        "variables_explicativas": x_cols,
        "descripcion_variables": descripcion_variables,
        "estandarizacion": escalar,
        **info_modo,
        "n_pronostico": int(n_pronostico) if modo_pronostico == MODO_EVALUACION else None,
        "filas_calibracion": len(train_df),
        "filas_pronostico": len(fore_df),
        "metricas": metricas,
        "metricas_pronostico": metricas_fore,
        "predicciones": pred_df.to_dict(orient="records"),
        "pronostico": pronostico_df.to_dict(orient="records"),
        "serie_modelo": serie_modelo_df.to_dict(orient="records"),
        "importancia_variables": importancia_df.to_dict(orient="records"),
    }

res = st.session_state["resultados_otras_regresiones_actuales"]
if res:
    st.subheader("Modo de pronóstico usado")
    st.info(f"**{res.get('modo_pronostico')}**\n\n**Uso:** {res.get('uso_modo_pronostico')}\n\n**Implicación:** {res.get('implicacion_modo_pronostico')}")

    st.subheader("Métricas de calibración")
    st.json(res["metricas"])

    pred_df = pd.DataFrame(res["predicciones"])
    pronostico_df = pd.DataFrame(res.get("pronostico", []))
    serie_modelo_df = pd.DataFrame(res.get("serie_modelo", []))
    imp_df = pd.DataFrame(res["importancia_variables"])
    desc_df = descripcion_variables_df(res.get("descripcion_variables", {}))
    metricas_pronostico_df = pd.DataFrame([res.get("metricas_pronostico", {})])

    st.plotly_chart(grafico_lineas_modelo(pred_df, "observacion", ["y_real", "y_predicho"], "Variable explicada real vs. predicha en calibración"), use_container_width=True)
    st.caption("Siempre que hay regresión, esta gráfica muestra la variable explicada real y el resultado del modelo.")

    st.plotly_chart(grafico_real_vs_predicho(pred_df["y_real"], pred_df["y_predicho"]), use_container_width=True)
    st.plotly_chart(grafico_residuales(pred_df["error"], "Errores de calibración"), use_container_width=True)
    st.plotly_chart(grafico_histograma_interactivo(pred_df["error"], "Distribución de errores de calibración"), use_container_width=True)

    st.subheader("Pronóstico")
    if pronostico_df.empty:
        st.info("No hay filas disponibles para pronóstico según el modo seleccionado.")
    else:
        st.dataframe(pronostico_df, use_container_width=True)
        if res.get("metricas_pronostico"):
            st.write("**Métricas de evaluación de pronóstico**")
            st.dataframe(metricas_pronostico_df, use_container_width=True)
        st.plotly_chart(grafico_lineas_modelo(serie_modelo_df, "observacion", ["y_observada", "y_estimada", "y_pronosticada"], "Calibración y pronóstico"), use_container_width=True)

    if not desc_df.empty:
        st.subheader("Descripción de variables")
        st.dataframe(desc_df, use_container_width=True)
    if not imp_df.empty:
        st.plotly_chart(grafico_importancia_variables(imp_df), use_container_width=True)

    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        if st.button("Guardar resultados para resumen ejecutivo"):
            st.session_state["resultados_otras_regresiones"] = {**res, "fecha_guardado": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            st.success("Resultados guardados.")
    with c2:
        excel = preparar_excel_generico(res, {
            "Descripcion_variables": desc_df,
            "Modo_pronostico": pd.DataFrame([{
                "modo_pronostico": res.get("modo_pronostico"),
                "uso_modo_pronostico": res.get("uso_modo_pronostico"),
                "implicacion_modo_pronostico": res.get("implicacion_modo_pronostico"),
            }]),
            "Metricas": pd.DataFrame([res["metricas"]]),
            "Metricas_pronostico": metricas_pronostico_df,
            "Predicciones": pred_df,
            "Pronostico": pronostico_df,
            "Serie_modelo": serie_modelo_df,
            "Importancia": imp_df
        })
        st.download_button("Descargar Excel", excel, "resultados_otras_regresiones.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with c3:
        st.download_button("Descargar JSON", preparar_json_descarga(res), "resultados_otras_regresiones.json", "application/json")
    with c4:
        figs = [
            ("Variable explicada real vs. predicha", grafico_lineas_modelo(pred_df, "observacion", ["y_real", "y_predicho"], "Variable explicada real vs. predicha"), "Muestra la variable explicada real y el resultado del modelo."),
            ("Real vs. predicho", grafico_real_vs_predicho(pred_df["y_real"], pred_df["y_predicho"]), "Puntos cercanos a la diagonal indican mejores predicciones."),
            ("Errores de calibración", grafico_residuales(pred_df["error"], "Errores de calibración"), "Permite observar errores de calibración."),
            ("Calibración y pronóstico", grafico_lineas_modelo(serie_modelo_df, "observacion", ["y_observada", "y_estimada", "y_pronosticada"], "Calibración y pronóstico"), "Muestra calibración y pronóstico según el modo seleccionado."),
        ]
        if not imp_df.empty:
            figs.append(("Importancia de variables", grafico_importancia_variables(imp_df), "Muestra qué variables pesan más en el modelo."))
        pdf_bytes = crear_pdf_modulo(
            "Informe del módulo: Otras regresiones",
            res,
            tablas=[
                ("Descripción de variables", desc_df),
                ("Modo de pronóstico", pd.DataFrame([{
                    "modo_pronostico": res.get("modo_pronostico"),
                    "uso_modo_pronostico": res.get("uso_modo_pronostico"),
                    "implicacion_modo_pronostico": res.get("implicacion_modo_pronostico"),
                }])),
                ("Métricas", pd.DataFrame([res["metricas"]])),
                ("Métricas de pronóstico", metricas_pronostico_df),
                ("Predicciones", pred_df),
                ("Pronóstico", pronostico_df),
                ("Importancia de variables", imp_df)
            ],
            figuras=figs,
            notas=[res.get("uso_modo_pronostico", ""), res.get("implicacion_modo_pronostico", "")]
        )
        st.download_button("Descargar PDF", pdf_bytes, "informe_otras_regresiones.pdf", "application/pdf")
    with c5:
        if st.button("Eliminar resultados guardados"):
            st.session_state.pop("resultados_otras_regresiones", None)
            st.success("Resultados eliminados.")
