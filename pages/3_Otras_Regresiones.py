from datetime import datetime
import pandas as pd
import streamlit as st

from utils.carga_datos import cargar_datos_modulo, detectar_columnas_numericas
from utils.diagnosticos import metricas_regresion
from utils.exportacion import preparar_excel_generico, preparar_json_descarga
from utils.graficos import grafico_histograma_interactivo, grafico_importancia_variables, grafico_lineas_modelo, grafico_real_vs_predicho, grafico_residuales
from utils.modelos import obtener_regresor
from utils.pdf_reportes import crear_pdf_modulo
from utils.transformaciones import preparar_xy, escalar_train_test
from utils.variables import capturar_descripcion_variables, descripcion_variables_df
from utils.estilos import mostrar_encabezado, caja_pedagogica

st.set_page_config(page_title="Otras regresiones", page_icon="🌲", layout="wide")
mostrar_encabezado("🌲 Otras regresiones", "Modelos predictivos de regresión con selección de algoritmo.")

caja_pedagogica("Permite comparar algoritmos de regresión no lineal y evaluar el error predictivo.")

df, nombre_archivo, fuente_datos = cargar_datos_modulo("data/datos_otras_regresiones.xlsx", "otras_regresiones")
st.subheader("Vista previa de los datos seleccionados")
st.dataframe(df.head(20), use_container_width=True)

num_cols = detectar_columnas_numericas(df)
y_col = st.sidebar.selectbox("Variable objetivo Y", num_cols)
x_cols = st.sidebar.multiselect("Variables explicativas X", [c for c in num_cols if c != y_col], default=[c for c in num_cols if c != y_col][:5])
alg = st.sidebar.selectbox("Algoritmo", ["Árbol de decisión", "Random Forest", "Gradient Boosting", "KNN Regressor", "Support Vector Regression"])
test_size = st.sidebar.slider("Porcentaje de prueba", 10, 40, 25) / 100
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

with st.expander("Explicación de configuración: porcentaje de prueba y estandarización", expanded=True):
    st.markdown(
        """
        **Porcentaje de prueba:** proporción de datos reservada para evaluar el modelo después del entrenamiento.
        Por ejemplo, 25% significa que el modelo aprende con el 75% y se evalúa con el 25% restante.

        **Estandarizar variables X:** transforma las variables explicativas para que tengan escala comparable.
        Es especialmente importante en KNN y Support Vector Regression.
        """
    )

if not x_cols:
    st.warning("Seleccione variables explicativas.")
    st.stop()

descripcion_variables = capturar_descripcion_variables([y_col] + x_cols, "otras_regresiones")

if "resultados_otras_regresiones_actuales" not in st.session_state:
    st.session_state["resultados_otras_regresiones_actuales"] = None

if st.button("Entrenar modelo"):
    X_train, X_test, y_train, y_test = preparar_xy(df, y_col, x_cols, test_size)
    X_train_model, X_test_model, _ = escalar_train_test(X_train, X_test, escalar)
    modelo = obtener_regresor(alg)
    modelo.fit(X_train_model, y_train)
    y_pred = modelo.predict(X_test_model)
    metricas = metricas_regresion(y_test, y_pred)

    pred_df = pd.DataFrame({"observacion": range(len(y_test)), "y_real": y_test.values, "y_predicho": y_pred, "error": y_test.values - y_pred})
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
        "test_size": test_size,
        "estandarizacion": escalar,
        "metricas": metricas,
        "predicciones": pred_df.to_dict(orient="records"),
        "importancia_variables": importancia_df.to_dict(orient="records"),
    }

res = st.session_state["resultados_otras_regresiones_actuales"]
if res:
    st.subheader("Métricas")
    st.json(res["metricas"])
    pred_df = pd.DataFrame(res["predicciones"])
    imp_df = pd.DataFrame(res["importancia_variables"])
    desc_df = descripcion_variables_df(res.get("descripcion_variables", {}))

    st.plotly_chart(grafico_lineas_modelo(pred_df, "observacion", ["y_real", "y_predicho"], "Variable explicada real vs. predicha"), use_container_width=True)
    st.caption("Siempre que hay regresión, esta gráfica muestra la variable explicada real y el resultado del modelo.")

    st.plotly_chart(grafico_real_vs_predicho(pred_df["y_real"], pred_df["y_predicho"]), use_container_width=True)
    st.caption("Puntos más cercanos a la línea diagonal indican mejores predicciones.")

    st.plotly_chart(grafico_residuales(pred_df["error"], "Errores de predicción"), use_container_width=True)
    st.caption("Permite observar si los errores tienen patrones o fluctúan alrededor de cero.")

    st.plotly_chart(grafico_histograma_interactivo(pred_df["error"], "Distribución de errores"), use_container_width=True)
    st.caption("Muestra si los errores están centrados alrededor de cero y si existen valores extremos.")

    if not desc_df.empty:
        st.subheader("Descripción de variables")
        st.dataframe(desc_df, use_container_width=True)

    if not imp_df.empty:
        st.plotly_chart(grafico_importancia_variables(imp_df), use_container_width=True)
        st.caption("Indica qué variables fueron más importantes para el modelo.")

    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        if st.button("Guardar resultados para resumen ejecutivo"):
            st.session_state["resultados_otras_regresiones"] = {**res, "fecha_guardado": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            st.success("Resultados guardados.")
    with c2:
        excel = preparar_excel_generico(res, {"Descripcion_variables": desc_df, "Metricas": pd.DataFrame([res["metricas"]]), "Predicciones": pred_df, "Importancia": imp_df})
        st.download_button("Descargar Excel", excel, "resultados_otras_regresiones.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with c3:
        st.download_button("Descargar JSON", preparar_json_descarga(res), "resultados_otras_regresiones.json", "application/json")
    with c4:
        figs = [
            ("Variable explicada real vs. predicha", grafico_lineas_modelo(pred_df, "observacion", ["y_real", "y_predicho"], "Variable explicada real vs. predicha"), "Muestra la variable explicada real y el resultado del modelo."),
            ("Real vs. predicho", grafico_real_vs_predicho(pred_df["y_real"], pred_df["y_predicho"]), "Puntos cercanos a la diagonal indican mejores predicciones."),
            ("Errores de predicción", grafico_residuales(pred_df["error"], "Errores de predicción"), "Permite observar si los errores tienen patrones."),
            ("Distribución de errores", grafico_histograma_interactivo(pred_df["error"], "Distribución de errores"), "Ayuda a revisar si los errores están centrados alrededor de cero."),
        ]
        if not imp_df.empty:
            figs.append(("Importancia de variables", grafico_importancia_variables(imp_df), "Muestra qué variables pesan más en el modelo."))
        pdf_bytes = crear_pdf_modulo(
            "Informe del módulo: Otras regresiones",
            res,
            tablas=[("Descripción de variables", desc_df), ("Métricas", pd.DataFrame([res["metricas"]])), ("Predicciones", pred_df), ("Importancia de variables", imp_df)],
            figuras=figs,
            notas=["El porcentaje de prueba corresponde a la proporción reservada para evaluación fuera de muestra.", "La estandarización vuelve comparables las escalas de las variables X."]
        )
        st.download_button("Descargar PDF", pdf_bytes, "informe_otras_regresiones.pdf", "application/pdf")
    with c5:
        if st.button("Eliminar resultados guardados"):
            st.session_state.pop("resultados_otras_regresiones", None)
            st.success("Resultados eliminados.")
