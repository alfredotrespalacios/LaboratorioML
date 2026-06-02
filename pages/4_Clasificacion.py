from datetime import datetime
import pandas as pd
import streamlit as st

from utils.carga_datos import cargar_datos_modulo, detectar_columnas_numericas
from utils.diagnosticos import curva_roc_df, matriz_confusion_df, metricas_clasificacion
from utils.exportacion import preparar_excel_generico, preparar_json_descarga
from utils.pdf_reportes import crear_pdf_modulo
from utils.graficos import grafico_curva_roc, grafico_importancia_variables, grafico_matriz_confusion
from utils.modelos import obtener_clasificador
from utils.transformaciones import codificar_y, escalar_train_test
from utils.estilos import mostrar_encabezado, caja_pedagogica
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Clasificación", page_icon="✅", layout="wide")
mostrar_encabezado("✅ Clasificación", "Modelos para predecir categorías con selección de algoritmo.")

caja_pedagogica("Evalúa accuracy, precision, recall, F1, matriz de confusión y curva ROC cuando aplica.")

df, nombre_archivo, fuente_datos = cargar_datos_modulo("data/datos_clasificacion.xlsx", "clasificacion")
st.dataframe(df.head(20), use_container_width=True)

num_cols = detectar_columnas_numericas(df)
target = st.sidebar.selectbox("Variable objetivo categórica", df.columns.tolist())
x_cols = st.sidebar.multiselect("Variables explicativas numéricas", [c for c in num_cols if c != target], default=[c for c in num_cols if c != target][:5])
alg = st.sidebar.selectbox("Algoritmo", ["Regresión logística", "Árbol de decisión", "Random Forest", "KNN Classifier", "Support Vector Machine", "Gradient Boosting"])
test_size = st.sidebar.slider("Porcentaje de prueba", 10, 40, 25) / 100
escalar = st.sidebar.checkbox("Estandarizar X", value=alg in ["Regresión logística", "KNN Classifier", "Support Vector Machine"])

if not x_cols:
    st.warning("Seleccione variables explicativas.")
    st.stop()

if "resultados_clasificacion_actuales" not in st.session_state:
    st.session_state["resultados_clasificacion_actuales"] = None

if st.button("Entrenar clasificador"):
    datos = df[[target] + x_cols].dropna()
    X = datos[x_cols]
    y_cod, le = codificar_y(datos[target])
    X_train, X_test, y_train, y_test = train_test_split(X, y_cod, test_size=test_size, random_state=42, stratify=y_cod if len(set(y_cod)) > 1 else None)
    X_train_model, X_test_model, _ = escalar_train_test(X_train, X_test, escalar)
    modelo = obtener_clasificador(alg)
    modelo.fit(X_train_model, y_train)
    y_pred = modelo.predict(X_test_model)

    y_prob = None
    if hasattr(modelo, "predict_proba") and len(le.classes_) == 2:
        y_prob = modelo.predict_proba(X_test_model)[:, 1]

    metricas = metricas_clasificacion(y_test, y_pred, y_prob)
    cm_df = matriz_confusion_df(y_test, y_pred, labels=list(range(len(le.classes_))))
    roc = pd.DataFrame()
    if y_prob is not None:
        roc = curva_roc_df(y_test, y_prob)

    importancia_df = pd.DataFrame()
    if hasattr(modelo, "feature_importances_"):
        importancia_df = pd.DataFrame({"variable": x_cols, "importancia": modelo.feature_importances_}).sort_values("importancia")

    pred_df = pd.DataFrame({"y_real": y_test, "y_predicho": y_pred})
    if y_prob is not None:
        pred_df["probabilidad_clase_1"] = y_prob

    st.session_state["resultados_clasificacion_actuales"] = {
        "tipo_analisis": "Clasificación",
        "fecha_ejecucion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "archivo_usado": nombre_archivo,
        "fuente_datos": fuente_datos,
        "algoritmo": alg,
        "variable_objetivo": target,
        "clases": list(le.classes_),
        "variables_explicativas": x_cols,
        "test_size": test_size,
        "estandarizacion": escalar,
        "metricas": metricas,
        "matriz_confusion": cm_df.reset_index().to_dict(orient="records"),
        "predicciones": pred_df.to_dict(orient="records"),
        "roc": roc.to_dict(orient="records"),
        "importancia_variables": importancia_df.to_dict(orient="records"),
    }

res = st.session_state["resultados_clasificacion_actuales"]
if res:
    st.subheader("Métricas")
    st.json(res["metricas"])
    cm_df = pd.DataFrame(res["matriz_confusion"]).set_index("index")
    st.plotly_chart(grafico_matriz_confusion(cm_df), use_container_width=True)
    roc_df = pd.DataFrame(res["roc"])
    if not roc_df.empty:
        st.plotly_chart(grafico_curva_roc(roc_df, res["metricas"].get("AUC")), use_container_width=True)
    imp_df = pd.DataFrame(res["importancia_variables"])
    if not imp_df.empty:
        st.plotly_chart(grafico_importancia_variables(imp_df), use_container_width=True)

    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        if st.button("Guardar resultados para resumen ejecutivo"):
            st.session_state["resultados_clasificacion"] = {**res, "fecha_guardado": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            st.success("Resultados guardados.")
    with c2:
        excel = preparar_excel_generico(res, {"Metricas": pd.DataFrame([res["metricas"]]), "Matriz_confusion": cm_df.reset_index(), "Predicciones": pd.DataFrame(res["predicciones"]), "ROC": roc_df, "Importancia": imp_df})
        st.download_button("Descargar Excel", excel, "resultados_clasificacion.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with c3:
        st.download_button("Descargar JSON", preparar_json_descarga(res), "resultados_clasificacion.json", "application/json")
    with c4:
        figs = [("Matriz de confusión", grafico_matriz_confusion(cm_df), "La diagonal principal muestra clasificaciones correctas.")]
        if not roc_df.empty:
            figs.append(("Curva ROC", grafico_curva_roc(roc_df, res["metricas"].get("AUC")), "La curva ROC evalúa la capacidad de discriminar entre clases."))
        if not imp_df.empty:
            figs.append(("Importancia de variables", grafico_importancia_variables(imp_df), "Muestra qué variables aportan más al modelo."))
        pdf_bytes = crear_pdf_modulo(
            "Informe del módulo: Clasificación",
            res,
            tablas=[("Métricas", pd.DataFrame([res["metricas"]])), ("Matriz de confusión", cm_df.reset_index()), ("Predicciones", pd.DataFrame(res["predicciones"])), ("ROC", roc_df), ("Importancia", imp_df)],
            figuras=figs,
            notas=["Accuracy, precision, recall y F1 resumen el desempeño clasificatorio.", "La matriz de confusión muestra aciertos y errores por clase."]
        )
        st.download_button("Descargar PDF", pdf_bytes, "informe_clasificacion.pdf", "application/pdf")
    with c5:
        if st.button("Eliminar resultados guardados"):
            st.session_state.pop("resultados_clasificacion", None)
            st.success("Resultados eliminados.")
