from datetime import datetime
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.carga_datos import cargar_datos_modulo, detectar_columnas_numericas
from utils.diagnosticos import matriz_confusion_df, metricas_clasificacion, metricas_regresion
from utils.exportacion import preparar_excel_generico, preparar_json_descarga
from utils.pdf_reportes import crear_pdf_modulo
from utils.graficos import grafico_loss_curve, grafico_matriz_confusion, grafico_real_vs_predicho
from utils.modelos import obtener_red_neuronal
from utils.transformaciones import codificar_y
from utils.estilos import mostrar_encabezado, caja_pedagogica

st.set_page_config(page_title="Red neuronal", page_icon="🧠", layout="wide")
mostrar_encabezado("🧠 Red neuronal", "Red neuronal multicapa para regresión o clasificación.")

caja_pedagogica("Permite configurar capas, neuronas, activación, iteraciones y tasa de aprendizaje.")

with st.expander("Guía rápida: ¿para qué sirven los parámetros de la red neuronal?", expanded=True):
    st.markdown(
        """
        **Tipo de problema:** define si la red se usará para predecir un valor continuo, como en regresión, o una categoría, como en clasificación.

        **Número de capas ocultas:** indica cuántas etapas internas de procesamiento tendrá la red. Más capas pueden capturar relaciones más complejas, pero también aumentan el riesgo de sobreajuste y hacen más difícil el entrenamiento.

        **Neuronas por capa:** define la capacidad de aprendizaje de cada capa. Más neuronas dan mayor flexibilidad al modelo, pero pueden hacerlo más pesado y propenso a aprender ruido.

        **Función de activación:** transforma la información dentro de la red. `relu` suele ser una opción práctica; `tanh` y `logistic` pueden servir en algunos casos, pero pueden entrenar más lentamente.

        **Iteraciones máximas:** número máximo de intentos de ajuste de los pesos de la red. Si son pocas, la red puede no aprender suficiente; si son muchas, puede tardar más o sobreajustarse.

        **Tasa de aprendizaje:** controla el tamaño de los pasos durante el entrenamiento. Una tasa muy alta puede impedir la convergencia; una tasa muy baja puede hacer el entrenamiento demasiado lento.

        **Porcentaje de prueba:** parte de los datos que se reserva para evaluar el modelo fuera de la muestra de entrenamiento. Ayuda a revisar si el modelo generaliza.
        """
    )

df, nombre_archivo, fuente_datos = cargar_datos_modulo("data/datos_red_neuronal.xlsx", "red_neuronal")
st.dataframe(df.head(20), use_container_width=True)

num_cols = detectar_columnas_numericas(df)
tipo = st.sidebar.selectbox("Tipo de problema", ["Regresión", "Clasificación"])
target_default = "Y_Regresion" if tipo == "Regresión" and "Y_Regresion" in df.columns else ("Clase" if "Clase" in df.columns else num_cols[0])
target = st.sidebar.selectbox("Variable objetivo", df.columns.tolist(), index=df.columns.tolist().index(target_default) if target_default in df.columns else 0)
x_cols = st.sidebar.multiselect("Variables explicativas numéricas", [c for c in num_cols if c != target], default=[c for c in num_cols if c != target][:5])

n_capas = st.sidebar.slider("Número de capas ocultas", 1, 3, 1)
neuronas = st.sidebar.slider("Neuronas por capa", 2, 50, 10)
activation = st.sidebar.selectbox("Función de activación", ["relu", "tanh", "logistic"])
max_iter = st.sidebar.slider("Iteraciones máximas", 100, 1000, 400, 50)
learning_rate = st.sidebar.number_input("Tasa de aprendizaje", min_value=0.0001, max_value=0.1, value=0.001, step=0.0005, format="%.4f")
test_size = st.sidebar.slider("Porcentaje de prueba", 10, 40, 25) / 100

if not x_cols:
    st.warning("Seleccione variables explicativas.")
    st.stop()

if "resultados_red_neuronal_actuales" not in st.session_state:
    st.session_state["resultados_red_neuronal_actuales"] = None

if st.button("Entrenar red neuronal"):
    datos = df[[target] + x_cols].dropna()
    X = datos[x_cols]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    if tipo == "Regresión":
        y = datos[target]
    else:
        y, le = codificar_y(datos[target])

    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=test_size, random_state=42)
    hidden = tuple([neuronas] * n_capas)
    modelo = obtener_red_neuronal(tipo, hidden, activation, max_iter, learning_rate)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    if tipo == "Regresión":
        metricas = metricas_regresion(y_test, y_pred)
        pred_df = pd.DataFrame({"y_real": y_test.values if hasattr(y_test, "values") else y_test, "y_predicho": y_pred})
        cm_df = pd.DataFrame()
    else:
        metricas = metricas_clasificacion(y_test, y_pred)
        pred_df = pd.DataFrame({"y_real": y_test, "y_predicho": y_pred})
        cm_df = matriz_confusion_df(y_test, y_pred)

    loss_curve = getattr(modelo, "loss_curve_", [])

    st.session_state["resultados_red_neuronal_actuales"] = {
        "tipo_analisis": "Red neuronal",
        "fecha_ejecucion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "archivo_usado": nombre_archivo,
        "fuente_datos": fuente_datos,
        "tipo_problema": tipo,
        "variable_objetivo": target,
        "variables_explicativas": x_cols,
        "arquitectura": str(hidden),
        "activation": activation,
        "max_iter": max_iter,
        "learning_rate_init": learning_rate,
        "test_size": test_size,
        "metricas": metricas,
        "predicciones": pred_df.to_dict(orient="records"),
        "matriz_confusion": cm_df.reset_index().to_dict(orient="records") if not cm_df.empty else [],
        "loss_curve": list(map(float, loss_curve)) if loss_curve is not None else [],
    }

res = st.session_state["resultados_red_neuronal_actuales"]
if res:
    st.subheader("Métricas")
    st.json(res["metricas"])
    st.caption(
        "Las métricas resumen el desempeño del modelo. En regresión se revisan medidas como R², MAE y RMSE; "
        "en clasificación se revisan accuracy, precision, recall y F1."
    )

    pred_df = pd.DataFrame(res["predicciones"])
    if res["tipo_problema"] == "Regresión":
        st.plotly_chart(grafico_real_vs_predicho(pred_df["y_real"], pred_df["y_predicho"]), use_container_width=True)
        st.caption(
            "Esta gráfica compara valores observados con valores estimados. Mientras más cerca estén los puntos "
            "de la línea diagonal, mejor será la capacidad predictiva del modelo."
        )
    else:
        cm_records = res["matriz_confusion"]
        if cm_records:
            cm_df = pd.DataFrame(cm_records).set_index("index")
            st.plotly_chart(grafico_matriz_confusion(cm_df), use_container_width=True)
            st.caption(
                "La matriz de confusión muestra aciertos y errores de clasificación. Los valores de la diagonal "
                "principal son clasificaciones correctas; los valores fuera de la diagonal son confusiones entre clases."
            )

    if res["loss_curve"]:
        st.plotly_chart(grafico_loss_curve(res["loss_curve"]), use_container_width=True)
        st.caption(
            "La curva de pérdida muestra cómo cambia el error durante el entrenamiento. Una pérdida decreciente "
            "indica aprendizaje; si se estabiliza, el modelo pudo haber llegado a una zona de convergencia."
        )

    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        if st.button("Guardar resultados para resumen ejecutivo"):
            st.session_state["resultados_red_neuronal"] = {**res, "fecha_guardado": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            st.success("Resultados guardados.")
    with c2:
        excel = preparar_excel_generico(res, {"Metricas": pd.DataFrame([res["metricas"]]), "Predicciones": pred_df, "Loss": pd.DataFrame({"loss": res["loss_curve"]})})
        st.download_button("Descargar Excel", excel, "resultados_red_neuronal.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with c3:
        st.download_button("Descargar JSON", preparar_json_descarga(res), "resultados_red_neuronal.json", "application/json")
    with c4:
        figs = []
        if res["tipo_problema"] == "Regresión":
            figs.append(("Real vs. predicho", grafico_real_vs_predicho(pred_df["y_real"], pred_df["y_predicho"]), "Puntos cercanos a la diagonal indican mejor predicción."))
        else:
            cm_records = res["matriz_confusion"]
            if cm_records:
                cm_df_pdf = pd.DataFrame(cm_records).set_index("index")
                figs.append(("Matriz de confusión", grafico_matriz_confusion(cm_df_pdf), "La diagonal principal muestra aciertos."))
        if res["loss_curve"]:
            figs.append(("Evolución de la pérdida", grafico_loss_curve(res["loss_curve"]), "Una pérdida decreciente indica aprendizaje durante el entrenamiento."))
        pdf_bytes = crear_pdf_modulo(
            "Informe del módulo: Red neuronal",
            res,
            tablas=[("Métricas", pd.DataFrame([res["metricas"]]),), ("Predicciones", pred_df), ("Loss curve", pd.DataFrame({"loss": res["loss_curve"]}))],
            figuras=figs,
            notas=["El número de capas, neuronas y tasa de aprendizaje controlan la capacidad y dinámica del entrenamiento.", "Los resultados son pedagógicos y requieren validación para cualquier uso real."]
        )
        st.download_button("Descargar PDF", pdf_bytes, "informe_red_neuronal.pdf", "application/pdf")
    with c5:
        if st.button("Eliminar resultados guardados"):
            st.session_state.pop("resultados_red_neuronal", None)
            st.success("Resultados eliminados.")
