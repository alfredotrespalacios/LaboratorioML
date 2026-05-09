from datetime import datetime
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils.carga_datos import cargar_datos_modulo, detectar_columnas_numericas
from utils.exportacion import preparar_excel_generico, preparar_json_descarga
from utils.pdf_reportes import crear_pdf_modulo
from utils.graficos import grafico_pca_2d, grafico_scree, grafico_varianza_acumulada
from utils.estilos import mostrar_encabezado, caja_pedagogica

st.set_page_config(page_title="PCA", page_icon="🧭", layout="wide")
mostrar_encabezado("🧭 Reducción de dimensionalidad - PCA", "Reduce variables numéricas a componentes principales.")

caja_pedagogica("PCA resume información de muchas variables en componentes principales. Las variables se estandarizan antes del análisis.")

df, nombre_archivo, fuente_datos = cargar_datos_modulo("data/datos_pca.xlsx", "pca")
st.dataframe(df.head(20), use_container_width=True)

num_cols = detectar_columnas_numericas(df)
variables = st.sidebar.multiselect("Variables numéricas", num_cols, default=num_cols[:min(6, len(num_cols))])
if len(variables) < 2:
    st.warning("Seleccione al menos dos variables.")
    st.stop()

n_comp = st.sidebar.slider("Número de componentes", 2, min(len(variables), 10), min(2, len(variables)))

if "resultados_pca_actuales" not in st.session_state:
    st.session_state["resultados_pca_actuales"] = None

if st.button("Ejecutar PCA"):
    datos = df[variables].dropna()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(datos)
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(Xs)

    var_df = pd.DataFrame({
        "componente": [f"PC{i+1}" for i in range(n_comp)],
        "varianza_explicada": pca.explained_variance_ratio_,
        "varianza_acumulada": pca.explained_variance_ratio_.cumsum(),
    })

    loadings = pd.DataFrame(pca.components_.T, index=variables, columns=[f"PC{i+1}" for i in range(n_comp)]).reset_index().rename(columns={"index": "variable"})
    scores_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(n_comp)])

    st.session_state["resultados_pca_actuales"] = {
        "tipo_analisis": "PCA",
        "fecha_ejecucion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "archivo_usado": nombre_archivo,
        "fuente_datos": fuente_datos,
        "variables_utilizadas": variables,
        "filas_utilizadas": len(datos),
        "numero_componentes": n_comp,
        "varianza_explicada": var_df.to_dict(orient="records"),
        "cargas_componentes": loadings.to_dict(orient="records"),
        "scores": scores_df.to_dict(orient="records"),
    }

res = st.session_state["resultados_pca_actuales"]
if res:
    var_df = pd.DataFrame(res["varianza_explicada"])
    loadings = pd.DataFrame(res["cargas_componentes"])
    scores_df = pd.DataFrame(res["scores"])

    st.subheader("Varianza explicada")
    st.dataframe(var_df, use_container_width=True)
    st.plotly_chart(grafico_scree(var_df), use_container_width=True)
    st.plotly_chart(grafico_varianza_acumulada(var_df), use_container_width=True)

    st.subheader("Cargas de componentes")
    st.dataframe(loadings, use_container_width=True)

    st.subheader("Proyección PCA")
    if "PC1" in scores_df.columns and "PC2" in scores_df.columns:
        st.plotly_chart(grafico_pca_2d(scores_df), use_container_width=True)

    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        if st.button("Guardar resultados para resumen ejecutivo"):
            st.session_state["resultados_pca"] = {**res, "fecha_guardado": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            st.success("Resultados guardados.")
    with c2:
        excel = preparar_excel_generico(res, {"Varianza": var_df, "Cargas": loadings, "Scores": scores_df})
        st.download_button("Descargar Excel", excel, "resultados_pca.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with c3:
        st.download_button("Descargar JSON", preparar_json_descarga(res), "resultados_pca.json", "application/json")
    with c4:
        figs = [
            ("Scree plot", grafico_scree(var_df), "Muestra la varianza explicada por cada componente."),
            ("Varianza acumulada", grafico_varianza_acumulada(var_df), "Muestra cuánta información se acumula al agregar componentes."),
        ]
        if "PC1" in scores_df.columns and "PC2" in scores_df.columns:
            figs.append(("Proyección PCA 2D", grafico_pca_2d(scores_df), "Representa las observaciones en los dos primeros componentes."))
        pdf_bytes = crear_pdf_modulo(
            "Informe del módulo: PCA",
            res,
            tablas=[("Varianza explicada", var_df), ("Cargas de componentes", loadings), ("Scores", scores_df)],
            figuras=figs,
            notas=["PCA reduce dimensionalidad mediante combinaciones lineales de las variables originales.", "Las cargas indican el peso de cada variable en cada componente."]
        )
        st.download_button("Descargar PDF", pdf_bytes, "informe_pca.pdf", "application/pdf")
    with c5:
        if st.button("Eliminar resultados guardados"):
            st.session_state.pop("resultados_pca", None)
            st.success("Resultados eliminados.")
