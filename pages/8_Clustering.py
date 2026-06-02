from datetime import datetime
import pandas as pd
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from utils.carga_datos import cargar_datos_modulo, detectar_columnas_numericas
from utils.exportacion import preparar_excel_generico, preparar_json_descarga
from utils.pdf_reportes import crear_pdf_modulo
from utils.graficos import grafico_elbow, grafico_pca_2d, grafico_perfil_clusters
from utils.estilos import mostrar_encabezado, caja_pedagogica

st.set_page_config(page_title="Clustering", page_icon="🧩", layout="wide")
mostrar_encabezado("🧩 Clustering", "Segmentación de observaciones con selección de algoritmo.")

caja_pedagogica("Permite formar grupos sin variable objetivo. La visualización 2D se realiza con PCA.")

df, nombre_archivo, fuente_datos = cargar_datos_modulo("data/datos_clustering.xlsx", "clustering")
st.dataframe(df.head(20), use_container_width=True)

num_cols = detectar_columnas_numericas(df)
variables = st.sidebar.multiselect("Variables para clustering", num_cols, default=num_cols[:min(5, len(num_cols))])
alg = st.sidebar.selectbox("Algoritmo", ["K-Means", "Agglomerative Clustering", "DBSCAN"])
n_clusters = st.sidebar.slider("Número de grupos", 2, 8, 3)
eps = st.sidebar.slider("DBSCAN eps", 0.1, 3.0, 0.7)
min_samples = st.sidebar.slider("DBSCAN min_samples", 2, 20, 5)

EXPLICACIONES_CLUSTERING = {
    "K-Means": """
    **K-Means** agrupa observaciones buscando centros o centroides. Cada observación se asigna al centro más cercano.
    El usuario debe definir el número de grupos. Es útil cuando se espera que los grupos sean relativamente compactos
    y separables en el espacio de variables.
    """,
    "Agglomerative Clustering": """
    **Agglomerative Clustering** construye grupos de forma jerárquica. Empieza tratando cada observación como un grupo
    separado y luego va uniendo los grupos más parecidos. Es útil para explorar estructuras jerárquicas o relaciones
    de cercanía entre observaciones.
    """,
    "DBSCAN": """
    **DBSCAN** agrupa observaciones según densidad. No requiere definir el número de grupos, pero sí parámetros como
    `eps` y `min_samples`. Puede identificar observaciones atípicas o ruido, marcadas usualmente como cluster `-1`.
    """
}

with st.expander("Explicación del método seleccionado", expanded=True):
    st.markdown(EXPLICACIONES_CLUSTERING[alg])

if len(variables) < 2:
    st.warning("Seleccione al menos dos variables.")
    st.stop()

if "resultados_clustering_actuales" not in st.session_state:
    st.session_state["resultados_clustering_actuales"] = None

if st.button("Ejecutar clustering"):
    datos = df[variables].dropna()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(datos)

    if alg == "K-Means":
        modelo = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = modelo.fit_predict(Xs)
    elif alg == "Agglomerative Clustering":
        modelo = AgglomerativeClustering(n_clusters=n_clusters)
        labels = modelo.fit_predict(Xs)
    else:
        modelo = DBSCAN(eps=eps, min_samples=min_samples)
        labels = modelo.fit_predict(Xs)

    resultados_df = datos.copy()
    resultados_df["cluster"] = labels.astype(str)
    perfil = resultados_df.groupby("cluster")[variables].mean()
    tamanos = resultados_df["cluster"].value_counts().sort_index().reset_index()
    tamanos.columns = ["cluster", "n_observaciones"]

    silhouette = None
    if len(set(labels)) > 1 and -1 not in set(labels):
        try:
            silhouette = float(silhouette_score(Xs, labels))
        except Exception:
            silhouette = None

    pca = PCA(n_components=2)
    scores = pca.fit_transform(Xs)
    scores_df = pd.DataFrame(scores, columns=["PC1", "PC2"])
    scores_df["cluster"] = labels.astype(str)

    elbow_df = pd.DataFrame()
    if alg == "K-Means":
        vals = []
        for k in range(1, 9):
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
            vals.append({"k": k, "inercia": km.inertia_})
        elbow_df = pd.DataFrame(vals)

    st.session_state["resultados_clustering_actuales"] = {
        "tipo_analisis": "Clustering",
        "fecha_ejecucion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "archivo_usado": nombre_archivo,
        "fuente_datos": fuente_datos,
        "algoritmo": alg,
        "variables_utilizadas": variables,
        "n_clusters": n_clusters if alg != "DBSCAN" else None,
        "eps": eps if alg == "DBSCAN" else None,
        "min_samples": min_samples if alg == "DBSCAN" else None,
        "silhouette": silhouette,
        "tamanos": tamanos.to_dict(orient="records"),
        "perfil_clusters": perfil.reset_index().to_dict(orient="records"),
        "resultados": resultados_df.reset_index(drop=True).to_dict(orient="records"),
        "scores_pca": scores_df.to_dict(orient="records"),
        "elbow": elbow_df.to_dict(orient="records"),
    }

res = st.session_state["resultados_clustering_actuales"]
if res:
    st.subheader("Métricas y tamaños")
    st.write(f"Silhouette: {res.get('silhouette')}")
    st.caption(
        "El coeficiente silhouette, cuando está disponible, resume qué tan separados y compactos están los grupos. "
        "Valores más altos suelen indicar una segmentación más clara."
    )

    tamanos = pd.DataFrame(res["tamanos"])
    perfil = pd.DataFrame(res["perfil_clusters"])
    resultados = pd.DataFrame(res["resultados"])
    scores = pd.DataFrame(res["scores_pca"])
    elbow = pd.DataFrame(res["elbow"])

    st.dataframe(tamanos, use_container_width=True)

    st.subheader("Perfil promedio por cluster")
    st.dataframe(perfil, use_container_width=True)
    if not perfil.empty:
        st.plotly_chart(grafico_perfil_clusters(perfil.set_index("cluster")), use_container_width=True)
        st.caption(
            "Este gráfico compara el promedio de cada variable dentro de cada cluster. Sirve para interpretar "
            "qué caracteriza a cada grupo y cómo se diferencian entre sí."
        )

    st.subheader("Proyección 2D con PCA")
    st.plotly_chart(grafico_pca_2d(scores, color_col="cluster", titulo="Clusters en espacio PCA"), use_container_width=True)
    st.caption(
        "Esta gráfica proyecta los datos en dos componentes principales para visualizar los grupos. Es una "
        "representación aproximada en 2D; los clusters se formaron usando las variables originales estandarizadas."
    )

    if not elbow.empty:
        st.plotly_chart(grafico_elbow(elbow), use_container_width=True)
        st.caption(
            "El método del codo ayuda a escoger un número razonable de clusters en K-Means. Se busca un punto "
            "donde la reducción de la inercia empiece a ser menos pronunciada."
        )

    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        if st.button("Guardar resultados para resumen ejecutivo"):
            st.session_state["resultados_clustering"] = {**res, "fecha_guardado": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            st.success("Resultados guardados.")
    with c2:
        excel = preparar_excel_generico(res, {"Tamanos": tamanos, "Perfil": perfil, "Resultados": resultados, "Scores_PCA": scores, "Elbow": elbow})
        st.download_button("Descargar Excel", excel, "resultados_clustering.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with c3:
        st.download_button("Descargar JSON", preparar_json_descarga(res), "resultados_clustering.json", "application/json")
    with c4:
        figs = [
            ("Perfil promedio por cluster", grafico_perfil_clusters(perfil.set_index("cluster")), "Compara características promedio entre grupos."),
            ("Clusters en espacio PCA", grafico_pca_2d(scores, color_col="cluster", titulo="Clusters en espacio PCA"), "Representación 2D aproximada de los grupos."),
        ]
        if not elbow.empty:
            figs.append(("Método del codo", grafico_elbow(elbow), "Ayuda a escoger un número razonable de clusters en K-Means."))
        pdf_bytes = crear_pdf_modulo(
            "Informe del módulo: Clustering",
            res,
            tablas=[("Tamaños de grupos", tamanos), ("Perfil promedio", perfil), ("Resultados", resultados), ("Scores PCA", scores), ("Elbow", elbow)],
            figuras=figs,
            notas=["Los clusters se forman con variables estandarizadas.", "La gráfica PCA es una proyección para visualizar, no el espacio completo usado por el algoritmo."]
        )
        st.download_button("Descargar PDF", pdf_bytes, "informe_clustering.pdf", "application/pdf")
    with c5:
        if st.button("Eliminar resultados guardados"):
            st.session_state.pop("resultados_clustering", None)
            st.success("Resultados eliminados.")
