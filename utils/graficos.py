import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

def grafico_serie_interactivo(df: pd.DataFrame, y: str, titulo: str, x: str | None = None):
    if x and x in df.columns:
        fig = px.line(df, x=x, y=y, title=titulo, markers=False)
    else:
        temp = df[[y]].copy().reset_index().rename(columns={"index": "Observación"})
        fig = px.line(temp, x="Observación", y=y, title=titulo, markers=False)
    fig.update_layout(hovermode="x unified")
    return fig

def grafico_histograma_interactivo(serie: pd.Series, titulo: str):
    temp = pd.DataFrame({"Valor": pd.to_numeric(serie, errors="coerce").dropna()})
    fig = px.histogram(temp, x="Valor", nbins=30, title=titulo, marginal="box")
    fig.update_layout(bargap=0.05)
    return fig

def grafico_boxplot_interactivo(serie: pd.Series, titulo: str):
    temp = pd.DataFrame({"Valor": pd.to_numeric(serie, errors="coerce").dropna()})
    return px.box(temp, y="Valor", title=titulo, points="outliers")

def grafico_matriz_correlacion_interactiva(df: pd.DataFrame, titulo: str):
    corr = df.corr(numeric_only=True)
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
        colorscale="RdBu", reversescale=True, text=corr.round(3).values,
        texttemplate="%{text}",
        hovertemplate="Variable X: %{x}<br>Variable Y: %{y}<br>Correlación: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(title=titulo)
    return fig

def grafico_acf_pacf_interactivo(valores: list[dict], titulo: str, n_obs: int | None = None):
    if not valores:
        fig = go.Figure()
        fig.update_layout(title=f"{titulo} - no disponible")
        return fig
    df = pd.DataFrame(valores)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["rezago"], y=df["valor"], name=titulo,
        hovertemplate="Rezago: %{x}<br>Valor: %{y:.4f}<extra></extra>"
    ))
    fig.add_hline(y=0, line_width=1)
    if n_obs and n_obs > 0:
        banda = 1.96 / math.sqrt(n_obs)
        fig.add_hline(y=banda, line_dash="dash", line_width=1)
        fig.add_hline(y=-banda, line_dash="dash", line_width=1)
    fig.update_layout(title=titulo, xaxis_title="Rezago", yaxis_title="Correlación", bargap=0.15)
    return fig

def grafico_real_vs_predicho(y_true, y_pred, titulo="Real vs. predicho"):
    df = pd.DataFrame({"Valor real": y_true, "Valor predicho": y_pred})
    fig = px.scatter(df, x="Valor real", y="Valor predicho", title=titulo, trendline=None)
    min_v = min(df["Valor real"].min(), df["Valor predicho"].min())
    max_v = max(df["Valor real"].max(), df["Valor predicho"].max())
    fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", name="Línea 45°"))
    return fig

def grafico_residuales(residuales, titulo="Residuales"):
    df = pd.DataFrame({"Observación": range(len(residuales)), "Residual": residuales})
    fig = px.line(df, x="Observación", y="Residual", title=titulo)
    fig.add_hline(y=0)
    return fig

def grafico_residuales_vs_ajustados(ajustados, residuales, titulo="Residuales vs. ajustados"):
    df = pd.DataFrame({"Ajustado": ajustados, "Residual": residuales})
    fig = px.scatter(df, x="Ajustado", y="Residual", title=titulo)
    fig.add_hline(y=0)
    return fig

def grafico_qq(residuales, titulo="QQ plot de residuales"):
    resid = pd.Series(residuales).dropna()
    if len(resid) < 3:
        fig = go.Figure()
        fig.update_layout(title=f"{titulo} - no disponible")
        return fig
    osm, osr = stats.probplot(resid, dist="norm", fit=False)
    fig = px.scatter(x=osm, y=osr, title=titulo, labels={"x": "Cuantiles teóricos", "y": "Cuantiles observados"})
    slope, intercept, *_ = stats.linregress(osm, osr)
    fig.add_trace(go.Scatter(x=list(osm), y=[intercept + slope*x for x in osm], mode="lines", name="Referencia"))
    return fig

def grafico_coeficientes(tabla_coef, titulo="Coeficientes con intervalos de confianza"):
    df = tabla_coef.copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["coeficiente"], y=df["variable"], orientation="h", name="Coeficiente",
        error_x=dict(
            type="data",
            symmetric=False,
            array=df["ic_sup"] - df["coeficiente"],
            arrayminus=df["coeficiente"] - df["ic_inf"],
        )
    ))
    fig.update_layout(title=titulo, xaxis_title="Coeficiente", yaxis_title="Variable")
    return fig

def grafico_importancia_variables(importancias: pd.DataFrame, titulo="Importancia de variables"):
    return px.bar(importancias, x="importancia", y="variable", orientation="h", title=titulo)

def grafico_metricas(metricas: dict, titulo="Métricas del modelo"):
    df = pd.DataFrame({"métrica": list(metricas.keys()), "valor": list(metricas.values())})
    return px.bar(df, x="métrica", y="valor", title=titulo)

def grafico_matriz_confusion(cm_df: pd.DataFrame, titulo="Matriz de confusión"):
    fig = go.Figure(data=go.Heatmap(
        z=cm_df.values,
        x=cm_df.columns,
        y=cm_df.index,
        text=cm_df.values,
        texttemplate="%{text}",
        colorscale="Blues",
        hovertemplate="Predicho: %{x}<br>Real: %{y}<br>Conteo: %{z}<extra></extra>"
    ))
    fig.update_layout(title=titulo)
    return fig

def grafico_curva_roc(roc_df: pd.DataFrame, auc_val=None):
    titulo = "Curva ROC" if auc_val is None else f"Curva ROC - AUC={auc_val:.3f}"
    fig = px.line(roc_df, x="fpr", y="tpr", title=titulo, labels={"fpr":"FPR", "tpr":"TPR"})
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Azar"))
    return fig

def grafico_scree(varianza_df: pd.DataFrame):
    return px.bar(varianza_df, x="componente", y="varianza_explicada", title="Scree plot")

def grafico_varianza_acumulada(varianza_df: pd.DataFrame):
    return px.line(varianza_df, x="componente", y="varianza_acumulada", markers=True, title="Varianza explicada acumulada")

def grafico_pca_2d(scores_df: pd.DataFrame, color_col: str | None = None, titulo="Proyección PCA 2D"):
    if color_col and color_col in scores_df.columns:
        return px.scatter(scores_df, x="PC1", y="PC2", color=color_col, title=titulo, hover_data=scores_df.columns)
    return px.scatter(scores_df, x="PC1", y="PC2", title=titulo, hover_data=scores_df.columns)

def grafico_elbow(elbow_df: pd.DataFrame):
    return px.line(elbow_df, x="k", y="inercia", markers=True, title="Método del codo - K-Means")

def grafico_perfil_clusters(perfil_df: pd.DataFrame):
    df_long = perfil_df.reset_index().melt(id_vars=perfil_df.index.name or "cluster", var_name="variable", value_name="promedio")
    return px.bar(df_long, x="variable", y="promedio", color=perfil_df.index.name or "cluster", barmode="group", title="Perfil promedio por cluster")

def grafico_loss_curve(loss_curve):
    df = pd.DataFrame({"Iteración": range(1, len(loss_curve)+1), "Pérdida": loss_curve})
    return px.line(df, x="Iteración", y="Pérdida", title="Evolución de la pérdida")
