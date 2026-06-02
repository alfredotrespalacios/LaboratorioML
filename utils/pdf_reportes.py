from __future__ import annotations

from io import BytesIO
from typing import Any
import numbers

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

MAX_ROWS_TABLE = 35
MAX_COLS_TABLE = 8
MAX_CELL_CHARS = 80


def _safe_text(value: Any, max_chars: int = MAX_CELL_CHARS) -> str:
    """
    Formats values for PDF tables.
    Numeric values are shown with a maximum of four decimals.
    Very small/large numeric values use scientific notation with four decimals.
    """
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    # Format numeric values with maximum four decimals.
    # This includes Python numeric types and numpy numeric types.
    try:
        if isinstance(value, numbers.Number) and not isinstance(value, bool):
            value_float = float(value)

            if value_float == 0:
                return "0"

            abs_value = abs(value_float)
            if abs_value < 0.0001 or abs_value >= 1_000_000:
                return f"{value_float:.4e}"

            text = f"{value_float:.4f}".rstrip("0").rstrip(".")
            return text
    except Exception:
        pass

    text = str(value)
    text = text.replace("\n", " ")
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def _df_to_table_data(df: pd.DataFrame, max_rows: int = MAX_ROWS_TABLE, max_cols: int = MAX_COLS_TABLE):
    if df is None or df.empty:
        return [["Sin datos"]]

    work = df.copy()
    if work.shape[1] > max_cols:
        work = work.iloc[:, :max_cols].copy()
        work["..."] = "más columnas"

    if work.shape[0] > max_rows:
        work = work.head(max_rows).copy()
        work.loc[len(work)] = ["..."] * work.shape[1]

    headers = [_safe_text(c, 35) for c in work.columns]
    rows = [[_safe_text(v) for v in row] for row in work.values.tolist()]
    return [headers] + rows


def _add_dataframe(story, title: str, df: pd.DataFrame, styles):
    story.append(Paragraph(title, styles["Heading2"]))
    data = _df_to_table_data(df)
    table = Table(data, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.18 * inch))


def _fig_to_image_flowable(fig, max_width=6.8 * inch, max_height=4.2 * inch):
    try:
        img_bytes = fig.to_image(format="png", width=1100, height=650, scale=2)
        bio = BytesIO(img_bytes)
        image = Image(bio)
        image._restrictSize(max_width, max_height)
        return image
    except Exception as exc:
        return Paragraph(
            f"No fue posible exportar esta gráfica al PDF. Revise que kaleido esté instalado. Detalle: {_safe_text(exc, 160)}",
            getSampleStyleSheet()["BodyText"],
        )


def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(0.6 * inch, 0.35 * inch, "Laboratorio ML e IA para Finanzas - uso exclusivamente pedagógico")
    canvas.drawRightString(7.9 * inch, 0.35 * inch, f"Página {doc.page}")
    canvas.restoreState()


def crear_pdf_modulo(
    titulo: str,
    resultados: dict,
    tablas: list[tuple[str, pd.DataFrame]] | None = None,
    figuras: list[tuple[str, Any, str]] | None = None,
    notas: list[str] | None = None,
) -> bytes:
    """
    Crea un PDF pedagógico con metadatos, tablas y gráficas estáticas exportadas desde Plotly.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleCenter",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontSize=16,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallNote",
            parent=styles["BodyText"],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#444444"),
        )
    )

    story = []
    story.append(Paragraph("Laboratorio de Machine Learning e Inteligencia Artificial para Finanzas", styles["TitleCenter"]))
    story.append(Paragraph(titulo, styles["Heading1"]))
    story.append(Paragraph("Aplicación creada por Alfredo Trespalacios como complemento a las memorias del curso.", styles["BodyText"]))
    story.append(Spacer(1, 0.08 * inch))
    story.append(
        Paragraph(
            "<b>Aclaración:</b> este reporte tiene únicamente fines pedagógicos. No se recomienda ni se autoriza "
            "su uso en actividades profesionales, decisiones empresariales, decisiones financieras, consultoría, "
            "valoración, predicción operativa o toma de decisiones reales sin validación técnica independiente.",
            styles["SmallNote"],
        )
    )
    story.append(Spacer(1, 0.18 * inch))

    metadatos = {
        "Tipo de análisis": resultados.get("tipo_analisis", titulo),
        "Fecha de ejecución": resultados.get("fecha_ejecucion", ""),
        "Fecha de guardado": resultados.get("fecha_guardado", ""),
        "Archivo usado": resultados.get("archivo_usado", ""),
        "Fuente de datos": resultados.get("fuente_datos", ""),
    }

    extra_keys = [
        "algoritmo",
        "tipo_problema",
        "variable_objetivo",
        "variable_dependiente_usada",
        "variables_analizadas",
        "variables_explicativas",
        "variables_utilizadas",
        "filas_utilizadas",
    ]
    for key in extra_keys:
        if key in resultados:
            metadatos[key.replace("_", " ").capitalize()] = resultados.get(key)

    meta_df = pd.DataFrame({"Campo": list(metadatos.keys()), "Valor": [_safe_text(v, 140) for v in metadatos.values()]})
    _add_dataframe(story, "Metadatos del análisis", meta_df, styles)

    if notas:
        story.append(Paragraph("Notas de interpretación", styles["Heading2"]))
        for nota in notas:
            story.append(Paragraph(f"- {_safe_text(nota, 500)}", styles["BodyText"]))
        story.append(Spacer(1, 0.12 * inch))

    if tablas:
        for title, df in tablas:
            _add_dataframe(story, title, df, styles)

    if figuras:
        story.append(PageBreak())
        story.append(Paragraph("Gráficas", styles["Heading1"]))
        for title, fig, caption in figuras:
            bloque = [
                Paragraph(title, styles["Heading2"]),
                _fig_to_image_flowable(fig),
                Paragraph(_safe_text(caption, 600), styles["SmallNote"]),
                Spacer(1, 0.20 * inch),
            ]
            story.append(KeepTogether(bloque))

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return buffer.getvalue()
