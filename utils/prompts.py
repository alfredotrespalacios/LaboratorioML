import json

def resultados_a_texto(resultados: dict) -> str:
    return json.dumps(resultados, ensure_ascii=False, indent=2, default=str)

def construir_prompt_integrado(
    resultados_cargados: dict,
    objetivo: str,
    descripcion_datos: str,
    audiencia: str,
    tono: str,
    extension: str,
    nivel_tecnico: str,
    links_contexto: list[str] | None = None,
    preguntas_clave: list[str] | None = None,
) -> str:
    bloques_resultados = []
    for nombre_analisis, resultados in resultados_cargados.items():
        bloques_resultados.append(f"""
### {nombre_analisis}

{resultados_a_texto(resultados)}
""")

    resultados_texto = "\n".join(bloques_resultados)

    links_contexto = [x.strip() for x in (links_contexto or []) if str(x).strip()]
    preguntas_clave = [x.strip() for x in (preguntas_clave or []) if str(x).strip()]

    bloque_links = ""
    if links_contexto:
        bloque_links = "\nLinks web de contexto aportados por el usuario:\n" + "\n".join([f"- {x}" for x in links_contexto])

    bloque_preguntas = ""
    instruccion_respuestas = ""
    if preguntas_clave:
        bloque_preguntas = "\nPreguntas del usuario para responder en la sección “Respuestas clave”:\n" + "\n".join([f"{i+1}. {p}" for i, p in enumerate(preguntas_clave)])
        instruccion_respuestas = """
9. Además del informe ejecutivo, incluye una sección llamada “Respuestas clave”, donde respondas de forma directa y argumentada las preguntas planteadas por el usuario. Cada pregunta debe aparecer como subtítulo dentro de esa sección.
"""

    prompt = f"""
Actúa como un analista cuantitativo senior y redacta un informe ejecutivo claro, sobrio y útil.

Objetivo del informe:
{objetivo}

Descripción breve de los datos:
{descripcion_datos}

Audiencia del informe:
{audiencia}

Tono deseado:
{tono}

Extensión deseada:
{extension}

Nivel técnico:
{nivel_tecnico}
{bloque_links}
{bloque_preguntas}

Resultados técnicos disponibles:
{resultados_texto}

Instrucciones:
1. Redacta un informe ejecutivo integrando los resultados disponibles.
2. No repitas mecánicamente las tablas; interpreta los hallazgos.
3. Conecta los resultados entre sí cuando sea posible.
4. Distingue entre descripción, asociación estadística, predicción, probabilidad, reducción de dimensionalidad y segmentación.
5. No afirmes causalidad si los resultados no la soportan.
6. Señala limitaciones metodológicas cuando existan.
7. Interpreta con prudencia pruebas de normalidad, raíz unitaria, ACF, PACF, residuales y métricas de modelo.
8. No inventes resultados, variables, cifras ni conclusiones.
{instruccion_respuestas}

Estructura solicitada:
1. Resumen ejecutivo
2. Objetivo del análisis
3. Descripción de los datos
4. Principales hallazgos
5. Interpretación integrada
6. Limitaciones
7. Recomendaciones
8. Respuestas clave, solo si el usuario formuló preguntas
9. Conclusión

Redacta en español profesional.
"""
    return prompt.strip()
