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
) -> str:
    bloques_resultados = []

    for nombre_analisis, resultados in resultados_cargados.items():
        bloques_resultados.append(f"""
### {nombre_analisis}

{resultados_a_texto(resultados)}
""")

    resultados_texto = "\n".join(bloques_resultados)

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

Resultados técnicos disponibles:
{resultados_texto}

Instrucciones:
1. Redacta un informe ejecutivo integrando los resultados disponibles.
2. No repitas mecánicamente las tablas; interpreta los hallazgos.
3. Conecta los resultados entre sí cuando sea posible.
4. Distingue entre descripción, asociación estadística, predicción, reducción de dimensionalidad y segmentación.
5. No afirmes causalidad si los resultados no la soportan.
6. Señala limitaciones metodológicas cuando existan.
7. Interpreta con prudencia pruebas de normalidad, raíz unitaria, ACF, PACF, residuales y métricas de modelo.
8. No inventes resultados, variables, cifras ni conclusiones.

Estructura solicitada:
1. Resumen ejecutivo
2. Objetivo del análisis
3. Descripción de los datos
4. Principales hallazgos
5. Interpretación integrada
6. Limitaciones
7. Recomendaciones
8. Conclusión

Redacta en español profesional.
"""
    return prompt.strip()
