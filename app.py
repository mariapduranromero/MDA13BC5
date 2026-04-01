# ============================================================
# CABECERA
# ============================================================
# Alumno: María Paula Durán
# URL Streamlit Cloud: https://mda13bc5-7ew4kwq3umsi9ztcmoenhc.streamlit.app
# URL GitHub: https://github.com/...

# ============================================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
# Modelo de OpenAI. No lo cambies.
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
# El system prompt es el conjunto de instrucciones que recibe el LLM
# ANTES de la pregunta del usuario. Define cómo se comporta el modelo:
# qué sabe, qué formato debe usar, y qué hacer con preguntas inesperadas.
#
# Puedes usar estos placeholders entre llaves — se rellenan automáticamente
# con información real del dataset cuando la app arranca:
#   {fecha_min}             → primera fecha del dataset
#   {fecha_max}             → última fecha del dataset
#   {plataformas}           → lista de plataformas (Android, iOS, etc.)
#   {reason_start_values}   → valores posibles de reason_start
#   {reason_end_values}     → valores posibles de reason_end
#
# IMPORTANTE: como el prompt usa llaves para los placeholders,
# si necesitas escribir llaves literales en el texto (por ejemplo para
# mostrar un JSON de ejemplo), usa doble llave: {{ y }}
#

SYSTEM_PROMPT = """
Eres un analista de datos experto en hábitos de escucha de Spotify.

OBJETIVO:
Convertir cada pregunta del usuario en un análisis sobre un DataFrame de pandas llamado `df` y devolver SIEMPRE un JSON válido.

FORMATO DE SALIDA (OBLIGATORIO):

1) Si la pregunta se puede responder con el dataset:
{{
  "tipo": "grafico",
  "codigo": "codigo Python ejecutable que crea una variable fig",
  "interpretacion": "explicación breve y clara del insight principal"
}}

2) Si la pregunta no puede responderse:
{{
  "tipo": "fuera_de_alcance",
  "codigo": "",
  "interpretacion": "explicación breve de por qué no puede responderse"
}}

REGLAS CRÍTICAS (ALTA PRIORIDAD):
- Devuelve SIEMPRE JSON válido.
- No uses markdown.
- No uses triple backticks.
- No añadas texto fuera del JSON.
- La clave `tipo` solo puede ser `grafico` o `fuera_de_alcance`.
- El contenido debe ser un JSON estrictamente válido y parseable con json.loads().
- Si no puedes generar código válido, devuelve `fuera_de_alcance`.


DATASET:

El DataFrame se llama `df`. Cada fila representa una reproducción.
Rango temporal: {fecha_min} a {fecha_max}

Columnas:
- ts, ms_played, reason_start, reason_end, shuffle, skipped, platform

Columnas derivadas:
- track_name: nombre de la canción
- artist_name: artista principal
- album_name: nombre del álbum
- track_uri: identificador único de la canción
- minutes_played: ms_played convertido a minutos
- hours_played: ms_played convertido a horas
- date: fecha de la reproducción
- year: año de la reproducción
- month_num: número de mes (1-12)
- month: nombre del mes
- year_month: mes en formato YYYY-MM para ordenar series temporales
- quarter: trimestre del año en formato Q1, Q2, Q3 o Q4
- hour: hora de la reproducción (basada en ts)
- weekday_num: día de la semana en número (lunes=0, domingo=6)
- weekday: nombre del día de la semana
- is_weekend: True si la reproducción fue sábado o domingo
- Nota: las variables de tiempo (hour, weekday, is_weekend) se calculan a partir del timestamp original en UTC del dataset
- skip_flag: True si la canción fue saltada; False en caso contrario
- season: estación del año asignada a partir del mes. Sus valores posibles son exactamente: Winter, Spring, Summer y Autumn
- is_first_listen: True si esa fila corresponde a la primera escucha registrada de esa canción en el dataset
- first_listen_ts: timestamp de la primera escucha registrada de cada canción
- first_listen_year: año de la primera escucha registrada
- first_listen_month_num: número de mes de la primera escucha registrada
- first_listen_month: nombre del mes de la primera escucha registrada
- first_listen_year_month: mes de la primera escucha registrada en formato YYYY-MM
- first_listen_month_label: etiqueta de la primera escucha en formato Mon YYYY
- first_listen_month_order: clave de orden YYYY-MM para ordenar meses de primera escucha

Valores observados:
- plataformas: {plataformas}
- reason_start: {reason_start_values}
- reason_end: {reason_end_values}

ALCANCE DEL ANÁLISIS:

A. Rankings:
- top artistas, canciones, álbumes
- más escuchado

B. Evolución temporal:
- por mes, trimestre, tiempo
- comparaciones temporales

C. Patrones:
- horas, días, fines de semana
- uso por plataforma

D. Comportamiento:
- skip, shuffle
- motivos de inicio/fin

E. Comparaciones:
- estaciones
- periodos
- Comparaciones entre periodos deben:
  1) calcular el top de cada periodo por separado
  2) limitar cada ranking al top N solicitado
  3) combinar los resultados en un único dataframe para visualización

FUERA DE ALCANCE:
Devuelve `fuera_de_alcance` si incluye:
- recomendaciones
- predicciones
- emociones o causas
- datos inexistentes
- conocimiento externo
- memoria conversacional
- combinación con otras fuentes


REGLAS DE INTERPRETACIÓN:

- "más escuchado" → hours_played
- "más reproducciones" → conteo de filas
- "canciones nuevas" → canciones cuya primera escucha registrada en el dataset (first_listen_ts) cae dentro del periodo analizado
- entre semana → lunes a viernes
- fin de semana → sábado y domingo
- estaciones → usar `season`
- porcentaje skip → skip_flag.mean() * 100
- Si la pregunta pide identificar un resultado principal, prioriza una respuesta que nombre explícitamente la categoría o periodo ganador.
- Si no hay evidencia suficiente para afirmar una diferencia clara, evita sobreinterpretar el resultado.
- Cuando la pregunta implique comparar periodos (ej. verano vs invierno, primer semestre vs segundo semestre), definir explícitamente cada periodo usando las variables disponibles (`season`, `month_num`, `year_month`, etc.)
- Calcular los resultados de cada periodo por separado antes de construir el gráfico
- Asegurar que la comparación sea consistente (mismas métricas y misma agregación)
- Si se comparan rankings entre periodos, calcular el top de cada periodo de forma independiente

Si es ambiguo:
→ elige la interpretación más estándar basada en el contexto del dataset

Solo devuelve `fuera_de_alcance` si:
- la pregunta requiere datos inexistentes en el dataset
- requiere conocimiento externo
- implica predicción o recomendación
- implica causalidad no observable en los datos
- requiere memoria de interacciones previas


REGLAS DE CÓDIGO:

- Usa solo: df, pd, px, go
- No importes librerías
- No uses archivos ni funciones externas
- Debe crear SIEMPRE `fig`
- No modificar `df` directamente
- Código claro, corto y robusto
- En comparaciones entre periodos, calcular cada periodo en dataframes separados antes de combinarlos
- Nunca usar import dentro del código generado
- Cuando uses variables categóricas derivadas como `season`, respeta exactamente los valores existentes en el dataset, incluyendo mayúsculas y escritura literal


Buenas prácticas:
- Agregar y ordenar cuando aplique
- Limitar categorías si hay demasiadas
- Definir explícitamente periodos
- Si el análisis usa `weekday`, ordenar siempre por `weekday_num` y no alfabéticamente
- Si el análisis usa `month`, ordenar siempre por `month_num` y no alfabéticamente
- Si el análisis usa series mensuales o comparaciones temporales, ordenar siempre por `year_month` o por la clave temporal correspondiente
- Si el análisis usa la variable `hour`, SIEMPRE debe: 1) agregar los datos por `hour`, 2) crear explícitamente un dataframe con las 24 horas de 0 a 23, 3) hacer un merge para incluir todas las horas, 4) rellenar con 0 las horas sin registros, 5) ordenar por `hour`, y 6) forzar en el eje X `tickmode="linear"` y `dtick=1`.
- calcular máximos, mínimos, líderes, top 1, top N y comparaciones principales en código, guardando explícitamente esos resultados en variables intermedias reutilizables para sostener la interpretación.
- Si la pregunta busca "el más", "la más", "top", "favorito", "más usado", "menos usado", "más escuchado" o equivalentes, identificar explícitamente la categoría líder o el resultado principal
- Si hay una comparación entre categorías, grupos o periodos, calcular explícitamente los resultados comparados antes de crear `fig`
- Si la pregunta requiere identificar un máximo, mínimo, hora pico, líder o ranking principal, calcular explícitamente ese resultado antes de construir `fig`
- La interpretación debe basarse en los resultados calculados en el código y no en una descripción visual aproximada del gráfico


REGLAS DE VISUALIZACIÓN:

- rankings → barras
- evolución → líneas
- distribuciones → barras
- evitar pie charts salvo que sea óptimo
- En gráficos por hora del día, mostrar explícitamente todas las horas de 0 a 23 en el eje X, aunque algunas tengan valor 0.
- En series temporales o secuencias discretas de tiempo, incluir todos los valores del periodo analizado en orden correcto aunque algunos no tengan registros.
- Cuando el eje X use `hour`, el gráfico debe mostrar explícitamente las 24 horas de 0 a 23 con una marca por cada hora.
- Cuando el eje X represente periodos como `year_month` o `first_listen_year_month`, forzar `type='category'` para evitar interpretaciones como fechas continuas
- En comparaciones de rankings entre periodos, usar gráficos agrupados (barras comparativas) donde cada periodo sea una serie distinta

Siempre:
- título
- ejes claros
- orden correcto
- sin gráficos vacíos


REGLAS DE INTERPRETACIÓN FINAL:

- 1 a 2 frases
- clara y ejecutiva
- basada en datos
- debe mencionar explícitamente las categorías, periodos o resultados principales detectados
- debe evitar frases genéricas o tautológicas como "es la más usada", "se observa una tendencia" o "muestra claramente la preferencia"
- cuando la pregunta implique identificar un máximo, mínimo, top 1, top N o comparación principal, debe nombrar explícitamente ese resultado
- si hay un líder claro, debe nombrarlo explícitamente y compararlo brevemente con las demás categorías relevantes
- si las diferencias son pequeñas o no concluyentes, debe indicarlo explícitamente sin exagerar
- no explicar código
- no inventar insights
- debe ser consistente con los valores calculados en el código
- no debe contradecir el resultado principal mostrado en el gráfico
- si menciona una categoría, periodo u hora como líder, debe corresponder al máximo calculado explícitamente
- La interpretación debe ser texto final ya resuelto, no debe contener código, variables, ni concatenaciones (no usar +, f-strings ni referencias a df)

EJEMPLOS:

- artista más escuchado
- top canciones
- evolución mensual
- hora más frecuente
- porcentaje skip
- comparación verano vs invierno


RECORDATORIO FINAL:
Devuelve SIEMPRE JSON válido y nada más.
"""


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
# Esta función se ejecuta UNA SOLA VEZ gracias a @st.cache_data.
# Lee el fichero JSON y prepara el DataFrame para que el código
# que genere el LLM sea lo más simple posible.
#
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

    # ----------------------------------------------------------
    # >>> TU PREPARACIÓN DE DATOS ESTÁ AQUÍ <<<
    # ----------------------------------------------------------
    # Transforma el dataset para facilitar el trabajo del LLM.
    # Lo que hagas aquí determina qué columnas tendrá `df`,
    # y tu system prompt debe describir exactamente esas columnas.
    #
    # Cosas que podrías considerar:
    # - Convertir 'ts' de string a datetime
    # - Crear columnas derivadas (hora, día de la semana, mes...)
    # - Convertir milisegundos a unidades más legibles
    # - Renombrar columnas largas para simplificar el código generado
    # - Filtrar registros que no aportan al análisis (podcasts, etc.)
    # ----------------------------------------------------------
    
    # 1. Convertir timestamp a datetime
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # 2. Ordenar cronológicamente
    df = df.sort_values("ts").reset_index(drop=True)

    # 3. Renombrar columnas largas para facilitar el código generado por el LLM
    df["track_name"] = df["master_metadata_track_name"]
    df["artist_name"] = df["master_metadata_album_artist_name"]
    df["album_name"] = df["master_metadata_album_album_name"]
    df["track_uri"] = df["spotify_track_uri"]

    # 4. Filtrar reproducciones musicales identificables
    # Esto evita que entren registros sin canción o artista a rankings y comparaciones
    df = df[df["track_name"].notna() & df["artist_name"].notna()].copy()

    # 5. Convertir milisegundos a métricas más legibles
    df["minutes_played"] = df["ms_played"] / 60000
    df["hours_played"] = df["ms_played"] / 3600000

    # 6. Crear variables temporales derivadas
    df["date"] = df["ts"].dt.date
    df["year"] = df["ts"].dt.year
    df["month_num"] = df["ts"].dt.month
    df["month"] = df["ts"].dt.month_name()
    df["year_month"] = df["ts"].dt.strftime("%Y-%m")
    df["quarter"] = "Q" + df["ts"].dt.quarter.astype(str)
    df["hour"] = df["ts"].dt.hour
    df["weekday_num"] = df["ts"].dt.weekday
    df["weekday"] = df["ts"].dt.day_name()
    df["is_weekend"] = (df["weekday_num"] >= 5)

    # 7. Crear variable clara de canciones saltadas
    # En este dataset, skipped = null implica que no fue saltada
    df["skip_flag"] = df["skipped"].fillna(False).astype(bool)

    # 8. Crear estación del año para comparaciones entre periodos
    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn"
    }
    df["season"] = df["month_num"].map(season_map)

    # 9. Variables de primera escucha para preguntas de "canciones nuevas"
    # Como el dataframe ya está ordenado por ts, la primera aparición de cada track_uri, representa el momento en que esa canción fue descubierta dentro del dataset.
    df["is_first_listen"] = ~df["track_uri"].duplicated()

    df["first_listen_ts"] = df.groupby("track_uri")["ts"].transform("min")
    df["first_listen_year"] = df["first_listen_ts"].dt.year
    df["first_listen_month_num"] = df["first_listen_ts"].dt.month
    df["first_listen_month"] = df["first_listen_ts"].dt.month_name()
    df["first_listen_year_month"] = df["first_listen_ts"].dt.strftime("%Y-%m")
    df["first_listen_month_label"] = df["first_listen_ts"].dt.strftime("%b %Y")
    df["first_listen_month_order"] = df["first_listen_ts"].dt.strftime("%Y-%m")
    
    return df


def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.
    Los valores que calcules aquí reemplazan a los placeholders
    {fecha_min}, {fecha_max}, etc. dentro de SYSTEM_PROMPT.

    Si añades columnas nuevas en load_data() y quieres que el LLM
    conozca sus valores posibles, añade aquí el cálculo y un nuevo
    placeholder en SYSTEM_PROMPT.
    """
    fecha_min = df["ts"].min()
    fecha_max = df["ts"].max()
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# Esta función envía DOS mensajes a la API de OpenAI:
# 1. El system prompt (instrucciones generales para el LLM)
# 2. La pregunta del usuario
#
# El LLM devuelve texto (que debería ser un JSON válido).
# temperature=0.2 hace que las respuestas sean más predecibles.
#
# No modifiques esta función.
#
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# El LLM devuelve un string que debería ser un JSON con esta forma:
#
#   {"tipo": "grafico",          "codigo": "...", "interpretacion": "..."}
#   {"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}
#
# Esta función convierte ese string en un diccionario de Python.
# Si el LLM envuelve el JSON en backticks de markdown (```json...```),
# los limpia antes de parsear.
#
# No modifiques esta función.
#
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# El LLM genera código Python como texto. Esta función lo ejecuta
# usando exec() y busca la variable `fig` que el código debe crear.
# `fig` debe ser una figura de Plotly (px o go).
#
# El código generado tiene acceso a: df, pd, px, go.
#
# No modifiques esta función.
#
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.
#

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
# Lee la contraseña de secrets.toml. Si no coincide, no muestra la app.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)

                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# Responde a estas tres preguntas con tus palabras. Sé concreto
# y haz referencia a tu solución, no a generalidades.
# No superes las 30 líneas en total entre las tres respuestas.
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    ¿Cómo funciona la arquitectura de tu aplicación? ¿Qué recibe
#    el LLM? ¿Qué devuelve? ¿Dónde se ejecuta el código generado?
#    ¿Por qué el LLM no recibe los datos directamente?
#
#    
#    La aplicación sigue una arquitectura text-to-code donde el LLM no analiza directamente los datos, 
#    sino que genera código a partir de una pregunta en lenguaje natural. El modelo recibe como entrada 
#    la pregunta del usuario y un system prompt con la descripción del dataset, las columnas disponibles y 
#    las reglas que debe seguir. Como salida, devuelve un JSON con tres campos: `tipo`, `codigo` e `interpretacion`. 
#    El código generado no se ejecuta en el modelo, sino localmente en la aplicación mediante `exec()`, utilizando el 
#    DataFrame `df` previamente cargado en Streamlit. El LLM no recibe los datos directamente para evitar enviar el 
#    dataset a la API, reducir riesgos de privacidad y asegurar que el análisis se realice de forma controlada y 
#    reproducible en el entorno local.
#
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    ¿Qué información le das al LLM y por qué? Pon un ejemplo
#    concreto de una pregunta que funciona gracias a algo específico
#    de tu prompt, y otro de una que falla o fallaría si quitases
#    una instrucción.
#
#    El system prompt es la pieza más importante de la solución porque define exactamente cómo debe comportarse 
#    el LLM antes de responder. En mi caso, le doy al modelo el formato obligatorio de salida en JSON, la descripción 
#    de las columnas originales y derivadas del DataFrame, reglas de interpretación de negocio y restricciones de código 
#    y visualización. Esto es clave para que el modelo no invente columnas, no devuelva texto fuera del JSON y genere 
#    análisis coherentes con el dataset. Un ejemplo que funciona gracias a una instrucción específica es “¿En qué mes 
#    descubrí más canciones nuevas?”, porque en el prompt definí `is_first_listen` y `first_listen_year_month`, lo que 
#    permite identificar primeras escuchas reales dentro del dataset. En cambio, la comparación “verano vs invierno” 
#    fallaba cuando no especificaba los valores exactos de `season`; al aclarar que los valores posibles son `Winter`, 
#    `Spring`, `Summer` y `Autumn`, el modelo empezó a filtrar correctamente y a construir la comparación de forma válida.
#
#
# 3. EL FLUJO COMPLETO
#    Describe paso a paso qué ocurre desde que el usuario escribe
#    una pregunta hasta que ve el gráfico en pantalla.
#
#    Cuando el usuario escribe una pregunta en la app, primero Streamlit carga y prepara el dataset con `load_data()`, 
#    donde se crean columnas derivadas que simplifican el análisis. Después, `build_prompt(df)` construye el system prompt 
#    insertando información dinámica como el rango temporal o los valores observados. La pregunta del usuario y el system 
#    prompt se envían al modelo mediante `get_response()`, que devuelve un JSON como texto. Este texto se convierte a 
#    diccionario con `parse_response()`. Si el tipo es `fuera_de_alcance`, la app muestra solo la explicación; si es `grafico`, 
#    el código generado se ejecuta con `execute_chart()` sobre el DataFrame `df`, creando la figura `fig`. Finalmente, 
#    Streamlit muestra el gráfico, la interpretación y el código generado en la interfaz.
