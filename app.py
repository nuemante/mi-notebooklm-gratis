import os
import time
import tempfile
import traceback
import streamlit as st
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA E INTERFAZ
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Cuaderno Inteligente & Chatbot IA",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Cuaderno Inteligente & Chatbot 100% Gratuito")
st.caption("Sube tus documentos (PDF, TXT, MD), realiza consultas en tiempo real, genera resúmenes y crea guiones de podcast.")

# -----------------------------------------------------------------------------
# 2. MOTOR DE IA CON GEMINI FREE TIER
# -----------------------------------------------------------------------------
class GeminiGratuitoNotebook:
    def __init__(self, api_key: str):
        # Inicializar el cliente oficial de Google GenAI
        self.client = genai.Client(api_key=api_key)
        # Modelo Gemini 3.6 Flash (reemplaza a 2.5 Flash, descontinuado para proyectos nuevos)
        self.model_name = "gemini-3.6-flash"

    def upload_file(self, file_path: str, max_retries: int = 3):
        """Sube el archivo a la File API de Gemini, con reintentos y detalle de error real."""
        last_error = None
        for intento in range(1, max_retries + 1):
            try:
                file_ref = self.client.files.upload(file=file_path)
                while file_ref.state.name == "PROCESSING":
                    time.sleep(1)
                    file_ref = self.client.files.get(name=file_ref.name)
                if file_ref.state.name == "FAILED":
                    raise ValueError(f"Error procesando el archivo: {file_ref.name}")
                return file_ref

            except genai_errors.ClientError as e:
                last_error = e
                status = getattr(e, "status_code", None) or getattr(e, "code", None)
                detalle = getattr(e, "message", None) or str(e)

                # Errores de cuota (429) o de servidor: reintentar con espera creciente
                if status in (429, 500, 503) and intento < max_retries:
                    espera = 3 * intento
                    st.warning(
                        f"⚠️ Intento {intento}/{max_retries} falló (código {status}). "
                        f"Reintentando en {espera}s..."
                    )
                    time.sleep(espera)
                    continue

                # Error no recuperable, o se agotaron los reintentos: mostrar detalle real
                st.error(f"❌ Error subiendo '{os.path.basename(file_path)}' — código: {status}")
                st.code(detalle)
                raise

            except Exception as e:
                last_error = e
                st.error(f"❌ Error inesperado subiendo '{os.path.basename(file_path)}': {type(e).__name__}")
                st.code(str(e))
                st.code(traceback.format_exc())
                raise

        if last_error:
            raise last_error

    def generate_summary(self, files: list) -> str:
        """Genera un resumen ejecutivo de los documentos cargados"""
        prompt = """
        Actúa como NotebookLM. Analiza las fuentes de información proporcionadas y genera:
        1. Resumen ejecutivo detallado.
        2. Los 5 conceptos o temas clave más relevantes.
        3. 3 preguntas sugeridas que el usuario puede hacer sobre este material.
        """
        response = self.client.models.generate_content(
            model=self.model_name, contents=files + [prompt]
        )
        return response.text

    def query_notebook(self, files: list, question: str, chat_history: list) -> str:
        """Responde preguntas basándose estrictamente en las fuentes"""
        system_instruction = """
        Eres un asistente de investigación preciso estilo NotebookLM.
        Responde ÚNICAMENTE basándote en la información de los documentos subidos.
        Si la respuesta no está en el texto, indica amablemente que la información no figura en las fuentes suministradas.
        """
        context = files + [system_instruction, "Historial de chat previo:"] + chat_history + [f"Pregunta del usuario: {question}"]

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=context,
            config=types.GenerateContentConfig(
                temperature=0.2
            )
        )
        return response.text

    def generate_podcast_script(self, files: list) -> str:
        """Crea un guion conversacional estilo Audio Overview"""
        prompt = """
        Crea un guion de podcast dinámico y educativo estilo 'Audio Overview' de NotebookLM.
        Debe ser una conversación amena entre dos locutores (Alex y Sonia) analizando los documentos.
        Formato:
        Alex: [Texto]
        Sonia: [Texto]
        """
        response = self.client.models.generate_content(
            model=self.model_name, contents=files + [prompt]
        )
        return response.text

# -----------------------------------------------------------------------------
# 3. GESTIÓN DE CREDENCIALES
# -----------------------------------------------------------------------------
# Obtener API Key desde los Secrets de Streamlit o pedirla si no existe
api_key = st.secrets.get("GEMINI_API_KEY", None)

if not api_key:
    api_key = st.sidebar.text_input(
        "Ingresa tu Gemini API Key (Gratis):",
        type="password",
        help="Obtén tu clave de costo $0 en https://aistudio.google.com/"
    )

if not api_key:
    st.info("👈 Por favor, configura tu API Key en los Secrets de Streamlit Cloud o ingresala en la barra lateral para continuar.")
    st.stop()

# Conectar cliente
try:
    notebook = GeminiGratuitoNotebook(api_key=api_key)
except Exception as e:
    st.error(f"Error al conectar con la API de Gemini: {e}")
    st.stop()

# Inicializar memoria de sesión
if "uploaded_files_ref" not in st.session_state:
    st.session_state.uploaded_files_ref = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------------------------------------------------------
# 4. BARRA LATERAL: CARGA DE ARCHIVOS
# -----------------------------------------------------------------------------
st.sidebar.header("📁 Carga de Documentos")
uploaded_files = st.sidebar.file_uploader(
    "Sube tus documentos (PDF, TXT, MD)",
    accept_multiple_files=True,
    type=["pdf", "txt", "md"]
)

if uploaded_files and st.sidebar.button("📂 Procesar Fuentes"):
    st.session_state.uploaded_files_ref = []
    exitosos = 0
    fallidos = []

    with st.spinner("Cargando y procesando fuentes en Gemini..."):
        for idx, uploaded_file in enumerate(uploaded_files):
            temp_path = os.path.join(tempfile.gettempdir(), f"temp_{uploaded_file.name}")
            try:
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                file_ref = notebook.upload_file(temp_path)
                st.session_state.uploaded_files_ref.append(file_ref)
                exitosos += 1

                # Pequeña pausa de cortesía entre archivos para no saturar el free tier
                if idx < len(uploaded_files) - 1:
                    time.sleep(2)

            except Exception:
                fallidos.append(uploaded_file.name)
                continue
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    if exitosos:
        st.sidebar.success(f"✅ {exitosos} fuente(s) procesada(s) con éxito.")
    if fallidos:
        st.sidebar.error(f"❌ No se pudieron procesar: {', '.join(fallidos)}")

# -----------------------------------------------------------------------------
# 5. PESTAÑAS PRINCIPALES DE LA APLICACIÓN
# -----------------------------------------------------------------------------
tab_chat, tab_summary, tab_podcast = st.tabs(["💬 Chatbot", "📑 Resumen Automático", "🎙️ Audio Overview (Podcast)"])

# PESTAÑA 1: CHATBOT INTERACTIVO
with tab_chat:
    st.subheader("Chatbot con tus Documentos")

    # Mostrar conversación previa
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Entrada del usuario
    if prompt := st.chat_input("Escribe tu pregunta sobre las fuentes..."):
        if not st.session_state.uploaded_files_ref:
            st.warning("⚠️ Primero sube y procesa un documento desde el menú lateral.")
        else:
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.spinner("Consultando fuentes..."):
                    history_str = [f"{m['role']}: {m['content']}" for m in st.session_state.messages[:-1]]
                    try:
                        respuesta = notebook.query_notebook(
                            st.session_state.uploaded_files_ref, prompt, history_str
                        )
                    except Exception as e:
                        respuesta = f"⚠️ Ocurrió un error al consultar: {e}"
                    st.markdown(respuesta)

            st.session_state.messages.append({"role": "assistant", "content": respuesta})

# PESTAÑA 2: RESUMEN AUTOMÁTICO
with tab_summary:
    st.subheader("Análisis de Fuentes")
    if st.button("✨ Generar Resumen General"):
        if not st.session_state.uploaded_files_ref:
            st.warning("⚠️ Debes cargar fuentes en la barra lateral primero.")
        else:
            with st.spinner("Generando resumen..."):
                try:
                    summary = notebook.generate_summary(st.session_state.uploaded_files_ref)
                    st.markdown(summary)
                except Exception as e:
                    st.error(f"Error generando el resumen: {e}")

# PESTAÑA 3: PODCAST / AUDIO OVERVIEW
with tab_podcast:
    st.subheader("Guion de Podcast")
    if st.button("🎙️ Generar Guion de Discusión"):
        if not st.session_state.uploaded_files_ref:
            st.warning("⚠️ Debes cargar fuentes en la barra lateral primero.")
        else:
            with st.spinner("Creando diálogo..."):
                try:
                    script = notebook.generate_podcast_script(st.session_state.uploaded_files_ref)
                    st.markdown(script)
                except Exception as e:
                    st.error(f"Error generando el guion: {e}")
