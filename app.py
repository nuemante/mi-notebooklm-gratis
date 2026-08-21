import os
import time
import streamlit as st
from google import genai
from google.genai import types

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
        # Modelo Gemini 2.5 Flash (Gratuito permanente en Google AI Studio)
        self.model_name = "gemini-2.5-flash"

    def upload_file(self, file_path: str):
        """Sube el archivo a la File API de Gemini"""
        file_ref = self.client.files.upload(file=file_path)
        while file_ref.state.name == "PROCESSING":
            time.sleep(1)
            file_ref = self.client.files.get(name=file_ref.name)
        if file_ref.state.name == "FAILED":
            raise ValueError(f"Error procesando el archivo: {file_ref.name}")
        return file_ref

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
    with st.spinner("Cargando y procesando fuentes en Gemini..."):
        for uploaded_file in uploaded_files:
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            file_ref = notebook.upload_file(temp_path)
            st.session_state.uploaded_files_ref.append(file_ref)
            os.remove(temp_path)
            
    st.sidebar.success("¡Fuentes procesadas con éxito!")

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
                    respuesta = notebook.query_notebook(
                        st.session_state.uploaded_files_ref, prompt, history_str
                    )
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
                summary = notebook.generate_summary(st.session_state.uploaded_files_ref)
                st.markdown(summary)

# PESTAÑA 3: PODCAST / AUDIO OVERVIEW
with tab_podcast:
    st.subheader("Guion de Podcast")
    if st.button("🎙️ Generar Guion de Discusión"):
        if not st.session_state.uploaded_files_ref:
            st.warning("⚠️ Debes cargar fuentes en la barra lateral primero.")
        else:
            with st.spinner("Creando diálogo..."):
                script = notebook.generate_podcast_script(st.session_state.uploaded_files_ref)
                st.markdown(script)
