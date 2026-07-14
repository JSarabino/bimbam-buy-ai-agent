"""Punto de entrada de la aplicación Streamlit de BimBam Assistant."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


# Carga las variables del archivo .env cuando se ejecuta localmente.
# En Docker u OCI también pueden proporcionarse como variables del sistema.
load_dotenv()


def env_to_bool(value: str | None) -> bool:
    """Determina si una variable contiene un valor útil."""

    return bool(value and value.strip())


def index_is_ready(index_path: Path) -> bool:
    """Comprueba si existen los archivos principales de un índice FAISS."""

    return (
        (index_path / "index.faiss").is_file()
        and (index_path / "index.pkl").is_file()
    )


def main() -> None:
    """Renderiza la pantalla inicial de la aplicación."""

    app_name = os.getenv("APP_NAME", "BimBam Assistant")
    app_environment = os.getenv("APP_ENV", "development")
    app_version = os.getenv("APP_VERSION", "0.1.0")

    documents_path = Path(
        os.getenv("DOCUMENTS_PATH", "data/documents")
    )
    faiss_index_path = Path(
        os.getenv("FAISS_INDEX_PATH", "storage/faiss_index")
    )

    api_key_configured = env_to_bool(os.getenv("GOOGLE_API_KEY"))

    pdf_files = (
        sorted(documents_path.glob("*.pdf"))
        if documents_path.is_dir()
        else []
    )

    faiss_ready = index_is_ready(faiss_index_path)

    st.set_page_config(
        page_title=app_name,
        page_icon="🛍️",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    st.title(f"🛍️ {app_name}")
    st.caption(
        "Agente inteligente para consultar las políticas y los "
        "documentos corporativos de BimBam Buy."
    )

    st.markdown(
        """
        Esta versión inicial verifica que el entorno, los documentos y
        la aplicación web estén correctamente configurados.

        En las siguientes etapas incorporaremos:

        - lectura y fragmentación de los PDF;
        - embeddings con Gemini;
        - almacenamiento vectorial con FAISS;
        - recuperación semántica;
        - generación de respuestas con fuentes;
        - flujo del agente con LangGraph.
        """
    )

    st.subheader("Estado del proyecto")

    column_api, column_documents, column_index = st.columns(3)

    with column_api:
        st.metric(
            label="Gemini API",
            value="Configurada" if api_key_configured else "Pendiente",
        )

    with column_documents:
        st.metric(
            label="Documentos PDF",
            value=len(pdf_files),
        )

    with column_index:
        st.metric(
            label="Índice FAISS",
            value="Disponible" if faiss_ready else "Pendiente",
        )

    if not api_key_configured:
        st.warning(
            "La variable GOOGLE_API_KEY todavía no está configurada. "
            "Agrégala al archivo .env para conectar la aplicación con Gemini."
        )

    if not documents_path.is_dir():
        st.warning(
            f"No se encontró la carpeta de documentos: "
            f"`{documents_path.as_posix()}`."
        )
    elif not pdf_files:
        st.warning(
            f"No se encontraron archivos PDF en "
            f"`{documents_path.as_posix()}`."
        )
    else:
        st.success(
            f"Se encontraron {len(pdf_files)} documentos PDF."
        )

        with st.expander("Ver documentos detectados"):
            for pdf_file in pdf_files:
                st.write(f"• {pdf_file.name}")

    if not faiss_ready:
        st.info(
            "El índice FAISS todavía no ha sido generado. "
            "Esto es normal en la configuración inicial."
        )

    st.subheader("Consulta documental")

    st.text_input(
        label="Escribe una pregunta sobre BimBam Buy",
        placeholder="Ejemplo: ¿Cuánto tarda un reembolso?",
        disabled=True,
        help=(
            "El campo se habilitará cuando implementemos "
            "el servicio RAG."
        ),
    )

    with st.sidebar:
        st.header("Configuración")

        st.write(f"**Aplicación:** {app_name}")
        st.write(f"**Versión:** {app_version}")
        st.write(f"**Entorno:** {app_environment}")
        st.write(
            f"**Modelo:** "
            f"{os.getenv('GEMINI_CHAT_MODEL', 'gemini-3.5-flash')}"
        )
        st.write(
            f"**Embeddings:** "
            f"{os.getenv('GEMINI_EMBEDDING_MODEL', 'gemini-embedding-001')}"
        )

        st.divider()

        st.caption(
            "La clave de Gemini nunca se muestra en la interfaz."
        )


if __name__ == "__main__":
    main()