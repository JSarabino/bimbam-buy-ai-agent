# Imagen base compatible con Python 3.11
FROM python:3.11-slim

# Configuración general del contenedor
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# curl se utiliza para el HEALTHCHECK de Streamlit
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencias en una capa separada para aprovechar la caché
COPY requirements.txt ./

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

# Usuario sin privilegios
RUN useradd \
    --create-home \
    --shell /bin/bash \
    appuser

# Copia el proyecto, incluido el índice FAISS generado localmente.
# .env permanece excluido mediante .dockerignore.
COPY --chown=appuser:appuser . .

# Prepara los directorios escribibles y falla durante el build
# cuando el índice requerido no fue incluido.
RUN mkdir -p \
        /app/storage/faiss_index \
        /app/storage/monitoring \
        /app/storage/maintenance \
        /app/storage/evaluation \
    && test -s /app/storage/faiss_index/index.faiss \
    && test -s /app/storage/faiss_index/documents.json \
    && test -s /app/storage/faiss_index/manifest.json \
    && test -s /app/storage/faiss_index/corpus_manifest.json \
    && chown -R appuser:appuser /app/storage

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
