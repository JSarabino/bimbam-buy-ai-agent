# Imagen base compatible con el entorno local definido para el proyecto
FROM python:3.11-slim

# Evita archivos .pyc y fuerza la salida inmediata de logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# curl se utiliza para comprobar el estado de Streamlit
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Las dependencias se copian primero para aprovechar la caché de Docker
COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

# Usuario sin privilegios para ejecutar la aplicación
RUN useradd --create-home --shell /bin/bash appuser

# Copia el proyecto dentro de la imagen
COPY --chown=appuser:appuser . .

# Garantiza que la aplicación pueda escribir el índice vectorial
RUN mkdir -p /app/storage \
    && chown -R appuser:appuser /app/storage

USER appuser

# Puerto predeterminado de Streamlit
EXPOSE 8501

# Comprobación de salud del contenedor
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Inicio de la aplicación
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]