# Imagen base con Python 3.10
FROM python:3.10-slim

# Metadata
LABEL maintainer="Agente Cálculo"
LABEL description="Sistema de QA sobre PDFs de cálculo - 100% offline"

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar configuración
COPY config.yaml .

# Copiar script de descarga de modelos
COPY download_models.py .

# Descargar modelos durante el build (esto hace la imagen offline)
RUN python download_models.py

# Copiar código fuente
COPY src/ ./src/
COPY tests/ ./tests/

# Crear directorios necesarios
RUN mkdir -p data chroma_db

# Exponer puerto de Streamlit
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501')"

# Comando de inicio
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]