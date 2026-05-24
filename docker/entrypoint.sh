#!/bin/bash
set -e

echo "Iniciando Agente de Cálculo..."

# Verificar que los modelos están descargados
if [ ! -d "/app/models_cache" ]; then
    echo "Modelos no encontrados, descargando..."
    python download_models.py
fi

# Iniciar Streamlit
echo "Iniciando interfaz web en http://localhost:8501"
exec streamlit run src/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false