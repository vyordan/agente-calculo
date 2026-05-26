[![Docker Image Size](https://img.shields.io/docker/image-size/vyordan/agente-calculo?label=Image%20Size)](https://hub.docker.com/r/vyordan/agente-calculo)
[![Docker Pulls](https://img.shields.io/docker/pulls/vyordan/agente-calculo?label=Pulls)](https://hub.docker.com/r/vyordan/agente-calculo)

# Agente de Cálculo - QA sobre PDFs

Sistema de preguntas y respuestas (QA) para libros y documentos de cálculo.  
Permite cargar un PDF, indexarlo semánticamente y realizar consultas en lenguaje natural, obteniendo respuestas extractivas y una generativas. Además resuelve automáticamente derivadas, integrales y límites usando **SymPy**.

## Características principales

- **Interfaz web** con Streamlit (puerto 8501).
- **Procesamiento offline**: todos los modelos se descargan durante la construcción y no requieren conexión a internet.
- **Pipeline completo**:
  1. Lectura y limpieza de PDFs (`pdfplumber`).
  2. División inteligente en chunks (`RecursiveCharacterTextSplitter`).
  3. Embeddings semánticos (`all-MiniLM-L6-v2`).
  4. Almacenamiento vectorial (`ChromaDB`).
  5. QA extractivo (`distilbert-base-cased-distilled-squad`) y generativo (`google/flan-t5-base`).
- **Detección automática de problemas matemáticos** (derivadas, integrales, límites) y resolución simbólica.
- **Configuración flexible** mediante `config.yaml`.

## Requisitos

- Python 3.10+
- Docker (opcional, pero recomendado)

## Uso rápido con Docker

```bash
# Construir la imagen (descarga modelos automáticamente)
docker build -t agente-calculo -f docker/Dockerfile .

# Ejecutar el contenedor
docker run -p 8501:8501 agente-calculo
```
Luego abre http://localhost:8501 en tu navegador.

## Uso manual (sin Docker)
Antes de ejecutar los siguientes ocmandos se sugiere usar un entorno virtual de python :) es buena practica
```bash
# Instalar dependencias
pip install -r requirements.txt

# Descargar los modelos
python download_models.py

# Iniciar la applicacion, una vez se tienen las dependencias y los
#modelos descargados solo es necesario este comando para volver a ejecutar
streamlit run src/app.py 
```
### Estructura
agente-calculo/
├── src/ # Código fuente (app, chunker, embedder, qa_engine, solver…)
├── tests/ # Tests unitarios
├── docker/ # Dockerfile + entrypoint
├── docker-compose.yml 
├── download_models.py # Descarga inicial de modelos
├── config.yaml # Parámetros del pipeline
└── requirements.txt # Lista de dependencias

