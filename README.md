[![Docker Image Size](https://img.shields.io/docker/image-size/vyordan/agente-calculo?label=Image%20Size)](https://hub.docker.com/r/vyordan/agente-calculo)
[![Docker Pulls](https://img.shields.io/docker/pulls/vyordan/agente-calculo?label=Pulls)](https://hub.docker.com/r/vyordan/agente-calculo)


## Ejecutar
RECOMENDACION: Antes de ejecutar los siguientes ocmandos es muy importante tener un entorno virtual y activarlo, por el tema de las dependencias
```bash
# Instalar dependencias
pip install -r requirements.txt

# Descargar los modelos
python download_models.py

# Iniciar la applicacion, una vez se tienen las dependencias y los
#modelos descargados solo es necesario este comando para volver a ejecutar
streamlit run src/app.py 
```

# Agente de Cálculo
Sistema de preguntas y respuestas para libros y documentos de cálculo. Permite subir un PDF, procesar su contenido y consultar información mediante un modelo de lenguaje y técnicas de búsqueda.

## Tecnologías utilizadas
Python

Streamlit: interfaz web.

Qwen2.5-Math-1.5B: modelo generativo para responder preguntas de matemáticas.

BM25: algoritmo de recuperación de fragmentos relevantes dentro del texto.

pdfplumber: extracción de texto de archivos PDF.

LangChain (RecursiveCharacterTextSplitter): división del texto en fragmentos manejables.

SymPy: resolución simbólica de problemas matemáticos (derivadas, integrales, límites).

Docker: containerización para facilitar el despliegue en diferentes arquitecturas (amd64 y arm64).

GitHub Actions: automatización del build multi-arquitectura de la imagen Docker.

## Componentes del sistema
PDF Reader (pdf_reader.py): extrae texto de archivos PDF, maneja PDFs complejos y realiza limpieza del texto extraído.

Chunker (chunker.py): divide el texto en fragmentos (chunks) usando técnicas avanzadas que respetan los límites semánticos de oraciones y párrafos.

BM25 Retriever (bm25_retriever.py): indexa los fragmentos y permite recuperar los más relevantes para una consulta.

Generator (generator.py): motor de generación de respuestas que utiliza un modelo Qwen2.5-Math-1.5B optimizado para CPU.

Solver (solver.py): módulo que utiliza SymPy para detectar y resolver problemas matemáticos (derivadas, integrales, límites) de manera simbólica.

Online LLM (online_llm.py): integración con el modelo deepseek-v3.2 a través de RouteLLM para consultas externas.

Main (main.py): orquesta el pipeline principal de procesamiento y consulta.

App (app.py): interfaz web desarrollada con Streamlit para cargar PDFs y realizar preguntas.
