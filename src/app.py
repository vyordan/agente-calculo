"""
Interfaz web con Streamlit para el sistema de QA sobre PDFs.
Permite subir PDFs, hacer preguntas y ver respuestas con contexto.
"""

import os
import sys
import streamlit as st
import tempfile
from pathlib import Path

# Añadir el directorio raíz al path para importaciones
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import load_config, process_pdf, ask_question

# Configuración de la página
st.set_page_config(
    page_title="Agente de Cálculo - QA sobre PDFs",
    layout="wide"
)

def init_session_state():
    """Inicializa el estado de la sesión"""
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'config' not in st.session_state:
        st.session_state.config = load_config()

def main():
    """Función principal de la aplicación Streamlit"""
    
    # Inicializar estado
    init_session_state()
    
    # Título y descripción
    st.title("Agente de QA sobre Libros de Cálculo")
    st.markdown("""
    
    - Extracción literal (sin generación de texto)
    - Resolución de problemas matemáticos con SymPy
    - Búsqueda semántica
    """)
    
    st.divider()
    
    # Sidebar para configuración
    with st.sidebar:
        st.header("Configuración")
        
        st.markdown("### Documento actual")
        if st.session_state.pdf_processed:
            st.success(f"PDF procesado")
            st.info(f"Fragmentos: {st.session_state.vector_store.count()}")
            
            if st.button("Cargar nuevo PDF"):
                st.session_state.pdf_processed = False
                st.session_state.vector_store = None
                st.session_state.history = []
                st.rerun()
        else:
            st.warning("No hay PDF cargado")
        
        st.divider()
        
        st.markdown("###Parámetros")
        config = st.session_state.config
        st.text(f"Modelo embeddings: all-MiniLM-L6-v2")
        st.text(f"Modelo QA: distilbert")
        st.text(f"Top-K chunks: {config['top_k_chunks']}")
        st.text(f"Chunk size: {config['chunk_size']}")
    
    # Sección principal
    if not st.session_state.pdf_processed:
        # Upload de PDF
        st.header("Cargar PDF")
        
        uploaded_file = st.file_uploader(
            "Sube tu libro de cálculo en PDF",
            type=['pdf'],
            help="El PDF será procesado y indexado para búsqueda semántica"
        )
        
        if uploaded_file is not None:
            # Guardar archivo temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Procesar PDF con barra de progreso
            with st.spinner("Procesando PDF..."):
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Leyendo PDF...")
                    progress_bar.progress(20)
                    
                    status_text.text("Dividiendo en fragmentos...")
                    progress_bar.progress(40)
                    
                    status_text.text("Generando embeddings...")
                    progress_bar.progress(60)
                    
                    # Procesar
                    vector_store = process_pdf(tmp_path, st.session_state.config)
                    
                    status_text.text("Almacenando en vector store...")
                    progress_bar.progress(80)
                    
                    # Guardar en estado
                    st.session_state.vector_store = vector_store
                    st.session_state.pdf_processed = True
                    
                    progress_bar.progress(100)
                    status_text.text("¡PDF procesado exitosamente!")
                    
                    # Limpiar archivo temporal
                    os.unlink(tmp_path)
                    
                    st.success(f"PDF indexado con {vector_store.count()} fragmentos")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error al procesar PDF: {str(e)}")
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
    
    else:
        # Interfaz de preguntas
        st.header("Haz tu pregunta")
        
        # Campo de pregunta
        question = st.text_input(
            "Escribe tu pregunta sobre el documento:",
            placeholder="Ejemplo: deriva ( x ^ 2)",
            key="question_input"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            submit_button = st.button("Preguntar", type="primary")
        
        # Procesar pregunta
        if submit_button and question:
            with st.spinner("Buscando respuesta..."):
                try:
                    result = ask_question(
                        st.session_state.vector_store,
                        question,
                        st.session_state.config
                    )
                    
                    # Añadir al historial
                    st.session_state.history.append({
                        'question': question,
                        'result': result
                    })
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Mostrar última respuesta destacada
        if st.session_state.history:
            latest = st.session_state.history[-1]
            
            st.divider()
            st.subheader("Respuesta")
            
            # Tipo de respuesta
            if latest['result']['type'] == 'mathematical':
                st.info(f"Problema matemático: {latest['result'].get('operation', 'N/A')}")
            
            # Respuesta principal
            st.markdown(f"### {latest['result']['answer']}")
            
            # Score de confianza
            if latest['result']['type'] == 'extractive':
                score = latest['result']['score']
                if score > 0.5:
                    st.success(f"Confianza: {score:.1%} (Alta)")
                elif score > 0.2:
                    st.warning(f"Confianza: {score:.1%} (Media)")
                else:
                    st.error(f"Confianza: {score:.1%} (Baja)")
            
            # Fragmentos fuente
            if latest['result']['sources']:
                with st.expander(f"Ver fragmentos fuente ({len(latest['result']['sources'])} encontrados)"):
                    for i, source in enumerate(latest['result']['sources'], 1):
                        st.markdown(f"**Fragmento {i}** (similitud: {source['score']:.1%})")
                        st.text_area(
                            f"Contenido {i}",
                            source['content'],
                            height=150,
                            key=f"source_{i}_{len(st.session_state.history)}"
                        )
                        st.divider()
        
        # Historial
        if len(st.session_state.history) > 1:
            st.divider()
            with st.expander(f"Historial de preguntas ({len(st.session_state.history)} preguntas)"):
                for i, item in enumerate(reversed(st.session_state.history[:-1]), 1):
                    st.markdown(f"**{i}. {item['question']}**")
                    st.text(f"Respuesta: {item['result']['answer'][:100]}...")
                    st.divider()

if __name__ == "__main__":
    main()