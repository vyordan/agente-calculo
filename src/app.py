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
    page_icon=" ",
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

    """)
    
    st.divider()
    
    # Sidebar para configuración
    with st.sidebar:
        st.header(" Configuración")
        
        st.markdown("### Documento actual")
        if st.session_state.pdf_processed:
            st.success(f" PDF procesado")
            st.info(f"Fragmentos: {st.session_state.vector_store.count()}")
            
            if st.button(" Cargar nuevo PDF"):
                st.session_state.pdf_processed = False
                st.session_state.vector_store = None
                st.session_state.history = []
                st.rerun()
        else:
            st.warning("No hay PDF cargado")
        
        st.divider()
        
        st.markdown("### Parámetros")
        config = st.session_state.config
        
        # Selector de modo de respuesta
        response_mode = st.selectbox(
            "Modo de respuesta",
            ["generative", "extractive", "hybrid"],
            index=0,
            help="Generativo: respuestas sintetizadas | Extractivo: texto literal | Híbrido: mejor de ambos"
        )
        config['response_mode'] = response_mode
        
        st.text(f"Modelo generativo: flan-t5-base")
        st.text(f"Modelo embeddings: all-MiniLM-L6-v2")
        st.text(f"Top-K chunks: {config['top_k_chunks']}")
        st.text(f"Chunk size: {config['chunk_size']}")
        
        # Parámetros avanzados
        with st.expander(" Parámetros avanzados"):
            config['generation_temperature'] = st.slider(
                "Temperatura (creatividad)",
                0.0, 1.0, 0.7, 0.1,
                help="0.0 = determinista, 1.0 = más creativo"
            )
            config['generation_num_beams'] = st.slider(
                "Beams (calidad)",
                1, 8, 4, 1,
                help="Mayor = mejor calidad pero más lento"
            )
    
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
            with st.spinner(" Procesando PDF..."):
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text(" Leyendo PDF...")
                    progress_bar.progress(20)
                    
                    status_text.text(" Dividiendo en fragmentos...")
                    progress_bar.progress(40)
                    
                    status_text.text("Generando embeddings...")
                    progress_bar.progress(60)
                    
                    # Procesar
                    vector_store = process_pdf(tmp_path, st.session_state.config)
                    
                    status_text.text(" Almacenando en vector store...")
                    progress_bar.progress(80)
                    
                    # Guardar en estado
                    st.session_state.vector_store = vector_store
                    st.session_state.pdf_processed = True
                    
                    progress_bar.progress(100)
                    status_text.text("PDF procesado exitosamente")
                    
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
            placeholder="Ejemplo: ¿Cuál es la derivada de x^2? o ¿Qué es una integral definida?",
            key="question_input"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            submit_button = st.button("Preguntar", type="primary")
        
        # Procesar pregunta
        if submit_button and question:
            with st.spinner("Generando respuesta..."):
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
            response_type_icons = {
                'mathematical': '',
                'generative': '',
                'extractive': '',
                'not_found': ''
            }
            
            response_type_names = {
                'mathematical': 'Problema matemático',
                'generative': 'Respuesta generada',
                'extractive': 'Respuesta extractiva',
                'not_found': 'No encontrado'
            }
            
            icon = response_type_icons.get(latest['result']['type'], ' ')
            type_name = response_type_names.get(latest['result']['type'], 'Respuesta')
            
            st.info(f"{icon} **{type_name}**")
            
            # Respuesta principal en un contenedor destacado
            st.markdown("### Respuesta:")
            st.markdown(f"**{latest['result']['answer']}**")
            
            # Score de confianza
            if latest['result']['type'] in ['generative', 'extractive']:
                score = latest['result']['score']
                
                # Barra de confianza visual
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.progress(score)
                with col2:
                    if score > 0.7:
                        st.success(f"{score:.1%}")
                    elif score > 0.4:
                        st.warning(f"{score:.1%}")
                    else:
                        st.error(f"{score:.1%}")
                
                st.caption(f"Confianza: {score:.1%}")
            
            # Respuesta alternativa (en modo híbrido)
            if 'alternative' in latest['result']:
                with st.expander("Ver respuesta alternativa"):
                    st.markdown(latest['result']['alternative'])
            
            # Fragmentos fuente (SIEMPRE MOSTRADOS)
            if latest['result']['sources']:
                st.divider()
                st.markdown("### Fragmentos fuente del documento")
                st.caption(f"Se encontraron {len(latest['result']['sources'])} fragmentos relevantes")
                
                # Mostrar chunks en tabs para mejor organización
                tabs = st.tabs([f"Fragmento {i+1}" for i in range(len(latest['result']['sources']))])
                
                for i, (tab, source) in enumerate(zip(tabs, latest['result']['sources'])):
                    with tab:
                        # Encabezado del fragmento
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**Fragmento {i+1}**")
                        with col2:
                            similarity = source['score']
                            if similarity > 0.8:
                                st.success(f"Similitud: {similarity:.1%}")
                            elif similarity > 0.6:
                                st.info(f"Similitud: {similarity:.1%}")
                            else:
                                st.warning(f"Similitud: {similarity:.1%}")
                        
                        # Contenido del fragmento
                        st.text_area(
                            f"Contenido del fragmento {i+1}",
                            source['content'],
                            height=200,
                            key=f"source_{i}_{len(st.session_state.history)}",
                            label_visibility="collapsed"
                        )
                        
                        # Información adicional
                        st.caption(f"Longitud: {len(source['content'])} caracteres")
        
        # Historial
        if len(st.session_state.history) > 1:
            st.divider()
            with st.expander(f"Historial de preguntas ({len(st.session_state.history)} preguntas)"):
                for i, item in enumerate(reversed(st.session_state.history[:-1]), 1):
                    st.markdown(f"**{i}. {item['question']}**")
                    response_preview = item['result']['answer'][:150]
                    if len(item['result']['answer']) > 150:
                        response_preview += "..."
                    st.text(f"{response_preview}")
                    st.caption(f"Tipo: {item['result']['type']} | Confianza: {item['result']['score']:.1%}")
                    st.divider()

if __name__ == "__main__":
    main()