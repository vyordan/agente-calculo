"""
Interfaz web con Streamlit para el sistema de QA sobre PDFs.
Permite subir PDFs, hacer preguntas y ver respuestas con contexto.
Incluye modo online opcional con deepseek-v3.2.
"""

import os
import sys
import streamlit as st
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import load_config, process_pdf, ask_question
from src.online_llm import generate_online_response

st.set_page_config(
    page_title="Agente de Calculo",
    page_icon=None,
    layout="wide"
)

def init_session_state():
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'config' not in st.session_state:
        st.session_state.config = load_config()
    if 'online_mode' not in st.session_state:
        st.session_state.online_mode = False

def main():
    init_session_state()
    
    st.title("Agente sobre Libros de Calculo")
    st.markdown("""
    Sube un PDF de calculo y haz preguntas. Con el modo online obtendras una
    respuesta adicional generada por un modelo de lenguaje avanzado.
    """)
    
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.header("Configuracion")
        
        st.markdown("### Documento actual")
        if st.session_state.pdf_processed:
            st.success("PDF procesado")
            st.info(f"Fragmentos: {st.session_state.vector_store.count()}")
            if st.button("Cargar nuevo PDF"):
                st.session_state.pdf_processed = False
                st.session_state.vector_store = None
                st.session_state.history = []
                st.rerun()
        else:
            st.warning("No hay PDF cargado")
        
        st.divider()
        
        st.markdown("### Parametros")
        config = st.session_state.config
        
        response_mode = st.selectbox(
            "Modo de respuesta",
            ["generative", "extractive", "hybrid"],
            index=0,
            help="Generativo: respuestas sintetizadas | Extractivo: texto literal | Hibrido: mejor de ambos"
        )
        config['response_mode'] = response_mode
        
        st.text(f"Modelo generativo: flan-t5-base")
        st.text(f"Modelo embeddings: all-MiniLM-L6-v2")
        st.text(f"Top-K chunks: {config['top_k_chunks']}")
        st.text(f"Chunk size: {config['chunk_size']}")
        
        with st.expander("Parametros avanzados"):
            config['generation_temperature'] = st.slider(
                "Temperatura (creatividad)", 0.0, 1.0, 0.7, 0.1
            )
            config['generation_num_beams'] = st.slider(
                "Beams (calidad)", 1, 8, 4, 1
            )
        
        st.divider()
        
        # Toggle para modo online
        st.markdown("### Modo online")
        online_mode = st.toggle(
            "Activar modo online",
            value=st.session_state.online_mode,
            help="Obten una respuesta adicional usando deepseek-v3.2 (requiere internet)"
        )
        if online_mode != st.session_state.online_mode:
            st.session_state.online_mode = online_mode
    
    # Seccion principal
    if not st.session_state.pdf_processed:
        st.header("Cargar PDF")
        uploaded_file = st.file_uploader(
            "Sube tu libro de calculo en PDF",
            type=['pdf'],
            help="El PDF sera procesado y indexado para busqueda semantica"
        )
        
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
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
                    
                    vector_store = process_pdf(tmp_path, st.session_state.config)
                    
                    status_text.text("Almacenando en vector store...")
                    progress_bar.progress(80)
                    
                    st.session_state.vector_store = vector_store
                    st.session_state.pdf_processed = True
                    
                    progress_bar.progress(100)
                    status_text.text("PDF procesado exitosamente")
                    
                    os.unlink(tmp_path)
                    
                    st.success(f"PDF indexado con {vector_store.count()} fragmentos")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error al procesar PDF: {str(e)}")
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
    
    else:
        st.header("Haz tu pregunta")
        question = st.text_input(
            "Escribe tu pregunta sobre el documento:",
            placeholder="Ejemplo: Cual es la derivada de (x^2)?",
            key="question_input"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            submit_button = st.button("Preguntar", type="primary")
        
        if submit_button and question:
            # Respuesta local offline
            with st.spinner("Generando respuesta local..."):
                try:
                    result = ask_question(
                        st.session_state.vector_store,
                        question,
                        st.session_state.config
                    )
                    st.session_state.history.append({
                        'question': question,
                        'result': result
                    })
                except Exception as e:
                    st.error(f"Error en respuesta local: {str(e)}")
                    result = None
            
            # Si el modo online esta activo, obtener respuesta extra
            if result is not None and st.session_state.online_mode:
                with st.spinner("Generando respuesta online (deepseek-v3.2)..."):
                    try:
                        online_response_placeholder = st.empty()
                        full_online_answer = ""
                        stream = generate_online_response(question, stream=True)
                        for chunk in stream:
                            full_online_answer += chunk
                            online_response_placeholder.markdown(full_online_answer)
                        online_answer = full_online_answer
                    except Exception as e:
                        online_answer = f"Error al consultar el modelo online: {str(e)}"
                        st.error(online_answer)
                
                st.session_state.history[-1]['online_answer'] = online_answer
        
        # Mostrar ultima respuesta
        if st.session_state.history:
            latest = st.session_state.history[-1]
            result = latest['result']
            
            st.divider()
            st.subheader("Respuesta")
            
            # Sin iconos, solo el tipo de respuesta
            response_type_names = {
                'mathematical': 'Problema matematico',
                'generative': 'Respuesta generada',
                'extractive': 'Respuesta extractiva',
                'not_found': 'No encontrado'
            }
            type_name = response_type_names.get(result['type'], 'Respuesta')
            st.info(f"**{type_name}**")
            
            st.markdown("### Respuesta:")
            st.markdown(f"**{result['answer']}**")
            
            if result['type'] in ['generative', 'extractive']:
                score = result['score']
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
            
            if 'alternative' in result:
                with st.expander("Ver respuesta alternativa"):
                    st.markdown(result['alternative'])
            
            if result['sources']:
                st.divider()
                st.markdown("### Fragmentos fuente del documento")
                st.caption(f"Se encontraron {len(result['sources'])} fragmentos relevantes")
                tabs = st.tabs([f"Fragmento {i+1}" for i in range(len(result['sources']))])
                for i, (tab, source) in enumerate(zip(tabs, result['sources'])):
                    with tab:
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
                        st.text_area(
                            f"Contenido del fragmento {i+1}",
                            source['content'],
                            height=200,
                            key=f"source_{i}_{len(st.session_state.history)}",
                            label_visibility="collapsed"
                        )
                        st.caption(f"Longitud: {len(source['content'])} caracteres")
            
            # Mostrar respuesta online si existe
            if 'online_answer' in latest:
                st.divider()
                st.markdown("### Respuesta del modelo online (deepseek-v3.2)")
                if latest['online_answer'].startswith("Error"):
                    st.error(latest['online_answer'])
                else:
                    with st.expander("Ver respuesta online completa"):
                        st.markdown(latest['online_answer'])
        
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
                    if 'online_answer' in item:
                        st.caption("Incluye respuesta online")
                    st.divider()

if __name__ == "__main__":
    main()