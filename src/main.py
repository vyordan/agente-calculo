"""
Módulo principal que orquesta todo el pipeline del sistema.
Coordina PDF reading, chunking, embedding, vector store y QA.
"""

import os
import yaml
from typing import Dict, List

# Importaciones absolutas en lugar de relativas
from src.pdf_reader import read_pdf
from src.chunker import split_text
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.qa_engine import QAEngine
from src.generator import ResponseGenerator  # NUEVO
from src.solver import is_math_problem, solve_math

def load_config(config_path: str = 'config.yaml') -> dict:
    """Carga la configuración desde el archivo YAML"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def process_pdf(pdf_path: str, config: dict = None) -> 'VectorStore':
    """
    Procesa un PDF completo y crea un vector store.
    
    Pipeline:
    1. Lee el PDF
    2. Divide en chunks
    3. Genera embeddings
    4. Almacena en vector store
    
    Args:
        pdf_path: Ruta al archivo PDF
        config: Diccionario de configuración (si None, carga de config.yaml)
        
    Returns:
        VectorStore con todos los fragmentos indexados
    """
    # Cargar configuración
    if config is None:
        config = load_config()
    
    # 1. Leer PDF
    print(f"Leyendo PDF: {pdf_path}")
    text = read_pdf(pdf_path)
    print(f"Texto extraído: {len(text)} caracteres")
    
    # 2. Dividir en chunks
    print(f"Dividiendo texto en fragmentos...")
    chunks = split_text(
        text,
        chunk_size=config['chunk_size'],
        overlap=config['chunk_overlap']
    )
    print(f"Creados {len(chunks)} fragmentos")
    
    # 3. Inicializar embedder
    print(f"Cargando modelo de embeddings...")
    embedder = Embedder(
        model_name=config['embedding_model'],
        cache_dir=config['models_cache_dir']
    )
    print(f"Modelo cargado: {embedder.embedding_dimension} dimensiones")
    
    # 4. Generar embeddings
    print(f"Generando embeddings...")
    embeddings = embedder.embed(chunks)
    print(f"Embeddings generados: {len(embeddings)} vectores")
    
    # 5. Crear vector store
    print(f"Creando vector store...")
    vector_store = VectorStore(
        persist_dir=config['persist_directory'],
        embedder=embedder
    )
    
    # Resetear si ya existía
    if vector_store.count() > 0:
        print("Vector store existente encontrado, reiniciando...")
        vector_store.reset()
    
    # Añadir documentos
    vector_store.add_documents(chunks, embeddings)
    print(f"Vector store creado con {vector_store.count()} fragmentos")
    
    return vector_store

def ask_question(
    vector_store: 'VectorStore',
    question: str,
    config: dict = None
) -> Dict:
    """
    Responde una pregunta usando el vector store.
    
    Pipeline:
    1. Detecta si es problema matemático
    2. Si es matemático: resuelve con SymPy
    3. Si no: búsqueda semántica + generación de respuesta
    
    Args:
        vector_store: Vector store con documentos indexados
        question: Pregunta del usuario
        config: Diccionario de configuración
        
    Returns:
        Dict con respuesta, score, fuentes, tipo
    """
    # Cargar configuración
    if config is None:
        config = load_config()
    
    # 1. Detectar si es problema matemático
    is_math, operation, expression = is_math_problem(question)
    
    if is_math:
        # Resolver problema matemático
        print(f"Detectado problema matemático: {operation}")
        answer = solve_math(expression, operation)
        
        return {
            'answer': answer,
            'score': 1.0,
            'sources': [],
            'type': 'mathematical',
            'operation': operation
        }
    
    # 2. Búsqueda semántica
    print(f"Buscando fragmentos relevantes...")
    relevant_chunks = vector_store.query(
        question,
        k=config['top_k_chunks']
    )
    
    if not relevant_chunks:
        return {
            'answer': "No se encontraron fragmentos relevantes en el documento.",
            'score': 0.0,
            'sources': [],
            'type': 'not_found'
        }
    
    print(f" Encontrados {len(relevant_chunks)} fragmentos relevantes")
    
    # 3. Generar respuesta (modo configurável)
    response_mode = config.get('response_mode', 'generative')
    contexts = [chunk['content'] for chunk in relevant_chunks]
    
    if response_mode == 'generative':
        # Usar generación de respuesta
        print(f"Generando respuesta coherente...")
        generator = ResponseGenerator(
            model_name=config.get('generator_model', 'google/flan-t5-small'),
            cache_dir=config['models_cache_dir']
        )
        
        result = generator.generate_answer(question, contexts)
        
        return {
            'answer': result['answer'],
            'score': result['confidence'],
            'sources': relevant_chunks,
            'type': 'generative',
            'num_contexts': result['num_contexts_used']
        }
    
    elif response_mode == 'extractive':
        # Usar QA extractivo (comportamiento original)
        print(f"Extrayendo respuesta...")
        qa_engine = QAEngine(
            model_name=config['qa_model'],
            cache_dir=config['models_cache_dir']
        )
        
        result = qa_engine.answer(
            question,
            contexts,
            min_score=config.get('min_qa_score', 0.01)
        )
        
        return {
            'answer': result['answer'],
            'score': result['score'],
            'sources': relevant_chunks,
            'type': 'extractive'
        }
    
    else:  # hybrid
        # Intentar ambos y elegir el mejor
        print(f"Usando modo híbrido...")
        
        # Generar ambas respuestas
        generator = ResponseGenerator(
            model_name=config.get('generator_model', 'google/flan-t5-small'),
            cache_dir=config['models_cache_dir']
        )
        gen_result = generator.generate_answer(question, contexts)
        
        qa_engine = QAEngine(
            model_name=config['qa_model'],
            cache_dir=config['models_cache_dir']
        )
        ext_result = qa_engine.answer(question, contexts)
        
        # Elegir la mejor basándose en confianza
        if gen_result['confidence'] > ext_result['score']:
            return {
                'answer': gen_result['answer'],
                'score': gen_result['confidence'],
                'sources': relevant_chunks,
                'type': 'generative',
                'alternative': ext_result['answer']
            }
        else:
            return {
                'answer': ext_result['answer'],
                'score': ext_result['score'],
                'sources': relevant_chunks,
                'type': 'extractive',
                'alternative': gen_result['answer']
            }

def main():
    """Punto de entrada CLI (opcional)"""
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python -m src.main <ruta_pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Procesar PDF
    vector_store = process_pdf(pdf_path)
    
    # Loop interactivo
    print("\n" + "="*60)
    print("Sistema de QA listo. Escribe 'salir' para terminar.")
    print("="*60 + "\n")
    
    while True:
        question = input("\n Tu pregunta: ").strip()
        
        if question.lower() in ['salir', 'exit', 'quit']:
            print(" ")
            break
        
        if not question:
            continue
        
        # Obtener respuesta
        result = ask_question(vector_store, question)
        
        print(f"\nRespuesta: {result['answer']}")
        print(f"Confianza: {result['score']:.2%}")
        print(f"Tipo: {result['type']}")
        
        if result['sources']:
            print(f"\nFuentes ({len(result['sources'])} fragmentos):")
            for i, source in enumerate(result['sources'][:2], 1):
                preview = source['content'][:100] + "..."
                print(f"  {i}. {preview}")

if __name__ == "__main__":
    main()