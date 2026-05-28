"""
Pipeline principal con Qwen2.5-Math.
"""

import os
import yaml
from typing import Dict, List

from src.pdf_reader import read_pdf
from src.chunker import split_text
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.qa_engine import QAEngine
from src.generator import QwenMathGenerator  # NUEVO
from src.solver import is_math_problem, solve_math

# Cache de modelos
_models_cache = {
    'qwen_math': None,  # NUEVO
    'config': None
}

def load_config(config_path: str = 'config.yaml') -> dict:
    global _models_cache
    if _models_cache['config'] is None:
        with open(config_path, 'r', encoding='utf-8') as f:
            _models_cache['config'] = yaml.safe_load(f)
    return _models_cache['config']

def get_qwen_math(config: dict):
    """Obtiene Qwen2.5-Math (cacheado)"""
    global _models_cache
    if _models_cache['qwen_math'] is None:
        _models_cache['qwen_math'] = QwenMathGenerator(
            cache_dir=config['models_cache_dir']
        )
    return _models_cache['qwen_math']

def process_pdf(pdf_path: str, config: dict = None) -> 'VectorStore':
    if config is None:
        config = load_config()
    
    print(f"Leyendo PDF: {pdf_path}")
    text = read_pdf(pdf_path)
    print(f"Texto extraído: {len(text)} caracteres")
    
    print(f"Dividiendo texto...")
    chunks = split_text(text, chunk_size=config['chunk_size'], overlap=config['chunk_overlap'])
    print(f"{len(chunks)} fragmentos creados")
    
    print(f"Cargando embeddings...")
    embedder = Embedder(config['embedding_model'], config['models_cache_dir'])
    
    print(f"Generando vectores...")
    embeddings = embedder.embed(chunks)
    
    print(f"Creando vector store...")
    vector_store = VectorStore(config['persist_directory'], embedder)
    
    if vector_store.count() > 0:
        vector_store.reset()
    
    vector_store.add_documents(chunks, embeddings)
    print(f"Vector store listo: {vector_store.count()} fragmentos")
    
    return vector_store

def ask_question(
    vector_store: 'VectorStore',
    question: str,
    config: dict = None
) -> Dict:
    """
    Responde pregunta usando Qwen2.5-Math + contexto del PDF.
    """
    if config is None:
        config = load_config()
    
    """ ----------- ESTO SE COMENTO PUES EL MODELO SABE RESOLVER PROBLEMAS MATEMATICOS SIMBOLICOS, entonces el documeno solver.py se puede remover xd
    # Detectar problema matemático simbólico
    is_math, operation, expression = is_math_problem(question)
    
    if is_math:
        print(f"Problema matemático: {operation}")
        answer = solve_math(expression, operation)
        return {
            'answer': answer,
            'score': 1.0,
            'sources': [],
            'type': 'mathematical'
        }
    """
    # Búsqueda en PDF
    print(f"Buscando en PDF...")
    relevant_chunks = vector_store.query(question, k=config['top_k_chunks'])
    contexts = [chunk['content'] for chunk in relevant_chunks] if relevant_chunks else []
    
    # Generar respuesta con Qwen2.5-Math
    print(f"Qwen2.5-Math generando respuesta...")
    qwen_math = get_qwen_math(config)
    result = qwen_math.generate_answer(question, contexts)
    
    return {
        'answer': result['answer'],
        'score': result['confidence'],
        'sources': relevant_chunks,
        'type': 'qwen_math'
    }