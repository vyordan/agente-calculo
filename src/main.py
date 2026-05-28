"""
Pipeline principal con Qwen2.5-Math + BM25.
"""

import os
import yaml
from typing import Dict, List

from src.pdf_reader import read_pdf
from src.chunker import split_text
from src.bm25_retriever import BM25Retriever   # Nuevo
from src.generator import QwenMathGenerator
from src.solver import is_math_problem, solve_math  # Aunque no se use, se mantiene por si acaso

# Cache de modelos
_models_cache = {
    'qwen_math': None,
    'config': None
}

def load_config(config_path: str = 'config.yaml') -> dict:
    """Carga la configuración desde archivo YAML (cacheada)."""
    global _models_cache
    if _models_cache['config'] is None:
        with open(config_path, 'r', encoding='utf-8') as f:
            _models_cache['config'] = yaml.safe_load(f)
    return _models_cache['config']

def get_qwen_math(config: dict):
    """Obtiene el generador Qwen2.5-Math (cacheado)."""
    global _models_cache
    if _models_cache['qwen_math'] is None:
        _models_cache['qwen_math'] = QwenMathGenerator(
            cache_dir=config['models_cache_dir']
        )
    return _models_cache['qwen_math']

def process_pdf(pdf_path: str, config: dict = None) -> BM25Retriever:
    """
    Procesa un PDF: extrae texto, divide en chunks e indexa con BM25.
    Retorna el objeto BM25Retriever listo para consultas.
    """
    if config is None:
        config = load_config()
    
    print(f"Leyendo PDF: {pdf_path}")
    text = read_pdf(pdf_path)
    print(f"Texto extraído: {len(text)} caracteres")
    
    print(f"Dividiendo texto...")
    chunks = split_text(
        text,
        chunk_size=config['chunk_size'],
        overlap=config['chunk_overlap']
    )
    print(f"{len(chunks)} fragmentos creados")
    
    print(f"Construyendo índice BM25...")
    retriever = BM25Retriever()
    retriever.index(chunks)
    
    return retriever

def ask_question(retriever: BM25Retriever, question: str, config: dict = None) -> Dict:
    if config is None:
        config = load_config()
    
    relevant_chunks = retriever.query(question, k=config['top_k_chunks'])
    contexts = [chunk['content'] for chunk in relevant_chunks] if relevant_chunks else []
    
    qwen_math = get_qwen_math(config)
    result = qwen_math.generate_answer(question, contexts)
    
    # Asegurar que el score sea un float simple
    confidence = float(result['confidence']) if 'confidence' in result else 0.0
    return {
        'answer': str(result['answer']),   # por si acaso
        'score': confidence,
        'sources': relevant_chunks,
        'type': 'qwen_math'
    }