"""
Descarga modelos
"""

import os
import yaml
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForQuestionAnswering, 
    AutoTokenizer,
    AutoModelForCausalLM
)

def load_config():
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def download_models():
    config = load_config()
    
    cache_dir = config['models_cache_dir']
    os.makedirs(cache_dir, exist_ok=True)
    
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_dir
    
    print("=" * 70)
    print("Descargando modelos...")
    print("=" * 70)
    
    # 1. Embeddings
    print(f"\n[1/3] Descargando embeddings...")
    embedding_model = SentenceTransformer(
        config['embedding_model'],
        cache_folder=cache_dir
    )
    print(f"Embeddings descargados")
    
    # 2. QA extractivo
    print(f"\n[2/3] Descargando modelo QA...")
    qa_model = AutoModelForQuestionAnswering.from_pretrained(
        config['qa_model'],
        cache_dir=cache_dir
    )
    qa_tokenizer = AutoTokenizer.from_pretrained(
        config['qa_model'],
        cache_dir=cache_dir
    )
    print(f"Modelo QA descargado")
    
    # 3. Qwen2.5-Math (NUEVO - Especializado en cálculo)
    print(f"\n[3/3] Descargando Qwen2.5-Math-1.5B...")
    print(f"   (Modelo especializado en matemáticas)")
    qwen_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B",
        cache_dir=cache_dir,
        torch_dtype="auto"
    )
    qwen_tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B",
        cache_dir=cache_dir
    )
    print(f"Qwen2.5-Math descargado (~3.5GB)")
    
    print("\n" + "=" * 70)
    print("Todos los modelos descargados")
    print(f"   Ubicación: {os.path.abspath(cache_dir)}")
    print("=" * 70)

if __name__ == "__main__":
    download_models()