"""
Script para precargar modelos de ML y guardarlos en caché local.
Esto permite que la aplicación funcione completamente offline.
"""

import os
import yaml
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AutoModelForSeq2SeqLM

def load_config():
    """Carga la configuración desde config.yaml"""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def download_models():
    """Descarga y cachea todos los modelos necesarios"""
    config = load_config()
    
    # Configurar directorio de caché
    cache_dir = config['models_cache_dir']
    os.makedirs(cache_dir, exist_ok=True)
    
    # Configurar variables de entorno para Hugging Face
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_dir
    
    print("=" * 60)
    print("Descargando modelos para uso offline...")
    print("=" * 60)
    
    # 1. Descargar modelo de embeddings
    print(f"\n[1/3] Descargando modelo de embeddings: {config['embedding_model']}")
    embedding_model = SentenceTransformer(
        config['embedding_model'],
        cache_folder=cache_dir
    )
    print(f"Modelo de embeddings descargado")
    
    # 2. Descargar modelo de QA
    print(f"\n[2/3] Descargando modelo de QA: {config['qa_model']}")
    qa_model = AutoModelForQuestionAnswering.from_pretrained(
        config['qa_model'],
        cache_dir=cache_dir
    )
    qa_tokenizer = AutoTokenizer.from_pretrained(
        config['qa_model'],
        cache_dir=cache_dir
    )
    print(f"Modelo de QA descargado correctamente")
    
    # 3. Descargar modelo generativo
    generator_model = config.get('generator_model', 'google/flan-t5-large')
    print(f"\n[3/3] Descargando modelo generativo: {generator_model}")
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(
        generator_model,
        cache_dir=cache_dir
    )
    gen_tokenizer = AutoTokenizer.from_pretrained(
        generator_model,
        cache_dir=cache_dir
    )
    print(f"Modelo generativo descargado correctamente")
    
    print("\n" + "=" * 60)
    print("Todos los modelos descargados exitosamente")
    print(f"Ubicación: {os.path.abspath(cache_dir)}")
    print("=" * 60)

if __name__ == "__main__":
    download_models()