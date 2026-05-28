import os, yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_config():
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def download_models():
    config = load_config()
    cache_dir = config['models_cache_dir']
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['HF_HOME'] = cache_dir

    print("Descargando Qwen2.5-Math-1.5B...")
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B", cache_dir=cache_dir)
    AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B", cache_dir=cache_dir)
    print("Modelo descargado.")

if __name__ == "__main__":
    download_models()