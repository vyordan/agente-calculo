"""
Módulo para generar embeddings de texto usando Sentence-Transformers.
Utiliza el modelo all-MiniLM-L6-v2 (384 dimensiones).
"""

import os
from typing import List
from sentence_transformers import SentenceTransformer

class Embedder:
    """
    Clase para generar embeddings de texto.
    Usa modelos de Sentence-Transformers cacheados localmente.
    """
    
    def __init__(self, model_name: str, cache_dir: str):
        """
        Inicializa el embedder con un modelo específico.
        
        Args:
            model_name: Nombre del modelo de Sentence-Transformers
            cache_dir: Directorio donde están cacheados los modelos
        """
        # Configurar variables de entorno para cache
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_dir
        os.environ['HF_HOME'] = cache_dir
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        
        # CRÍTICO: Forzar modo offline
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
        # Cargar el modelo desde cache
        self.model = SentenceTransformer(
            model_name,
            cache_folder=cache_dir,
            device='cpu',  # Forzar uso de CPU
            local_files_only=True  # NUEVO: Solo archivos locales
        )
        
        self.model_name = model_name
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings para una lista de textos.
        
        Args:
            texts: Lista de strings
            
        Returns:
            Lista de embeddings (cada uno es una lista de floats)
        """
        if not texts:
            return []
        
        # Generar embeddings
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Convertir a lista de listas
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Genera embedding para un solo texto (útil para queries).
        
        Args:
            text: String de consulta
            
        Returns:
            Embedding como lista de floats
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        return embedding.tolist()
    
    def get_dimension(self) -> int:
        """Retorna la dimensión de los embeddings"""
        return self.embedding_dimension