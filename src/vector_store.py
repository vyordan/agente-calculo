"""
Módulo para almacenar y buscar vectores usando ChromaDB.
Permite búsqueda por similitud semántica.
"""

import os
from typing import List, Dict
import chromadb
from chromadb.config import Settings

class VectorStore:
    """
    Almacén vectorial para búsqueda semántica.
    Usa ChromaDB como backend.
    """
    
    def __init__(self, persist_dir: str, embedder):
        """
        Inicializa o carga el vector store.
        
        Args:
            persist_dir: Directorio para persistir la base de datos
            embedder: Instancia de Embedder para generar embeddings
        """
        self.persist_dir = persist_dir
        self.embedder = embedder
        
        # Crear directorio si no existe
        os.makedirs(persist_dir, exist_ok=True)
        
        # Inicializar cliente de ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Crear o obtener colección
        self.collection_name = "pdf_chunks"
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]] = None
    ):
        """
        Añade documentos al vector store.
        
        Args:
            texts: Lista de textos a añadir
            embeddings: Embeddings precalculados (opcional)
        """
        if not texts:
            return
        
        # Generar embeddings si no se proporcionaron
        if embeddings is None:
            embeddings = self.embedder.embed(texts)
        
        # Generar IDs únicos
        existing_count = self.collection.count()
        ids = [f"chunk_{existing_count + i}" for i in range(len(texts))]
        
        # Añadir a la colección
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            ids=ids
        )
    
    def query(
        self,
        question: str,
        k: int = 5
    ) -> List[Dict]:
        """
        Busca los k fragmentos más relevantes para una pregunta.
        
        Args:
            question: Pregunta del usuario
            k: Número de resultados a retornar
            
        Returns:
            Lista de dicts con 'content' y 'score'
        """
        # Generar embedding de la pregunta
        query_embedding = self.embedder.embed_query(question)
        
        # Buscar en ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        # Formatear resultados
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                # ChromaDB retorna distancias (menor es mejor)
                # Convertir a score de similitud (mayor es mejor)
                distance = results['distances'][0][i] if results['distances'] else 1.0
                similarity_score = 1.0 - distance
                
                formatted_results.append({
                    'content': doc,
                    'score': similarity_score
                })
        
        return formatted_results
    
    def reset(self):
        """Elimina todos los documentos de la colección"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def count(self) -> int:
        """Retorna el número de documentos almacenados"""
        return self.collection.count()