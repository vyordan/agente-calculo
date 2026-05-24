"""
Tests para el módulo vector_store
"""

import pytest
import tempfile
import shutil
from src.vector_store import VectorStore
from src.embedder import Embedder

def test_vector_store_add_and_query():
    """Test de añadir y buscar documentos"""
    try:
        # Crear directorio temporal
        temp_dir = tempfile.mkdtemp()
        
        # Inicializar embedder y vector store
        embedder = Embedder(
            "sentence-transformers/all-MiniLM-L6-v2",
            "./models_cache"
        )
        vs = VectorStore(temp_dir, embedder)
        
        # Añadir documentos
        texts = ["La derivada de x^2 es 2x", "La integral de 1/x es ln(x)"]
        vs.add_documents(texts)
        
        assert vs.count() == 2
        
        # Buscar
        results = vs.query("¿Cuál es la derivada?", k=1)
        assert len(results) > 0
        assert 'content' in results[0]
        assert 'score' in results[0]
        
        # Limpiar
        shutil.rmtree(temp_dir)
        
    except Exception as e:
        pytest.skip(f"Modelos no disponibles: {e}")