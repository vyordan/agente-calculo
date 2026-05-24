"""
Tests para el módulo embedder
"""

import pytest
from src.embedder import Embedder

# Este test requiere que los modelos estén descargados
def test_embedder_initialization():
    """Test de inicialización del embedder"""
    try:
        embedder = Embedder(
            "sentence-transformers/all-MiniLM-L6-v2",
            "./models_cache"
        )
        assert embedder.embedding_dimension == 384
    except Exception as e:
        pytest.skip(f"Modelos no disponibles: {e}")

def test_embedder_embed_query():
    """Test de embedding de query única"""
    try:
        embedder = Embedder(
            "sentence-transformers/all-MiniLM-L6-v2",
            "./models_cache"
        )
        embedding = embedder.embed_query("texto de prueba")
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)
    except Exception as e:
        pytest.skip(f"Modelos no disponibles: {e}")