"""
Tests para el módulo chunker
"""

import pytest
from src.chunker import split_text, split_text_with_metadata

def test_split_text_basic():
    """Test básico de división de texto"""
    text = "Este es un texto de prueba. " * 100
    chunks = split_text(text, chunk_size=100, overlap=20)
    
    assert len(chunks) > 1
    assert all(len(chunk) <= 150 for chunk in chunks)  # Margen por palabras

def test_split_text_empty():
    """Test con texto vacío"""
    chunks = split_text("", chunk_size=100, overlap=20)
    assert len(chunks) == 0

def test_split_text_with_metadata():
    """Test de división con metadata"""
    text = "Texto de prueba. " * 50
    chunks = split_text_with_metadata(text, chunk_size=100, overlap=20)
    
    assert len(chunks) > 0
    assert all('content' in chunk for chunk in chunks)
    assert all('chunk_id' in chunk for chunk in chunks)
    assert all('length' in chunk for chunk in chunks)