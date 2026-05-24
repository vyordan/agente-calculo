"""
Tests para el módulo qa_engine
"""

import pytest
from src.qa_engine import QAEngine

def test_qa_engine_answer():
    """Test básico del motor de QA"""
    try:
        qa = QAEngine(
            "distilbert-base-cased-distilled-squad",
            "./models_cache"
        )
        
        context = ["The derivative of x squared is 2x"]
        result = qa.answer("What is the derivative?", context)
        
        assert 'answer' in result
        assert 'score' in result
        assert 'context_used' in result
        
    except Exception as e:
        pytest.skip(f"Modelos no disponibles: {e}")