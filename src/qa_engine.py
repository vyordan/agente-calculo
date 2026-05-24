"""
Motor de QA extractivo usando transformers de Hugging Face.
Extrae respuestas literales del texto sin generar contenido nuevo.
"""

import os
from typing import List, Dict
from transformers import pipeline

class QAEngine:
    """
    Motor de Question Answering extractivo.
    Usa modelos de Hugging Face para extraer respuestas literales.
    """
    
    def __init__(self, model_name: str, cache_dir: str):
        """
        Inicializa el motor de QA.
        
        Args:
            model_name: Nombre del modelo de Hugging Face
            cache_dir: Directorio de caché de modelos
        """
        # Configurar variables de entorno
        os.environ['HF_HOME'] = cache_dir
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        
        # Cargar pipeline de QA
        self.qa_pipeline = pipeline(
            "question-answering",
            model=model_name,
            tokenizer=model_name,
            device=-1,  # Forzar CPU (-1 significa CPU en transformers)
            model_kwargs={'cache_dir': cache_dir}
        )
        
        self.model_name = model_name
        # Límite de tokens del modelo (distilbert tiene ~512)
        self.max_length = 512
    
    def answer(
        self,
        question: str,
        contexts: List[str],
        min_score: float = 0.01
    ) -> Dict:
        """
        Extrae una respuesta de los contextos proporcionados.
        
        Args:
            question: Pregunta del usuario
            contexts: Lista de fragmentos de texto relevantes
            min_score: Score mínimo para considerar válida la respuesta
            
        Returns:
            Dict con 'answer', 'score', 'context_used'
        """
        if not contexts:
            return {
                'answer': "No se encontraron fragmentos relevantes en el documento.",
                'score': 0.0,
                'context_used': ""
            }
        
        # Concatenar contextos (limitando tokens)
        combined_context = self._combine_contexts(contexts)
        
        try:
            # Ejecutar el modelo de QA
            result = self.qa_pipeline(
                question=question,
                context=combined_context,
                max_answer_len=200,
                handle_impossible_answer=True
            )
            
            # Verificar score mínimo
            if result['score'] < min_score:
                return {
                    'answer': "No se encontró una respuesta confiable en el documento.",
                    'score': result['score'],
                    'context_used': combined_context[:500] + "..."
                }
            
            return {
                'answer': result['answer'],
                'score': result['score'],
                'context_used': combined_context
            }
            
        except Exception as e:
            return {
                'answer': f"Error al procesar la pregunta: {str(e)}",
                'score': 0.0,
                'context_used': combined_context
            }
    
    def _combine_contexts(self, contexts: List[str]) -> str:
        """
        Combina múltiples contextos respetando el límite de tokens.
        
        Args:
            contexts: Lista de textos
            
        Returns:
            String con contextos combinados
        """
        combined = ""
        # Estimación: ~4 caracteres por token
        max_chars = self.max_length * 3
        
        for context in contexts:
            if len(combined) + len(context) > max_chars:
                break
            combined += context + "\n\n"
        
        return combined.strip()