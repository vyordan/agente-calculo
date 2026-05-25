"""
Motor de QA extractivo usando transformers de Hugging Face.
Extrae respuestas literales del texto sin generar contenido nuevo.
"""

import os
from typing import List, Dict
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

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
        
        # Cargar modelo y tokenizer directamente
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        # Forzar CPU
        self.device = torch.device('cpu')
        self.model.to(self.device)
        
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
            # Tokenizar input
            inputs = self.tokenizer(
                question,
                combined_context,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Ejecutar el modelo
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Obtener respuesta
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1
            
            # Calcular score de confianza
            start_score = torch.max(outputs.start_logits).item()
            end_score = torch.max(outputs.end_logits).item()
            score = (start_score + end_score) / 2
            
            # Convertir score de logit a probabilidad aproximada
            score = 1 / (1 + abs(score))  # Normalización simple
            
            # Extraer texto de la respuesta
            answer = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(
                    inputs['input_ids'][0][answer_start:answer_end]
                )
            )
            
            # Limpiar respuesta
            answer = answer.strip()
            
            # Verificar score mínimo
            if score < min_score or not answer:
                return {
                    'answer': "No se encontró una respuesta confiable en el documento.",
                    'score': score,
                    'context_used': combined_context[:500] + "..."
                }
            
            return {
                'answer': answer,
                'score': score,
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
