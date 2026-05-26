"""
Motor de generación de respuestas usando modelos seq2seq.
Sintetiza respuestas coherentes a partir de fragmentos recuperados.
"""

import os
from typing import List, Dict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

class ResponseGenerator:
    """
    Generador de respuestas usando modelos de lenguaje seq2seq.
    Convierte múltiples fragmentos en una respuesta coherente.
    """
    
    def __init__(self, model_name: str, cache_dir: str):
        """
        Inicializa el generador de respuestas.
        
        Args:
            model_name: Nombre del modelo (ej: "google/flan-t5-base")
            cache_dir: Directorio de caché de modelos
        """
        # Configurar variables de entorno
        os.environ['HF_HOME'] = cache_dir
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
        print(f"Cargando modelo generativo: {model_name}")
        
        # Cargar modelo y tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True
        )
        
        # Forzar CPU
        self.device = torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()  # Modo evaluación
        
        self.model_name = model_name
        self.max_input_length = 512
        
        print(f"Modelo generativo cargado")
    
    def generate_answer(
        self,
        question: str,
        contexts: List[str],
        temperature: float = 0.7,
        num_beams: int = 4,
        max_length: int = 300
    ) -> Dict:
        """
        Genera una respuesta sintética a partir de contextos.
        
        Args:
            question: Pregunta del usuario
            contexts: Lista de fragmentos relevantes
            temperature: Control de aleatoriedad (0.0 = determinista, 1.0 = creativo)
            num_beams: Número de beams para beam search
            max_length: Longitud máxima de la respuesta generada
            
        Returns:
            Dict con 'answer', 'confidence', 'num_contexts_used'
        """
        if not contexts:
            return {
                'answer': "No encontré información relevante en el documento para responder tu pregunta.",
                'confidence': 0.0,
                'num_contexts_used': 0
            }
        
        try:
            # Construir el prompt
            prompt = self._build_prompt(question, contexts)
            
            # Tokenizar
            inputs = self.tokenizer(
                prompt,
                max_length=self.max_input_length,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generar respuesta
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    temperature=max(temperature, 0.01),  # Evitar división por cero
                    do_sample=temperature > 0,
                    early_stopping=True,
                    no_repeat_ngram_size=3,  # Evitar repeticiones
                    length_penalty=1.0,
                    repetition_penalty=1.2  # NUEVO: Penalizar repeticiones
                )
            
            # Decodificar respuesta
            answer = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Calcular confianza aproximada
            confidence = self._estimate_confidence(answer, contexts)
            
            return {
                'answer': answer.strip(),
                'confidence': confidence,
                'num_contexts_used': len(contexts)
            }
            
        except Exception as e:
            return {
                'answer': f"Error al generar respuesta: {str(e)}",
                'confidence': 0.0,
                'num_contexts_used': 0
            }
    
    def _build_prompt(self, question: str, contexts: List[str]) -> str:
        """
        Construye el prompt optimizado para Flan-T5.
        
        Args:
            question: Pregunta del usuario
            contexts: Fragmentos de contexto
            
        Returns:
            Prompt formateado
        """
        # Combinar los 3 contextos más relevantes
        combined_context = "\n\n".join(contexts[:3])
        
        # Limitar longitud total del contexto
        max_context_chars = self.max_input_length * 3
        if len(combined_context) > max_context_chars:
            combined_context = combined_context[:max_context_chars] + "..."
        
        # Prompt optimizado para Flan-T5-base (mejor que small en seguir instrucciones)
        prompt = f"""Responde la siguiente pregunta basándote ÚNICAMENTE en el contexto proporcionado. Si la información no está en el contexto, di que no lo sabes.

Contexto:
{combined_context}

Pregunta: {question}

Responde de forma clara, completa y en español:"""
        
        return prompt
    
    def _estimate_confidence(self, answer: str, contexts: List[str]) -> float:
        """
        Estima la confianza de la respuesta generada.
        
        Args:
            answer: Respuesta generada
            contexts: Contextos usados
            
        Returns:
            Score de confianza entre 0 y 1
        """
        # Heurísticas mejoradas para Flan-T5-base
        
        # 1. Respuestas que indican no saber
        uncertainty_phrases = [
            "no lo sé", "no sé", "no puedo", "no está", "no hay información",
            "no se menciona", "no se especifica", "i don't know", "not mentioned"
        ]
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            return 0.2
        
        # 2. Longitud de la respuesta
        if len(answer) < 15:
            return 0.3
        if len(answer) > 800:  # Muy larga puede ser repetitiva
            return 0.5
        
        # 3. Overlap de palabras con contextos
        answer_words = set(answer.lower().split())
        context_words = set(" ".join(contexts).lower().split())
        
        # Palabras técnicas/específicas (mayor peso)
        technical_overlap = len([w for w in answer_words if len(w) > 6 and w in context_words])
        
        if len(answer_words) > 0:
            general_overlap = len(answer_words & context_words) / len(answer_words)
        else:
            general_overlap = 0.0
        
        # 4. Calcular score final
        base_confidence = 0.65  # Base para Flan-T5-base
        overlap_bonus = general_overlap * 0.25
        technical_bonus = min(technical_overlap * 0.02, 0.10)
        
        final_score = base_confidence + overlap_bonus + technical_bonus
        
        return min(final_score, 0.95)  # Cap al 95%