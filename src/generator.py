"""
Motor de generación de respuestas.
Usa Qwen2.5-Math-1.5B (especializado en matemáticas).
"""

import os
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class QwenMathGenerator:
    """
    Usa Qwen2.5-Math-1.5B - Modelo especializado en cálculo.
    Optimizado para razonamiento matemático.
    """
    
    def __init__(self, cache_dir: str):
        """
        Inicializa Qwen2.5-Math.
        """
        os.environ['HF_HOME'] = cache_dir
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
        print(f"Cargando Qwen2.5-Math-1.5B...")
        
        # Cargar modelo matemático
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-Math-1.5B",
            cache_dir=cache_dir,
            local_files_only=True,
            device_map="cpu",
            torch_dtype=torch.float32
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-Math-1.5B",
            cache_dir=cache_dir,
            local_files_only=True
        )
        
        self.device = torch.device('cpu')
        self.model.eval()
        
        self.model_name = "Qwen2.5-Math-1.5B"
        
        print(f"Qwen2.5-Math cargado - Especializado en matemáticas")
    
    def generate_answer(
        self,
        question: str,
        contexts: List[str],
        temperature: float = 0.3,  # Bajo para matemáticas (determinístico)
        max_length: int = 800
    ) -> Dict:
        """
        Genera respuesta matemática con Qwen2.5-Math.
        Mucho mejor que Flan-T5 para cálculo.
        """
        try:
            # Construir prompt para matemáticas
            prompt = self._build_math_prompt(question, contexts)
            
            # Tokenizar
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt"
            ).to(self.device)
            
            # Generar respuesta
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=max(temperature, 0.1),
                    do_sample=temperature > 0,
                    top_p=0.95,
                    top_k=50,
                    repetition_penalty=1.2
                )
            
            # Decodificar
            answer = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # Extraer solo la parte generada (sin prompt)
            answer = answer[len(prompt):].strip()
            
            confidence = self._estimate_confidence(answer)
            
            return {
                'answer': answer,
                'confidence': confidence,
                'source': 'qwen_math'
            }
            
        except Exception as e:
            return {
                'answer': f"Error: {str(e)}",
                'confidence': 0.0,
                'source': 'qwen_math'
            }
    
    def _build_math_prompt(self, question: str, contexts: List[str]) -> str:
        """
        Prompt optimizado para Qwen2.5-Math.
        """
        combined_context = "\n\n".join(contexts[:3]) if contexts else ""
        
        if combined_context:
            prompt = f"""<|im_start|>system
Eres un profesor experto en matemáticas y cálculo. Proporciona respuestas claras, detalladas, correctas y totalmente en español.
Usa notación matemática clara. Explica los pasos cuando sea necesario.
<|im_end|>
<|im_start|>user


Responde esta pregunta:
{question}
<|im_end|>
<|im_start|>assistant
"""
        else:
            prompt = f"""<|im_start|>system
Eres un profesor experto en matemáticas y cálculo. Proporciona respuestas claras, detalladas y correctas en español.
Usa notación matemática clara. Explica los pasos cuando sea necesario.
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
"""
        
        return prompt
    
    def _estimate_confidence(self, answer: str) -> float:
        """
        Estima confianza para respuestas matemáticas.
        """
        if not answer or len(answer) < 20:
            return 0.3
        
        # Palabras que indican incertidumbre
        uncertainty = ["no sé", "desconozco", "no puedo"]
        if any(word in answer.lower() for word in uncertainty):
            return 0.2
        
        # Respuestas completas tienen mayor confianza
        if len(answer) > 100:
            return 0.85
        elif len(answer) > 50:
            return 0.75
        else:
            return 0.60