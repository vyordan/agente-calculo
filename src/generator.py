"""
Motor de generación de respuestas.
Usa Qwen2.5-Math-1.5B optimizado para máxima velocidad en CPU.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict

class QwenMathGenerator:
    def __init__(self, cache_dir: str):
        # -------------------- Ajustes de rendimiento global --------------------
        num_cores = os.cpu_count()  # usa todos los núcleos disponibles
        os.environ["OMP_NUM_THREADS"] = str(num_cores)
        os.environ["MKL_NUM_THREADS"] = str(num_cores)
        torch.set_num_threads(num_cores)
        torch.set_num_interop_threads(num_cores)
        # ----------------------------------------------------------------------

        os.environ['HF_HOME'] = cache_dir
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'

        print("Cargando Qwen2.5-Math-1.5B")

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

        # Compilación JIT para acelerar inferencia (solo PyTorch 2.0+)
        #self.model = torch.compile(self.model, mode="reduce-overhead")

        # Cuantización dinámica a INT8 – descomenta si quieres probar
        # self.model = torch.quantization.quantize_dynamic(
        #     self.model, {torch.nn.Linear}, dtype=torch.qint8
        # )

        self.device = torch.device('cpu')
        self.model.eval()
        print("Qwen2.5-Math cargado")

    def generate_answer(
        self,
        question: str,
        contexts: List[str] = None,
        temperature: float = 0.3,
        max_new_tokens: int = 512
    ) -> Dict:
        try:
            prompt = self._build_math_prompt(question, contexts or [])

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,                # greedy → más rápido y determinista
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )

            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = answer[len(prompt):].strip()
            confidence = self._estimate_confidence(answer)

            return {'answer': answer, 'confidence': confidence, 'source': 'qwen_math'}

        except Exception as e:
            return {'answer': f"Error: {str(e)}", 'confidence': 0.0, 'source': 'qwen_math'}

    def _build_math_prompt(self, question: str, contexts: List[str]) -> str:
        
        return f"""<|im_start|>system
Eres un profesor experto en matemáticas y cálculo. Proporciona respuestas claras.
Usa notación matemática clara. Explica los pasos cuando sea necesario. Tu respuesta debe ser en Español.
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
"""

    def _estimate_confidence(self, answer: str) -> float:
        if not answer or len(answer) < 20:
            return 0.3
        if any(word in answer.lower() for word in ["no sé", "desconozco", "no puedo"]):
            return 0.2
        if len(answer) > 100:
            return 0.85
        elif len(answer) > 50:
            return 0.75
        return 0.60