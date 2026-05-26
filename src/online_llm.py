"""
Módulo para consultar el modelo online deepseek-v3.2 a través de RouteLLM.
"""

import os
from openai import OpenAI

# Configuración de la API (idealmente usar variables de entorno)
API_KEY = os.environ.get("ROUTELLM_API_KEY", "s2_b7d2338b7e524267ba4aadee0e45b808")
BASE_URL = "https://routellm.abacus.ai/v1"
MODEL = "deepseek-v3.2"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def generate_online_response(question: str, stream: bool = False):
    """
    Genera una respuesta usando el modelo online.
    
    Args:
        question: Pregunta del usuario.
        stream: Si True, retorna un generador que produce fragmentos de texto.
    
    Returns:
        Si stream=False: str con la respuesta completa.
        Si stream=True: generador de str con los fragmentos.
    """
    system_prompt = (
        "Eres un asistente matemático experto. Responde de forma clara y detallada, "
        "usando Markdown y LaTeX para las fórmulas ($...$ para inline, $$...$$ para bloque). "
        "Si el usuario pregunta sobre un concepto de cálculo, explica paso a paso."
    )
    
    if stream:
        # Devuelve un generador para integrar con st.write_stream
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            stream=True,
            temperature=0.7,
            max_tokens=1024
        )
        def generate():
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        return generate()
    else:
        # Respuesta completa (más simple)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            stream=False,
            temperature=0.7,
            max_tokens=1024
        )
        return response.choices[0].message.content