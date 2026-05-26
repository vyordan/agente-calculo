"""
Módulo para leer y extraer texto de archivos PDF
Con soporte mejorado para PDFs de matemáticas
"""

import pdfplumber
import re
from typing import Optional

def read_pdf(file_path: str) -> str:
    """
    Extrae todo el texto de un archivo PDF con limpieza avanzada.
    Optimizado para libros de matemáticas con símbolos.
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            if len(pdf.pages) == 0:
                raise ValueError("El PDF no contiene páginas")
            
            text_parts = []
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                
                if page_text:
                    # Limpiar el texto de cada página
                    cleaned_text = clean_pdf_text(page_text)
                    
                    # Agregar número de página como referencia
                    text_parts.append(f"\n--- Página {i + 1} ---\n{cleaned_text}")
            
            full_text = "\n".join(text_parts)
            
            if not full_text.strip():
                raise ValueError(
                    "No se pudo extraer texto del PDF. "
                    "Puede ser un PDF escaneado sin OCR."
                )
            
            return full_text
            
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
    except Exception as e:
        raise Exception(f"Error al leer el PDF: {str(e)}")


def clean_pdf_text(text: str) -> str:
    """
    Limpia el texto de PDF optimizado para matemáticas.
    
    Problemas que corrige:
    - Palabras unidas sin espacios
    - Símbolos matemáticos fragmentados
    - Espacios múltiples
    - Saltos de línea dentro de ecuaciones
    """
    if not text:
        return ""
    
    # 1. CRÍTICO: Detectar y separar palabras unidas
    text = re.sub(r'([a-zñáéíóúü])([A-ZÑÁÉÍÓÚÜ])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Zñáéíóúü])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Zñáéíóúü])', r'\1 \2', text)
    
    # 2. Separar palabras unidas (estrategia mejorada)
    text = separate_merged_words(text)
    
    # 3. NO eliminar guiones dentro de fórmulas/ecuaciones
    # Pero sí limpiar guiones de separación de línea claros
    text = re.sub(r'-\s*\n\s*', '', text)
    
    # 4. Normalizar saltos de línea múltiples
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 5. Limpiar espacios múltiples (preservar uno)
    text = re.sub(r' {2,}', ' ', text)
    
    # 6. IMPORTANTE: Preservar espacios alrededor de símbolos matemáticos
    # Asegurar espacio después de: +, -, *, /, =, <, >, (, ), [, ]
    text = re.sub(r'([\+\-\*/=<>\(\)\[\]]{1})([a-zA-Z0-9])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z0-9])([\+\-\*/=<>\(\)\[\]])', r'\1 \2', text)
    
    # 7. Limpiar espacios al inicio/final de líneas
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # 8. Normalizar espacios alrededor de puntuación
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'([.,;:!?])([a-zA-Z])', r'\1 \2', text)
    
    # 9. Limpiar caracteres de control raros
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t\r')
    
    return text.strip()


def separate_merged_words(text: str) -> str:
    """
    Detecta y separa palabras que están unidas incorrectamente.
    Optimizado para términos matemáticos.
    """
    # Diccionario de palabras comunes + términos matemáticos
    common_words = {
        'la', 'el', 'de', 'en', 'es', 'un', 'una', 'que', 'del', 'para',
        'con', 'por', 'se', 'al', 'los', 'las', 'como', 'más', 'pero',
        'sus', 'puede', 'son', 'este', 'esta', 'estos', 'estas',
        # Términos matemáticos
        'derivada', 'integral', 'función', 'límite', 'variable', 'constante',
        'teorema', 'definición', 'propiedad', 'demostración', 'ejemplo',
        'ejercicio', 'solución', 'fórmula', 'ecuación', 'expresión',
        'suma', 'resta', 'multiplicación', 'división', 'potencia',
        'raíz', 'logaritmo', 'exponencial', 'seno', 'coseno', 'tangente',
        'matriz', 'vector', 'conjunto', 'intervalo', 'dominio', 'rango',
        'máximo', 'mínimo', 'valor', 'punto', 'recta', 'curva', 'gráfica',
        'continuidad', 'diferenciabilidad', 'convergencia', 'divergencia',
        'serie', 'sucesión', 'término', 'coeficiente', 'polinomio'
    }
    
    lines = text.split('\n')
    processed_lines = []
    
    for line in lines:
        words = line.split()
        processed_words = []
        
        for word in words:
            # Solo procesar palabras largas sin espacios
            if len(word) > 30 and word.isalpha():
                separated = try_separate_word(word.lower(), common_words)
                processed_words.append(separated)
            else:
                processed_words.append(word)
        
        processed_lines.append(' '.join(processed_words))
    
    return '\n'.join(processed_lines)


def try_separate_word(word: str, dictionary: set) -> str:
    """
    Intenta separar una palabra larga usando programación dinámica.
    """
    n = len(word)
    
    dp = [False] * (n + 1)
    dp[0] = True
    parent = [-1] * (n + 1)
    
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and word[j:i] in dictionary:
                dp[i] = True
                parent[i] = j
                break
    
    if dp[n]:
        result = []
        i = n
        while i > 0:
            j = parent[i]
            result.append(word[j:i])
            i = j
        return ' '.join(reversed(result))
    
    return word


def get_pdf_info(file_path: str) -> dict:
    """Obtiene información metadata del PDF."""
    try:
        with pdfplumber.open(file_path) as pdf:
            return {
                'num_pages': len(pdf.pages),
                'metadata': pdf.metadata,
            }
    except Exception as e:
        return {'error': str(e)}