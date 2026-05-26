"""
Módulo para leer y extraer texto de archivos PDF
Utiliza pdfplumber para la extracción robusta de texto
"""

import pdfplumber
import re
from typing import Optional

def read_pdf(file_path: str) -> str:
    """
    Extrae todo el texto de un archivo PDF con limpieza avanzada.
    
    Args:
        file_path: Ruta al archivo PDF
        
    Returns:
        String con el texto extraído y limpio
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el PDF no contiene texto extraíble
    """
    try:
        # Abrir el PDF
        with pdfplumber.open(file_path) as pdf:
            # Verificar que el PDF tenga páginas
            if len(pdf.pages) == 0:
                raise ValueError("El PDF no contiene páginas")
            
            # Extraer texto de todas las páginas
            text_parts = []
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                
                if page_text:
                    # Limpiar el texto de cada página
                    cleaned_text = clean_pdf_text(page_text)
                    
                    # Agregar número de página como referencia
                    text_parts.append(f"\n--- Página {i + 1} ---\n{cleaned_text}")
            
            # Unir todo el texto
            full_text = "\n".join(text_parts)
            
            # Verificar que se extrajo texto
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
    Limpia el texto extraído de PDF para corregir problemas comunes.
    
    Problemas que corrige:
    - Palabras unidas sin espacios (ej: "Laderivadaes" -> "La derivada es")
    - Guiones de separación de línea
    - Espacios múltiples
    - Saltos de línea innecesarios
    
    Args:
        text: Texto crudo extraído del PDF
        
    Returns:
        Texto limpio y legible
    """
    if not text:
        return ""
    
    # 1. CRÍTICO: Detectar y separar palabras unidas
    # Patrón: minúscula seguida de mayúscula sin espacio
    # Ejemplo: "derivadaEs" -> "derivada Es"
    text = re.sub(r'([a-zñáéíóúü])([A-ZÑÁÉÍÓÚÜ])', r'\1 \2', text)
    
    # Patrón: letra seguida de número sin espacio
    # Ejemplo: "función1" -> "función 1"
    text = re.sub(r'([a-zA-Zñáéíóúü])(\d)', r'\1 \2', text)
    
    # Patrón: número seguido de letra sin espacio
    # Ejemplo: "1función" -> "1 función"
    text = re.sub(r'(\d)([a-zA-Zñáéíóúü])', r'\1 \2', text)
    
    # 2. Detectar palabras unidas por contexto (heurística avanzada)
    text = separate_merged_words(text)
    
    # 3. Limpiar guiones de separación de línea
    # "deriva-\nda" -> "derivada"
    text = re.sub(r'-\s*\n\s*', '', text)
    
    # 4. Normalizar saltos de línea múltiples
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 5. Limpiar espacios múltiples (pero preservar uno)
    text = re.sub(r' {2,}', ' ', text)
    
    # 6. Limpiar espacios al inicio/final de líneas
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # 7. Normalizar espacios alrededor de puntuación
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Quitar espacio antes
    text = re.sub(r'([.,;:!?])([a-zA-Z])', r'\1 \2', text)  # Agregar espacio después
    
    return text.strip()


def separate_merged_words(text: str) -> str:
    """
    Detecta y separa palabras que están unidas incorrectamente.
    
    Estrategia: Busca secuencias largas sin espacios y las divide usando
    un diccionario de palabras comunes en español y términos matemáticos.
    
    Args:
        text: Texto con posibles palabras unidas
        
    Returns:
        Texto con palabras separadas
    """
    # Diccionario de palabras comunes que suelen aparecer unidas
    common_words = {
        'la', 'el', 'de', 'en', 'es', 'un', 'una', 'que', 'del', 'para',
        'con', 'por', 'se', 'al', 'los', 'las', 'como', 'más', 'pero',
        'sus', 'puede', 'son', 'este', 'esta', 'estos', 'estas',
        'derivada', 'integral', 'función', 'límite', 'variable', 'constante',
        'teorema', 'definición', 'propiedad', 'demostración', 'ejemplo',
        'ejercicio', 'solución', 'fórmula', 'ecuación', 'expresión'
    }
    
    # Procesar línea por línea
    lines = text.split('\n')
    processed_lines = []
    
    for line in lines:
        # Buscar secuencias largas sin espacios (>30 caracteres)
        words = line.split()
        processed_words = []
        
        for word in words:
            if len(word) > 30 and word.isalpha():
                # Intentar separar usando palabras conocidas
                separated = try_separate_word(word.lower(), common_words)
                processed_words.append(separated)
            else:
                processed_words.append(word)
        
        processed_lines.append(' '.join(processed_words))
    
    return '\n'.join(processed_lines)


def try_separate_word(word: str, dictionary: set) -> str:
    """
    Intenta separar una palabra larga usando un diccionario.
    
    Usa programación dinámica para encontrar la mejor división.
    
    Args:
        word: Palabra larga potencialmente unida
        dictionary: Set de palabras válidas
        
    Returns:
        Palabra separada o palabra original si no se puede separar
    """
    n = len(word)
    
    # dp[i] = True si word[0:i] puede ser segmentado
    dp = [False] * (n + 1)
    dp[0] = True
    
    # parent[i] guarda el punto de división óptimo
    parent = [-1] * (n + 1)
    
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and word[j:i] in dictionary:
                dp[i] = True
                parent[i] = j
                break
    
    # Reconstruir la segmentación
    if dp[n]:
        result = []
        i = n
        while i > 0:
            j = parent[i]
            result.append(word[j:i])
            i = j
        return ' '.join(reversed(result))
    
    # Si no se puede segmentar, devolver original
    return word


def get_pdf_info(file_path: str) -> dict:
    """
    Obtiene información metadata del PDF.
    
    Args:
        file_path: Ruta al archivo PDF
        
    Returns:
        Dict con información del PDF (páginas, metadata, etc.)
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            return {
                'num_pages': len(pdf.pages),
                'metadata': pdf.metadata,
            }
    except Exception as e:
        return {'error': str(e)}