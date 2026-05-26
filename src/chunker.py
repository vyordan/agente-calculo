"""
Módulo para dividir texto en fragmentos (chunks) manejables.
Usa RecursiveCharacterTextSplitter de LangChain con separadores inteligentes.
"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

def split_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100
) -> List[str]:
    """
    Divide el texto en fragmentos respetando oraciones y párrafos.
    
    Args:
        text: Texto completo a dividir
        chunk_size: Tamaño máximo de cada fragmento (en caracteres)
        overlap: Número de caracteres de solapamiento entre fragmentos
        
    Returns:
        Lista de strings, cada uno es un fragmento de texto
    """
    # Configurar el splitter con separadores jerárquicos
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=[
            "\n\n\n",      # Separación entre secciones
            "\n\n",        # Separación entre párrafos
            "\n",          # Saltos de línea simples
            ". ",          # Fin de oración con espacio
            ".\n",         # Fin de oración con salto
            "; ",          # Punto y coma
            ", ",          # Comas (última opción antes de palabras)
            " ",           # Espacios entre palabras
            ""             # Caracteres individuales (último recurso)
        ],
        keep_separator=True
    )
    
    # Dividir el texto
    chunks = text_splitter.split_text(text)
    
    # Post-procesar chunks para limpiar y mejorar
    processed_chunks = []
    for chunk in chunks:
        # Limpiar espacios extras
        chunk = chunk.strip()
        
        # NUEVO: Limpiar chunk adicional
        chunk = clean_chunk(chunk)
        
        # Saltar chunks vacíos o muy cortos
        if len(chunk) < 20:
            continue
        
        # Asegurar que los chunks terminen en límites de oración cuando sea posible
        chunk = _ensure_sentence_boundary(chunk)
        
        processed_chunks.append(chunk)
    
    return processed_chunks


def clean_chunk(chunk: str) -> str:
    """
    Limpia un chunk individual para corregir problemas residuales.
    
    Args:
        chunk: Fragmento de texto
        
    Returns:
        Chunk limpio
    """
    # Detectar palabras unidas al inicio del chunk
    # Patrón común: "---PáginaXXX---Laderivada"
    chunk = re.sub(r'---([A-Za-z])', r'--- \1', chunk)
    
    # Separar palabras unidas después de números de página
    chunk = re.sub(r'(\d{1,3})---([A-Za-z])', r'\1--- \2', chunk)
    
    # Corregir espacios alrededor de paréntesis matemáticos
    chunk = re.sub(r'\(\s+', '(', chunk)
    chunk = re.sub(r'\s+\)', ')', chunk)
    
    # Normalizar espacios múltiples
    chunk = re.sub(r' {2,}', ' ', chunk)
    
    return chunk


def split_text_with_metadata(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100
) -> List[dict]:
    """
    Divide el texto y retorna chunks con metadata enriquecida.
    
    Args:
        text: Texto completo a dividir
        chunk_size: Tamaño máximo de cada fragmento
        overlap: Número de caracteres de solapamiento
        
    Returns:
        Lista de dicts con 'content', 'chunk_id', 'length', 'sentence_count'
    """
    chunks = split_text(text, chunk_size, overlap)
    
    return [
        {
            'content': chunk,
            'chunk_id': i,
            'length': len(chunk),
            'sentence_count': _count_sentences(chunk),
            'is_complete_paragraph': _is_complete_paragraph(chunk)
        }
        for i, chunk in enumerate(chunks)
    ]


def _ensure_sentence_boundary(text: str) -> str:
    """
    Asegura que el texto termine en un límite de oración válido.
    
    Args:
        text: Texto a procesar
        
    Returns:
        Texto ajustado para terminar en límite de oración
    """
    # Si el texto ya termina con puntuación final, dejarlo como está
    if text.endswith(('.', '!', '?', '.\n', '!\n', '?\n')):
        return text
    
    # Buscar el último punto antes del final
    last_period_pos = max(
        text.rfind('. '),
        text.rfind('.\n'),
        text.rfind('! '),
        text.rfind('!\n'),
        text.rfind('? '),
        text.rfind('?\n')
    )
    
    # Si encontramos un punto y está en el último 20% del texto,
    # cortar ahí para mantener oraciones completas
    if last_period_pos > len(text) * 0.8:
        return text[:last_period_pos + 1].strip()
    
    # Si no, devolver el texto original
    return text


def _count_sentences(text: str) -> int:
    """
    Cuenta el número aproximado de oraciones en el texto.
    
    Args:
        text: Texto a analizar
        
    Returns:
        Número de oraciones
    """
    # Contar signos de puntuación final
    sentence_endings = re.findall(r'[.!?]+', text)
    return len(sentence_endings)


def _is_complete_paragraph(text: str) -> bool:
    """
    Determina si el chunk representa un párrafo completo.
    
    Args:
        text: Texto a evaluar
        
    Returns:
        True si parece ser un párrafo completo
    """
    # Un párrafo completo típicamente:
    # 1. Empieza con mayúscula o espacio de indentación
    # 2. Termina con puntuación
    # 3. Tiene al menos una oración completa
    
    starts_properly = text[0].isupper() or text.startswith(' ')
    ends_properly = text.rstrip().endswith(('.', '!', '?'))
    has_content = len(text.strip()) > 50
    
    return starts_properly and ends_properly and has_content


def split_by_semantic_sections(text: str, max_chunk_size: int = 800) -> List[str]:
    """
    Divide el texto intentando preservar secciones semánticas completas.
    Útil para textos académicos con títulos y subsecciones.
    
    Args:
        text: Texto completo
        max_chunk_size: Tamaño máximo permitido por chunk
        
    Returns:
        Lista de chunks que respetan secciones del documento
    """
    chunks = []
    current_chunk = ""
    
    # Dividir por líneas
    lines = text.split('\n')
    
    for line in lines:
        line_stripped = line.strip()
        
        # Detectar posibles títulos de sección
        is_section_header = (
            len(line_stripped) < 60 and
            len(line_stripped) > 0 and
            (line_stripped[0].isupper() or line_stripped[0].isdigit()) and
            not line_stripped.endswith((',', ';'))
        )
        
        # Si encontramos un encabezado y el chunk actual es grande,
        # guardar el chunk actual e iniciar uno nuevo
        if is_section_header and len(current_chunk) > max_chunk_size * 0.5:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = line + '\n'
        else:
            # Agregar la línea al chunk actual
            current_chunk += line + '\n'
            
            # Si el chunk supera el tamaño máximo, guardarlo
            if len(current_chunk) > max_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = ""
    
    # Agregar el último chunk si existe
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks