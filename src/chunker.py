"""
Módulo para dividir texto en fragmentos (chunks) manejables.
Usa RecursiveCharacterTextSplitter de LangChain.
"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[str]:
    """
    Divide el texto en fragmentos con solapamiento.
    
    Args:
        text: Texto completo a dividir
        chunk_size: Tamaño máximo de cada fragmento (en caracteres)
        overlap: Número de caracteres de solapamiento entre fragmentos
        
    Returns:
        Lista de strings, cada uno es un fragmento de texto
    """
    # Configurar el splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Dividir el texto
    chunks = text_splitter.split_text(text)
    
    # Filtrar chunks vacíos
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    
    return chunks

def split_text_with_metadata(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[dict]:
    """
    Divide el texto y retorna chunks con metadata.
    
    Args:
        text: Texto completo a dividir
        chunk_size: Tamaño máximo de cada fragmento
        overlap: Número de caracteres de solapamiento
        
    Returns:
        Lista de dicts con 'content', 'chunk_id', 'length'
    """
    chunks = split_text(text, chunk_size, overlap)
    
    return [
        {
            'content': chunk,
            'chunk_id': i,
            'length': len(chunk)
        }
        for i, chunk in enumerate(chunks)
    ]
