"""
Módulo para leer y extraer texto de archivos PDF
Utiliza pdfplumber para la extracción robusta de texto
"""

import pdfplumber
from typing import Optional

def read_pdf(file_path: str) -> str:
    """
    Extrae todo el texto de un archivo PDF.
    
    Args:
        file_path: Ruta al archivo PDF
        
    Returns:
        String con el texto extraído
        
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
                    # Agregar número de página como referencia
                    text_parts.append(f"\n--- Página {i + 1} ---\n{page_text}")
            
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