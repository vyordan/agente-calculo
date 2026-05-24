"""
Tests para el módulo pdf_reader
"""

import pytest
import os
from src.pdf_reader import read_pdf, get_pdf_info

def test_read_pdf_file_not_found():
    """Test que verifica el manejo de archivos no existentes"""
    with pytest.raises(FileNotFoundError):
        read_pdf("archivo_inexistente.pdf")

def test_get_pdf_info_error():
    """Test que verifica get_pdf_info con archivo inválido"""
    info = get_pdf_info("archivo_inexistente.pdf")
    assert 'error' in info