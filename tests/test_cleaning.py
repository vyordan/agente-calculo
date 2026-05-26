"""
Script para probar la limpieza de texto de PDFs
"""

from src.pdf_reader import clean_pdf_text

# Ejemplos de texto con problemas comunes
test_cases = [
    # Caso 1: Palabras unidas con mayúscula
    "LaderivadaEs una función queRepresenta la tasa deCambio",
    
    # Caso 2: Palabras unidas con números
    "La función1tiene derivada2en el punto3",
    
    # Caso 3: Guiones de separación
    "La deriva-\nda de una función\nes importante",
    
    # Caso 4: Espacios múltiples
    "La  derivada    es   una función",
    
    # Caso 5: Puntuación sin espacios
    "La derivada es importante.Sin embargo,no siempre existe.",
]

print("=" * 70)
print("Prueba de limpieza de texto")
print("=" * 70)

for i, test in enumerate(test_cases, 1):
    print(f"\n[Caso {i}]")
    print(f"Original: {test}")
    cleaned = clean_pdf_text(test)
    print(f"Limpio:   {cleaned}")
    print("-" * 70)