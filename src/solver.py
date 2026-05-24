"""
Módulo para resolver problemas matemáticos usando SymPy.
Detecta y resuelve derivadas, integrales y límites.
"""

import re
from typing import Tuple
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

# Transformaciones para parsear expresiones
TRANSFORMATIONS = (standard_transformations + (implicit_multiplication_application,))

def is_math_problem(question: str) -> Tuple[bool, str, str]:
    """
    Detecta si la pregunta es un problema matemático.
    
    Args:
        question: Pregunta del usuario
        
    Returns:
        Tupla (es_matematico, operacion, expresion)
        - es_matematico: True si es un problema matemático
        - operacion: 'derivada', 'integral', 'limite', o ''
        - expresion: La expresión matemática extraída, o ''
    """
    question_lower = question.lower()
    
    # Patrones para detectar operaciones
    patterns = {
        'derivada': [
            r'deriva(?:da|r|tiva)?.*de\s+(.+)',
            r'd/dx\s*\(?(.+)\)?',
            r'derivada:\s*(.+)',
        ],
        'integral': [
            r'integra(?:l|r)?.*de\s+(.+)',
            r'∫\s*(.+)',
            r'integral:\s*(.+)',
        ],
        'limite': [
            r'l[íi]mite.*de\s+(.+)',
            r'lim.*de\s+(.+)',
            r'limite:\s*(.+)',
        ]
    }
    
    # Buscar patrones
    for operation, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, question_lower)
            if match:
                expression = match.group(1).strip()
                return (True, operation, expression)
    
    # No es un problema matemático
    return (False, '', '')

def solve_math(expression: str, operation: str) -> str:
    """
    Resuelve un problema matemático usando SymPy.
    
    Args:
        expression: Expresión matemática en formato string
        operation: Tipo de operación ('derivada', 'integral', 'limite')
        
    Returns:
        String con el resultado
    """
    try:
        # Definir variable simbólica
        x = sp.Symbol('x')
        
        # Limpiar y preparar la expresión
        expr_clean = _clean_expression(expression)
        
        # Parsear la expresión
        expr = parse_expr(expr_clean, transformations=TRANSFORMATIONS)
        
        # Ejecutar la operación correspondiente
        if operation == 'derivada':
            result = sp.diff(expr, x)
            return f"La derivada de {expr} es: {result}"
            
        elif operation == 'integral':
            result = sp.integrate(expr, x)
            return f"La integral de {expr} es: {result} + C"
            
        elif operation == 'limite':
            # Intentar detectar el punto del límite
            # Por defecto, usar x -> 0
            point = 0
            result = sp.limit(expr, x, point)
            return f"El límite de {expr} cuando x → {point} es: {result}"
        
        else:
            return f"Operación '{operation}' no soportada"
            
    except Exception as e:
        return f"Error al resolver: {str(e)}. Verifica que la expresión esté bien escrita."

def _clean_expression(expr: str) -> str:
    """
    Limpia y normaliza una expresión matemática.
    
    Args:
        expr: Expresión en string
        
    Returns:
        Expresión limpia
    """
    # Reemplazos comunes
    replacements = {
        '^': '**',
        'sen': 'sin',
        'tg': 'tan',
        'ctg': 'cot',
        'arcsen': 'asin',
        'arccos': 'acos',
        'arctan': 'atan',
        'ln': 'log',
        '√': 'sqrt',
    }
    
    expr_clean = expr
    for old, new in replacements.items():
        expr_clean = expr_clean.replace(old, new)
    
    # Eliminar espacios
    expr_clean = expr_clean.strip()
    
    return expr_clean