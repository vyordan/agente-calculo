"""
Tests para el módulo solver (matemático)
"""

import pytest
from src.solver import is_math_problem, solve_math

def test_is_math_problem_derivada():
    """Test de detección de derivadas"""
    is_math, op, expr = is_math_problem("¿Cuál es la derivada de x^2?")
    assert is_math == True
    assert op == 'derivada'
    assert 'x^2' in expr or 'x**2' in expr

def test_is_math_problem_integral():
    """Test de detección de integrales"""
    is_math, op, expr = is_math_problem("Calcula la integral de 2x")
    assert is_math == True
    assert op == 'integral'

def test_is_math_problem_not_math():
    """Test con pregunta no matemática"""
    is_math, op, expr = is_math_problem("¿Qué es el cálculo?")
    assert is_math == False

def test_solve_math_derivada():
    """Test de resolución de derivadas"""
    result = solve_math("x**2", "derivada")
    assert "2*x" in result or "2x" in result

def test_solve_math_integral():
    """Test de resolución de integrales"""
    result = solve_math("x", "integral")
    assert "x**2" in result or "x²" in result