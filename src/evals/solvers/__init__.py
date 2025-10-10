"""Custom solvers for evals."""

from .mcq_solver import multiple_choice_prefill
from .math_solver import math_solver

__all__ = ["multiple_choice_prefill", "math_solver"]