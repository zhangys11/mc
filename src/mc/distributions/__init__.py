'''
This module uses MC to generate various distributions.
'''
from ._benford import Benford
from ._binom import Binom
from ._chisq import Chisq
from ._exponential import Exponential
from ._f import F
from ._poisson import Poisson
from ._student import Student
from ._zipf import Zipf

__all__ = [
    "Benford",
    "Binom",
    "Chisq",
    "Exponential",
    "F",
    "Poisson",
    "Student",
    "Zipf"
]
