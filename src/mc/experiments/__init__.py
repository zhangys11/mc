'''
This modules uses MC to solve classic probability puzzles or simulate physical experiments.
'''

from ._dices import Dices
from ._galton_board import Galton_Board
from ._paper_clips import Paper_Clips
from ._parcel import Parcel
from ._pi import Pi
from ._prisoners import Prisoners
from ._sudden_death import Sudden_Death

__all__ = [
    "Dices",
    "Galton_Board",
    "Paper_Clips",
    "Parcel",
    "Pi",
    "Prisoners",
    "Sudden_Death"
]
