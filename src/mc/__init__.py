import importlib.metadata

try:
    __version__ = importlib.metadata.version("mc-tk")
except importlib.metadata.PackageNotFoundError:  # running from a source checkout
    __version__ = "1.2.0"

import os.path
DATA_FOLDER = os.path.dirname(os.path.realpath(__file__)) + "/data/"

# Controls the plot style for bar-charts and histograms.
BARPLOT_KWARGS = {"facecolor": "none", "edgecolor": "black", "alpha": 0.8, "linewidth": 1.1}

from ._mcbase import McBase
