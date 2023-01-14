import importlib.metadata
__version__ = importlib.metadata.version('mc-tk')

import os.path
DATA_FOLDER = os.path.dirname( os.path.realpath(__file__)) + "/data/"

# Controls the plot style for bar-charts and histograms.
BARPLOT_KWARGS = {"facecolor":"none", "edgecolor":"black", "alpha":0.8, "linewidth":1.1}
