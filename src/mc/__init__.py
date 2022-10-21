import importlib.metadata
__version__ = importlib.metadata.version('mc-tk')

import os.path
DATA_FOLDER = os.path.dirname( os.path.realpath(__file__)) + "/data/"