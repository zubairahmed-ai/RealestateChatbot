from os import listdir
from os.path import dirname
import os
__all__ = [i[:-3] for i in listdir(dirname(__file__)) if not i.startswith('__') and i.endswith('.py')]