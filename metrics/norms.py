from scipy.spatial import distance
from sklearn.preprocessing import normalize
import numpy as np
import torch

class Metric():
    def __init__(self, quant, **kwargs):
        self.requires = ['features']
        self.name     = 'norms'
        self.quant = quant
        self.name = 'norms@{}'.format(quant)

    def __call__(self, features):
        return np.percentile(np.linalg.norm(features, axis=1), self.quant)
