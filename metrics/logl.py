import numpy as np

class Metric():
    def __init__(self, **kwargs):
        self.requires = ['target_label_prob_ned']
        self.name = 'logl'

    def __call__(self, target_label_prob_ned):
        return np.mean(np.log(np.maximum(target_label_prob_ned, 0.0001)))
