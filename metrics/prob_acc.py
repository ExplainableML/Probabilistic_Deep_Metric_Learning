import numpy as np

class Metric():
    def __init__(self, **kwargs):
        self.requires = ['label_pred_ned', 'target_labels']
        self.name = 'prob_acc'

    def __call__(self, label_pred_ned, target_labels):
        return np.mean(label_pred_ned == target_labels[:,0])
