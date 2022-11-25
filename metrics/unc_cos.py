import numpy as np
from sklearn.metrics import roc_auc_score

class Metric():
    def __init__(self, **kwargs):
        self.requires = ['features_cosine', 'nearest_points_cosine', 'nearest_features_cosine', 'target_labels']
        self.name     = 'unc_cos'

    def __call__(self, features_cosine, nearest_points_cosine, k_closest_classes_cosine, target_labels, **kwargs):
        # Calculate distance to closest neighbor
        cos_sim = np.sum(features_cosine[nearest_points_cosine[:, 1], :] * features_cosine, axis=1)

        # See if this predicts mispredictions
        is_correct = k_closest_classes_cosine[:,0] == target_labels.squeeze()
        auroc = roc_auc_score(y_true=is_correct, y_score=cos_sim)

        return auroc
