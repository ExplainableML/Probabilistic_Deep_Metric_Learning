import numpy as np
from sklearn.metrics import roc_auc_score

class Metric():
    def __init__(self, **kwargs):
        self.requires = ['features', 'nearest_points_cosine', 'nearest_features_cosine', 'target_labels']
        self.name     = 'unc_l2'

    def __call__(self, features, nearest_points_cosine, k_closest_classes_cosine, target_labels, **kwargs):
        # Calculate distance to closest neighbor
        l2_dist = np.sqrt(np.sum((features[nearest_points_cosine[:, 1], :] - features)**2, axis=1))

        # See if this predicts mispredictions
        is_correct = k_closest_classes_cosine[:,0] != target_labels.squeeze()
        auroc = roc_auc_score(y_true=is_correct, y_score=l2_dist)

        return auroc
