import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from utilities.misc import log_ppk_vmf_vec

class Metric():
    def __init__(self, **kwargs):
        self.requires = ['features', 'nearest_points_cosine', 'nearest_features_cosine', 'target_labels']
        self.name     = 'unc_logppk'

    def __call__(self, features, nearest_points_cosine, k_closest_classes_cosine, target_labels, **kwargs):
        # Calculate distance to closest neighbor
        query_norm = np.linalg.norm(features, axis=1)
        query_dir = features / np.expand_dims(query_norm, 1)
        neighbor_norm = np.linalg.norm(features[nearest_points_cosine[:, 1], :], axis=1)
        neighbor_dir = features[nearest_points_cosine[:, 1], :] / np.expand_dims(neighbor_norm, 1)
        logppk_dist = log_ppk_vmf_vec(torch.from_numpy(query_dir),
                                      torch.from_numpy(query_norm).unsqueeze(1),
                                      torch.from_numpy(neighbor_dir),
                                      torch.from_numpy(neighbor_norm).unsqueeze(1)).numpy()

        # See if this predicts mispredictions
        is_correct = k_closest_classes_cosine[:,0] == target_labels.squeeze()
        auroc = roc_auc_score(y_true=is_correct, y_score=logppk_dist)

        return auroc
