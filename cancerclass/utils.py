import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder

def nup(X):
    return X.detach().numpy()

def identity_transform(X):
    return X

def ensure_numpy(X):
    if type(X) is np.ndarray:
        return X
    elif type(X) is pd.DataFrame:
        return X.to_numpy()
    else:
        return np.array(X)

def ohe_data(X):
    enc = OrdinalEncoder()
    X = ensure_numpy(X).reshape(-1, 1)
    return enc.fit_transform(X)

def build_labels(label_list: list) -> tuple[np.ndarray, list, dict]:
    """Build plotting information for plotting colored clusters

    Args:
        label_list (list): List of labels for every sample.

    Returns:
        tuple[np.ndarray, list, dict]: Colors for every point, unique label set, and dict that converts labels to colors.
    """
    unique_labels = list(sorted(set(label_list)))
    colors = plt.cm.tab10(range(len(unique_labels)))
    label2color = dict(zip(unique_labels, colors))
    point_colors = [label2color[l] for l in label_list]
    return np.array(point_colors), unique_labels, label2color