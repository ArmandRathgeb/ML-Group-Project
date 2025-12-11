import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

def identity_transform(X):
    return X

def ensure_numpy(X):
    if type(X) is np.ndarray:
        return X
    elif type(X) is pd.DataFrame:
        return X.to_numpy()
    else:
        return np.array(X)

def ohe_data(X) -> np.ndarray:
    """Ordinally encode labels for categories

    Args:
        X: Labels of any type that need to be encoded of shape (N,)

    Returns:
        np.ndarray: Encoded labels as integers
    """
    enc = LabelEncoder()#OrdinalEncoder()
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

def plot_heatmap(ax, data, vmin=0.0, vmax=1.0):
    ax.imshow(data, vmin=vmin, vmax=vmax, cmap='bwr')
    for i in range(len(data[0])):
        for j in range(len(data[1])):
            text = ax.text(j, i, np.round(data[i, j],2),
                            ha="center", va="center", color="w")
            

def filter_by_missingness(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Filters out rows (proteins) that have too many missing values.
    
    Args:
        df: The proteomics dataframe (Rows=Proteins, Cols=Samples)
        threshold: Max allowed fraction of missing values (default 0.5)
    
    Returns:
        pd.DataFrame: The filtered dataframe
    """
    # Calculate missing ratio per row (axis=1)
    missing_ratio = df.isna().mean(axis=1)
    
    # Filter
    df_filtered = df[missing_ratio <= threshold]
    
    print(f"[Preprocessing] Original proteins: {df.shape[0]}")
    print(f"[Preprocessing] Kept proteins:     {df_filtered.shape[0]}")
    print(f"[Preprocessing] Dropped:           {df.shape[0] - df_filtered.shape[0]}")
    
    return df_filtered