import numpy as np
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score

def calculate_nrmse(y_true, y_pred):
    """
    Normalized Root Mean Squared Error (NRMSE)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    data_range = np.max(y_true) - np.min(y_true)

    if data_range == 0:
        return 0.0
        
    return rmse / data_range

def calculate_f1(y_true, y_pred, average='weighted'):
    return f1_score(y_true, y_pred, average=average)

def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def run_masking_experiment(X, imputer_model, mask_fraction=0.1, seed=42):
    """
    Experiment Logic: Corrupts data -> Imputes -> Returns NRMSE.
    """
    X_true = np.array(X).copy()
    X_masked = X_true.copy()
    n_samples, n_features = X_masked.shape
    
    # Create random mask
    np.random.seed(seed)
    mask = np.random.rand(n_samples, n_features) < mask_fraction
    X_masked[mask] = np.nan
    
    # Run Imputation
    # We clone the imputer to avoid resetting the original one
    from sklearn.base import clone
    imputer = clone(imputer_model)
    X_imputed = imputer.fit_transform(X_masked)
    
    # Calculate NRMSE only on the masked values
    return calculate_nrmse(X_true[mask], X_imputed[mask])