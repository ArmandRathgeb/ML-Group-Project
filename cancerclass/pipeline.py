from cancerclass import utils
from cancerclass import bayes
from cancerclass.impute import DreamAIImputer
from cancerclass.metrics import calculate_nrmse, calculate_f1, calculate_accuracy,run_masking_experiment
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV, BayesianRidge
from sklearn.svm import SVC

__imputation = {
    'none' : FunctionTransformer(utils.identity_transform),
    'mean' : SimpleImputer(),
    'mice' : IterativeImputer(max_iter=30),
    'dream': DreamAIImputer(),
}

__preprocessor = {
    'none'    : FunctionTransformer(utils.identity_transform),
    '01'      : MinMaxScaler(),
    'standard': StandardScaler(),
}

__classification = {
    'naivebayes': MultinomialNB,
    'elasticnet': ElasticNetCV,
    'logistic'  : LogisticRegressionCV,
    'bayeslog'  : bayes.BayesLogisticRegression,
    'svm'       : SVC,
    'bayesridge': BayesianRidge,
}

__metrics = {
    'accuracy': calculate_accuracy,
    'f1': calculate_f1,
    'nrmse': calculate_nrmse
}

def train_pipeline(
        X: np.ndarray,
        Y: np.ndarray,
        imputation_type: str = 'dream',
        return_imputed_data: bool = False,
        preprocessing_step: str = '01',
        classifier: str = 'naivebayes',
        **classifier_kwargs
    ):

    imputer = __imputation[imputation_type]
    if classifier == 'naivebayes':
        preprocessor = MinMaxScaler()
    else:
        preprocessor = __preprocessor[preprocessing_step]
    classifier = __classification[classifier](**classifier_kwargs)

    pipeline = make_pipeline(imputer, preprocessor, classifier)
    pipeline.fit(X, Y)

    return pipeline

def accuracy(model, X_test: np.ndarray, Y_test: np.ndarray, metric: str):
    metric = __metrics[metric]
    Y_pred = model.predict(X_test)

    return metric(Y_pred, Y_test)

def analysis_pipeline(model, data: tuple, metric: str):
    pass

def benchmark_methods(X_train, Y_train, X_test, Y_test, methods=['mean', 'dream']):
    """
    Runs Validation 1 (NRMSE) and Validation 2 (F1) for all methods
    and returns a clean comparison table.
    """
    results = []
    
    print(f"Running Benchmark on methods: {methods}...")
    
    for method in methods:
        # Setup
        imputer = __imputation[method]
        
        #  Validation 1: Masking Experiment (NRMSE)
        # We pass the imputer object to metrics.py
        nrmse = run_masking_experiment(X_train, imputer)
        
        # Validation 2: Downstream Classification (F1)
        # Train full pipeline
        model = train_pipeline(X_train, Y_train, imputation_type=method)
        Y_pred = model.predict(X_test)
        f1 = calculate_f1(Y_test, Y_pred)
        
        results.append({
            'Method': method.upper(),
            'Imputation Error (NRMSE)': nrmse,
            'Classification F1': f1
        })

    # Convert to DataFrame for a nice table display
    df_results = pd.DataFrame(results).set_index('Method')
    
    # Calculate % Improvement relative to the first method (Baseline)
    baseline = df_results.iloc[0]
    df_results['NRMSE Improvement'] = ((baseline['Imputation Error (NRMSE)'] - df_results['Imputation Error (NRMSE)']) / baseline['Imputation Error (NRMSE)']) * 100
    df_results['F1 Improvement'] = ((df_results['Classification F1'] - baseline['Classification F1']) / baseline['Classification F1']) * 100
    
    print("\n--- FINAL BENCHMARK RESULTS ---")
    print(df_results)
    
    return df_results