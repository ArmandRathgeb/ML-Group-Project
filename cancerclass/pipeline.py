from sklearn.experimental import enable_iterative_imputer

from cancerclass import utils
from cancerclass import bayes
from cancerclass.impute import DreamAIImputer
from cancerclass.metrics import calculate_nrmse, calculate_f1, calculate_accuracy,run_masking_experiment
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
    return pipeline.fit(X, Y)

def accuracy(model, X_test: np.ndarray, Y_test: np.ndarray, metric: str):
    metric = __metrics[metric]
    Y_pred = model.predict(X_test)

    return metric(Y_pred, Y_test)

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

def __heatmap(data: np.ndarray, ax, categories=False):
    sns.heatmap(data, ax=ax, vmin=0.0, vmax=1.0, cmap='bwr', annot=True, xticklabels=categories, yticklabels=False, fmt=".2f")

def analysis_pipeline(model, data: tuple, metric: str, category_names: list):
    X_train,X_test,Y_train,Y_test = data
    categories = set(category_names)
    Y_trainp = model.predict(X_train)[:,None]
    Y_testp  = model.predict(X_test)[:,None]
    train_accuracy = np.sum(Y_trainp == Y_train) / len(Y_train)
    test_accuracy  = np.sum(Y_testp == Y_test) / len(Y_test)
    try: 
        train_prob = model.predict_proba(X_train)
        test_prob  = model.predict_proba(X_test)
        fig,ax = plt.subplots(1,2, figsize=(10,12), gridspec_kw={'width_ratios': [4,1]})
        ax[0].set_title("Training confidence")
        __heatmap(train_prob, ax[0], categories)
        ax[1].set_title(f"Accuracy {train_accuracy*100}%")
        __heatmap(Y_trainp == Y_train, ax[1])

        fig,ax = plt.subplots(1,2, figsize=(10,12), gridspec_kw={'width_ratios': [4,1]})
        ax[0].set_title("Testing confidence")
        __heatmap(test_prob, ax[0], categories)
        ax[1].set_title(f"Accuracy {test_accuracy*100}%")
        __heatmap(Y_testp == Y_test, ax[1])

    except Exception as e:
        print(f"Warning in probabilities: {e}")
        pass
