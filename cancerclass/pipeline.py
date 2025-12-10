from cancerclass import utils
from cancerclass import bayes
from cancerclass.impute import DreamAIImputer
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
