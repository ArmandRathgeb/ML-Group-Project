from cancerclass import utils

from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import IterativeImputer

def run_pipeline(
        X, Y,
        imputation_type: str = 'MICE',

    ):

    imputation = {
        'NONE' : FunctionTransformer(utils.identity_transform),
        'MICE' : IterativeImputer(max_iter=30),
        #'DREAM': 
    }
    imputer = imputation[imputation_type]

    Xi = imputer.fit_transform(X)