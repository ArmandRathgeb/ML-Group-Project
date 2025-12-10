from sklearn.base import BaseEstimator
import numpy as np
import pymc as pm
import arviz as az
from scipy.special import softmax, logit
from cancerclass import utils

__links__ = {
    'softmax': pm.math.softmax,
    'logit'  : pm.math.logit,
}

__likelihoods__ = {
    'categorical': pm.Categorical,
    'bernouilli' : pm.Bernoulli,
}

__eval_links__ = {
    'softmax' : softmax,
    'logit'   : logit,
}

__priors__ = {
    'normal'    : lambda n,s: pm.Normal(n, mu=0.0, sigma=1.0, shape=s),
    'halfnormal': lambda n,s: pm.HalfNormal(n, sigma=1.0, shape=s),
    'laplace'   : lambda n,s: pm.Laplace(n, mu=0.0, b=1.0, shape=s),
    'uniform'   : lambda n,s: pm.Uniform(n, lower=0.0, upper=1.0, shape=s),
}

class BayesLogisticRegression(BaseEstimator):
    def __init__(self,
            n_categories: str|int = 'auto',
            link: str             = 'softmax', 
            likelihood: str       = 'categorical', 
            weight_prior: str     = 'normal', 
            intercept_prior: str  = 'normal',
            burn_in: int          = 100,
            max_iter: int         = 100,
            chains: int           = 4,
            seed: None|int        = None
        ):
        self.link            = __links__[link]
        self.eval_link       = __eval_links__[link]
        self.likelihood      = __likelihoods__[likelihood]
        self.weight_prior    = __priors__[weight_prior]
        self.intercept_prior = __priors__[intercept_prior]

        self.n_categories = n_categories
        self.sampler_kwargs = {
            'random_seed': seed,
            'draws': max_iter,
            'tune': burn_in,
            'chains': chains,
        }

        self.model = pm.Model()


    def fit(self, X, Y):
        unique_categories = len(np.unique(Y))
        if self.n_categories == 'auto':
            self.n_categories = unique_categories
        n_samples, n_features = X.shape

        with self.model:
            weights = self.weight_prior('w', (self.n_categories, n_features))
            intercepts = self.intercept_prior('b', (self.n_categories,))

            g_x = pm.math.dot(X, weights.T) + intercepts
            if self.n_categories < unique_categories:
                zeros_col = pm.math.zeros((g_x.shape[0], 1))
                g_x = pm.math.concatenate([g_x, zeros_col], axis=1)

            #pi_x = self.link(g_x)
            y_obs = self.likelihood('y_obs', logit_p=g_x, observed=Y)
            trace = pm.sample(**self.sampler_kwargs,
                                target_accept=0.9, 
                                nuts_sampler='numpyro',
                                cores=None, 
                                progressbar=True,
                                )

        self.post = az.extract(trace)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        weights = self.post['w'].values
        intercepts = self.post['b'].values
        g_x = np.einsum('ij,kjs->iks', X, weights) + intercepts[None,...]
        pi_x = self.eval_link(g_x, axis=1)
        pi_x = pi_x.mean(axis=2)
        return pi_x


    def predict(self, X):
        pi_x = predict_proba(self, X)
        return np.argmax(pi_x, axis=1)
