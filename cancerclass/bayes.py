from sklearn.base import BaseEstimator
import numpy as np
import pymc as pm
import pytensor as pt
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
            burn_in: int          = 1000,
            max_iter: int         = 1000,
            chains: int           = 4,
            seed: None|int        = None
        ):
        """Bayesian logistic regression.

        Args:
            n_categories (str | int, optional): Number of categories to classify. If 'auto', defaults to number of classes.
                If < n_classes, appends a zero column and uses it as the baseline. Defaults to 'auto'.
            link (str, optional): Link function ['softmax', 'logit']. Defaults to 'softmax'.
            likelihood (str, optional): Likelihood function ['categorical', 'bernoulli']. Defaults to 'categorical'.
            weight_prior (str, optional): Weight matrix prior ['normal', 'halfnormal', 'laplace', 'uniform']. Defaults to 'normal'.
            intercept_prior (str, optional): Intercept matrix prior. Same options as `weight_prior`. Defaults to 'normal'.
            burn_in (int, optional): Tuning before sampling in earnest. Defaults to 1000.
            max_iter (int, optional): Maximum number of samples to draw. Defaults to 1000.
            chains (int, optional): Number of inference chains to run. Helps with convergence. Defaults to 4.
            seed (None | int, optional): Random seed. Defaults to None.
        """
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
        """Fit model to data.

        Args:
            X: (n_samples, n_features): Input data.
            Y: (n_samples,): Labels

        Returns:
            BayesLogisticRegression: Fitted model
        """
        unique_categories = len(np.unique(Y))
        if self.n_categories == 'auto':
            self.n_categories = unique_categories
        n_samples, n_features = X.shape

        with self.model:
            weights = self.weight_prior('w', (self.n_categories, n_features))
            intercepts = self.intercept_prior('b', (self.n_categories,))

            if self.n_categories < unique_categories:
                zeros_w    = pm.math.zeros((1, weights.shape[1]))
                zeros_b    = pm.math.zeros((1,))
                weights    = pm.math.concatenate([zeros_w, weights], axis=0)
                intercepts = pm.math.concatenate([zeros_b, intercepts], axis=0)

            g_x = pm.math.dot(X, weights.T) + intercepts

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
        """Predict the probability of each class.

        Args:
            X: (n_samples, n_features): Input data.

        Returns:
            np.ndarray: (n_samples, n_classes): Probabilities of being each class.
        """
        weights = self.post['w'].values
        intercepts = self.post['b'].values
        g_x = np.einsum('ij,kjs->iks', X, weights) + intercepts[None,...]
        pi_x = self.eval_link(g_x, axis=1)
        pi_x = pi_x.mean(axis=2)
        return pi_x


    def predict(self, X):
        """Predict which class each sample is.

        Args:
            X: (n_samples, n_features): Input data

        Returns:
            np.ndarray: (n_samples,): Predicted class
        """
        pi_x = self.predict_proba(X)
        return np.argmax(pi_x, axis=1)
