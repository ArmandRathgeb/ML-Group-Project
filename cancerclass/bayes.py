from sklearn.base import BaseEstimator
import numpy as np
import pymc as pm
#import arviz as az
from cancerclass import utils

__links__ = {
    'softmax': pm.math.softmax,
    'logit'  : pm.math.logit,
}

__likelihoods__ = {
    'categorical': pm.Categorical,
    #'bernouilli' : ,
    #'poisson'    : torch.poisson
}

__priors__ = {
    'normal'    : lambda n,s: pm.Normal(n, mu=0.0, sigma=1.0, shape=s),
    'halfnormal': lambda n,s: pm.HalfNormal(n, sigma=1.0, tau=1.0, shape=s),
    'laplace'   : lambda n,s: pm.Laplace(n, mu=0.0, b=1.0, shape=s),
    'uniform'   : lambda n,s: pm.Uniform(n, lower=0.0, upper=1.0, shape=s),
}

__loss__ = {
    'cce': None
}

class BayesLogisticRegression(BaseEstimator):
    def __init__(self,
            n_categories:int='auto',
            link='softmax', 
            likelihood='categorical', 
            weight_prior='normal', 
            intercept_prior='halfnormal',
            loss='cce',
            learning_rate=.001,
            batch_size=32,
            max_iter=100,
            seed=None
        ):
        self.link            = __links__[link]
        self.likelihood      = __likelihoods__[likelihood]
        self.weight_prior    = __priors__[weight_prior]
        self.intercept_prior = __priors__[intercept_prior]
        self.loss            = __loss__[loss]

        self.max_iter      = max_iter
        self.batch_size    = batch_size
        self.learning_rate = learning_rate

        self.n_categories = n_categories

        self.model = pm.Model()


    def fit(self, X, Y):
        if self.n_categories == 'auto':
            self.n_categories = len(np.unique(Y))
        n_samples, n_features = X.shape

        with self.model:
            sigma_w = pm.HalfNormal('sigma_w', sigma=1.0)
            weights = pm.Normal('w', mu=0.0, sigma=sigma_w, size=(self.n_categories, n_features))
            intercepts = pm.Normal('b', size=(self.n_categories,))
            #weights = pm.Laplace('w', mu=0.0, b=1.0, size=(self.n_categories, n_features))
            #intercepts = pm.Laplace('b', mu=0.0, b=1.0, size=(self.n_categories,))

            g_x = X @ weights.T + intercepts

            #pi_x = self.link(g_x)
            pi_x = pm.math.log_softmax(g_x)
            print("Linked")
            #y_obs = pm.Categorical('y_obs', p=pi_x, observed=Y)
            y_obs = pm.Multinomial('y_obs', n=self.n_categories - 1, p=pi_x, observed=Y)
            print("Started sampling")
            trace = pm.sample(draws=100, 
                                tune=100, 
                                chains=4, 
                                target_accept=0.95, 
                                return_inferencedata=True, 
                                cores=None, 
                                n_init=50, 
                                progressbar=True)

        self.weights = weights.eval()
        self.intercepts = intercepts.eval()
        return self,trace

    def predict(self, X):
        g_x = X @ self.weights.T + self.intercepts
        #pi_x = self.link(torch.from_numpy(g_x))
        #return utils.nup(torch.multinomial(pi_x, num_samples=1)),pi_x
