import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, ElasticNetCV
# from tailscreening import TailScreening
import jax.numpy as jnp
from skscope import ScopeSolver, IHTSolver


# Heuristic Lasso
class HLasso(object):
    def __init__(self, n_alphas=100, alphas=None, n_jobs=None):
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.n_jobs = n_jobs
    
    def __str__(self):
        return 'H-Lasso'

    def fit(self, X, y):
        lasso = LassoCV(
            cv=5, 
            n_alphas = self.n_alphas,
            alphas=self.alphas, 
            positive=True, 
            n_jobs=self.n_jobs
        ).fit(X, y)
        if lasso.coef_.sum() == 0:
            self.coef_ = np.zeros_like(lasso.coef_)
        else:
            self.coef_ = lasso.coef_ / lasso.coef_.sum()
        return self

# Heuristic ElasticNet
class HElasticNet(object):
    def __init__(self, alphas=None):
        self.alphas = alphas
    
    def __str__(self):
        return 'H-ElasticNet'

    def fit(self, X, y):
        elasticnet = ElasticNetCV(cv=5, alphas=self.alphas, positive=True, n_jobs=20).fit(X, y)
        if elasticnet.coef_.sum() == 0:
            self.coef_ = np.zeros_like(elasticnet.coef_)
        else:
            self.coef_ = elasticnet.coef_ / elasticnet.coef_.sum()
        return self

# IHT
class IHT(object):
    def __init__(self, sparsity=None, eta=1e-2, tol=1e-9):
        self.sparsity = sparsity
        self.eta = eta
        self.tol = tol
    
    def __str__(self):
        return 'IHT'
    
    def get_loss(self, X, y, w):
        loss = np.mean((X @ w - y) ** 2) / 2
        return loss
    
    def get_grad(self, X, y, w):
        n = len(y)
        grad = X.T @ (X @ w - y) / n
        return grad
    
    def argmax(self, arr, k):
        '''get the index of k largest elements of arr'''
        arr = np.array(arr)
        ind = arr.argsort()[::-1][:k]
        return np.sort(ind)
    
    def simplex_proj(self, y, a=1):
        '''project the vector onto the simplex whose sum is a[=1]'''
        y = np.array(y)
        p = y.shape[0]
        zeros = np.zeros(p)
        u = np.sort(y)[::-1]
        t = u + (a - np.cumsum(u)) / (np.arange(p) + 1)
        rho = np.sum(t > 0)
        lam = (t - u)[rho - 1]
        return np.maximum(y+lam, zeros)

    def sparse_proj(self, y, k):
        ind = self.argmax(y, k)
        y_k = y[ind].copy()
        y = np.zeros_like(y)
        y[ind] = self.simplex_proj(y_k)
        return y
    
    def fit(self, X, y):
        n, p = X.shape
        w = np.ones(p) / p
        while True:
            u = w.copy()
            grad = self.get_grad(X, y, u)
            u -= self.eta * grad
            u = self.sparse_proj(u, self.sparsity)
            if np.linalg.norm(u-w) <= self.tol:
                break
            else:
                w = u.copy()
        self.coef_ = u
        return self

class Scope(object):
    def __init__(self, sparsity=5):
        self.sparsity = sparsity
    
    def __str__(self):
        return 'SCOPE'

    def fit(self, X, y):
        n, p = X.shape          
        init_params = jnp.ones(p) / p

        def custom_objective(params):
            params = jnp.square(params) / jnp.sum(jnp.square(params))
            loss = jnp.mean((X @ params - y) ** 2)
            return loss

        solver = ScopeSolver(p, self.sparsity)
        params = solver.solve(custom_objective, init_params=init_params)
        self.coef_ = np.square(params) / np.sum(np.square(params))
        return self



if __name__ == '__main__':
    n, p, s = 300, 500, 10
    X = np.random.randn(n, p)
    beta = np.zeros(p)
    beta[:s] = np.ones(s) / s
    assert np.abs(beta.sum() - 1) < 1e-10
    Xbeta = X @ beta
    snr = 1
    sigma = np.sqrt(np.var(Xbeta) / snr) 
    noise = np.random.randn(n) * sigma
    y = Xbeta + noise

    model = Scope(s)
    model.fit(X, y)
    model.coef_

#     # alphas = np.logspace(-3, 5, 100)
#     alphas = np.linspace(0.1, 100, 50)

    model1 = HLasso(n_jobs=1)
    model1 = model1.fit(X, y)
    support1 = np.nonzero(model1.coef_)[0]
    print('S1: ', support1)

#     model2 = elasticnet_h(alphas)
#     model2 = model2.fit(X, y)
#     support2 = np.nonzero(model2.coef_)[0]
#     print('S2: ', support2)

#     model3 = iht(s)
#     model3 = model3.fit(X, y)
#     support3 = np.nonzero(model3.coef_)[0]
#     print('S3: ', support3)
#     print('Err IHT: ', np.linalg.norm(beta - model3.coef_, ord=1).round(5))

#     oracle = np.zeros(p)
#     model_tmp = tail_screening()
#     oracle[:s] = model_tmp.grad_method(X=X[:, :s], 
#                                         y=y, 
#                                         w_init=np.ones(s)/s,
#                                         prec=True)
#     print('Err Oracle: ', np.linalg.norm(beta - oracle, ord=1).round(5))

#     np.sqrt(sigma**2 * np.log(p) / n)