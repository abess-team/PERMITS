import numpy as np

def make_correlated_data(
        n_samples, 
        n_features, 
        sparsity=None,
        w_true=None,
        corr=0, 
        snr=1,
        random_state=None,
):      
        rng = np.random.default_rng(random_state)
        if (w_true is None) and (not sparsity is None):
            assert isinstance(sparsity, int)
            supp = rng.choice(np.arange(n_features), sparsity, replace=False)
            w_true = np.zeros(n_features)
            w_true[supp] = 1 / sparsity
        elif not w_true is None:
            w_true = np.array(w_true).reshape(-1)
        else:
            raise ValueError('sparsity and w_true can not be None simultaneously.')
        assert len(w_true) == n_features 

        cov_matrix = np.fromfunction(
            lambda i, j: (corr) ** np.abs(i-j), 
            shape=(n_features, n_features)
        )
        
        X = rng.multivariate_normal(
            mean=np.zeros(n_features),
            cov=cov_matrix,
            size=n_samples
        )

        signal = X @ w_true
        sigma = np.std(signal) / np.sqrt(snr)  # compute the noise-level according to SNR
        noise = rng.standard_normal(n_samples) * sigma
        y = X @ w_true + noise
        return X, y, w_true