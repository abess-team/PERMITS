import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
import itertools
sns.set_style('whitegrid')
from datetime import datetime

from models import HLasso, HElasticNet, IHT, Scope
from tailscreening import TailScreening, SimplexSolver
import warnings
warnings.filterwarnings('ignore')

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

class Simulator(object):
    def __init__(self,
            n_list,
            p_list,
            s_list,
            corr_list,
            model_list,
            snr_list,
    ):
        self.n_list = n_list
        self.p_list = p_list
        self.s_list = s_list
        self.corr_list = corr_list
        self.model_list = model_list
        self.snr_list = snr_list

    def simulate(self, num_rep=50):
        self.df = pd.DataFrame(columns=['n', 'p', 's', 'corr', 'snr', 'model', 
                                        'exact', 'accuracy', 'inclusion', 
                                        'exclusion', 'err', 'time'])
        counter = 0
        for n, p, s, corr, snr, i in itertools.product(
            n_list, 
            p_list, 
            s_list, 
            corr_list, 
            snr_list, 
            range(num_rep)
        ):
            # make data
            X, y, w_true = make_correlated_data(
                n, p, s, corr=corr, 
                snr=snr, random_state=None
            )
            supp_true = np.nonzero(w_true)[0]
            assert len(supp_true) == s

            # model comparison
            for model in model_list:
                t_begin = time.time()
                if str(model) == 'Oracle':
                    w_est = np.zeros(p)
                    w_est[supp_true] = SimplexSolver().solve(
                        X=X[:, supp_true], y=y
                    ).coef_
                elif str(model) in ['IHT', 'SCOPE']:
                    model.sparsity = s
                    model = model.fit(X=X, y=y)
                    w_est = model.coef_
                else:
                    model = model.fit(X=X, y=y)
                    w_est = model.coef_
                t_end = time.time()
                time_ = t_end - t_begin

                supp_est = np.nonzero(w_est)[0]
                num_tp = len(set(supp_est) & set(supp_true))
                num_fp = len(set(supp_est) - set(supp_true))
                num_fn = len(set(supp_true) - set(supp_est))
                num_tn = len(set(np.arange(p)) - set(supp_true) - set(supp_est))

                exact = int(set(supp_est) == set(supp_true))
                accuracy = num_tp / (num_tp+num_fp+num_fn)
                inclusion = num_tp / s
                if len(supp_est) == 0:
                    exclusion = 1
                else:
                    exclusion = 1 - (num_fp / len(supp_est))
                err = np.linalg.norm(w_true - w_est, ord=2)
                result = {
                    'n': n,
                    'p': p, 
                    's': s,
                    'corr': corr,
                    'snr': snr,
                    'model': str(model), 
                    'exact': exact, 
                    'accuracy': accuracy,
                    'inclusion': inclusion, 
                    'exclusion': exclusion,
                    'err': err, 
                    'time': time_
                }
                self.df.loc[counter] = result
                counter += 1
                print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), result)



if __name__ == '__main__':
    n_list = np.arange(100, 1001, 50)
    p_list = [1000]
    s_list = [10]
    corr_list = [0, 0.5, -0.5]
    snr_list = [1]
    model_list = [
        'Oracle',
        TailScreening(),
        IHT(), 
        HLasso(), 
        # HElasticNet(),
        Scope(),
    ]
    simulator = Simulator(
        n_list=n_list, 
        p_list=p_list, 
        s_list=s_list, 
        corr_list=corr_list,
        snr_list=snr_list,
        model_list=model_list,
    )
    simulator.simulate(num_rep=50)
    df = simulator.df

    df.to_csv('./time_sample_size.csv', index=False)

    # conda activate skscope
    # cd C:\Users\uest\Desktop\new-self-regularization
    # python simu_time.py > simu_time_log.txt
    
    # cd /home/chenpeng/Docs/new-self-regularization
    # nohup python -u simulation.py > simulation_log.txt 2>&1 &
    # jobs -l
    # ps -aux | grep seq_train.py| grep -v grep
    # kill -9 PID

    # nohup python -u seq_test.py > test_log.txt 2>&1 &