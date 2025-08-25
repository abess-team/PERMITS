import time
import sys
sys.path.append('../')
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style('whitegrid')
from datetime import datetime

from models import HLasso, HElasticNet, IHT, Scope
from tailscreening import TailScreening, SimplexSolver
from utils import make_correlated_data
import warnings
warnings.filterwarnings('ignore')

class Simulator(object):
    def __init__(self,
            n_list,
            p_list,
            s_list,
            corr_list,
            snr_list,
            model_list,
    ):
        self.n_list = n_list
        self.p_list = p_list
        self.s_list = s_list
        self.corr_list = corr_list
        self.snr_list = snr_list
        self.model_list = model_list

    def simulate(self, num_rep=50):
        self.df = pd.DataFrame(columns=['n', 'p', 's', 'corr', 'snr', 'model', 
                                        'exact', 'accuracy', 'inclusion', 
                                        'exclusion', 'err', 'time'])
        counter = 0
        for n, p, s, corr, snr, i in itertools.product(
            self.n_list, 
            self.p_list, 
            self.s_list, 
            self.corr_list, 
            self.snr_list, 
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
            for model in self.model_list:
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
    n_list = np.arange(100, 601, 50)
    p_list = [1000]
    s_list = [10]
    corr_list = [0, 0.5, -0.5]
    snr_list = [0.5, 1, 5]
    model_list = [
        'Oracle',
        TailScreening(tol_multiple=1e-3),
        TailScreening(tol_multiple=1e-4),
        TailScreening(tol_multiple=1e-5),
        IHT(), 
        HLasso(), 
    ]
    for model in model_list:
        if str(model) == 'PERMITS':
            model.name = 'PERMITS-' + str(model.tol_multiple)
    for model in model_list:
        print(str(model))
    
    # accuracy and error with n varied
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
    df.to_csv('./tol_acc_error.csv', index=False)


    # time with p varied
    simulator = Simulator(
        n_list=[500], 
        p_list=np.arange(100, 1001, 50), 
        s_list=s_list, 
        corr_list=corr_list,
        snr_list=snr_list,
        model_list=model_list,
    )
    simulator.simulate(num_rep=50)
    df = simulator.df
    df.to_csv('./tol_time.csv', index=False)