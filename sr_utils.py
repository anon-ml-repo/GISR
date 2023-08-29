import os
import numpy as np
import sympy as sp
from pysr import PySRRegressor

from tqdm import tqdm


def expr_similarity(expr1, expr2, X_dim, range_min=-1, range_max=1, N=10000):
    X = (range_max - range_min) * np.random.rand(N, X_dim) + range_min
    symbol_string = ', '.join([f'x{i}' for i in range(X_dim)])
    f1 = sp.lambdify(symbol_string, expr1, 'numpy')
    f2 = sp.lambdify(symbol_string, expr2, 'numpy')

    y1 = f1(*[X[:, i] for i in range(X_dim)])
    y2 = f2(*[X[:, i] for i in range(X_dim)])
    return 1/len(y1) * np.sum((y1-y2) ** 2)


def get_all_expressions(sr_data, dataset_name):
    result_dir = "/n/data1/hms/dbmi/zitnik/lab/users/jb607/iterSR/results/"
    eq_file = dataset_name 

    expressions = []
    for i, dataset in tqdm(enumerate(sr_data)):
        sr_model = PySRRegressor(
            niterations=30,
            binary_operators=["+", "-", "*", "/"],
            equation_file = os.path.join(result_dir, eq_file + '/pair-%d.csv' % i),
            progress=False,
            verbosity=0,
            warm_start=False,
            should_optimize_constants=False,
            procs = 0,
            maxdepth=3,
            turbo=True
        )
        
        X = dataset[:, :-1]
        y = dataset[:, -1:]

        X_train = X.detach().cpu().numpy()
        y_train = y.squeeze().detach().cpu().numpy()
        sr_model.fit(X_train, y_train)

        expressions.append(sr_model.sympy())
    return expressions