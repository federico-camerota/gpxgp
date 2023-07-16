import os

import numpy as np
import seaborn as sns
import pandas as pd

from scipy.stats import ranksums


def get_pval(base_dir, alg, ds_name, method):

    dfs = []
    base_dir_cb = base_dir + f'/results{500}'
    li = []
    for filename in os.listdir(base_dir_cb):
        df = pd.read_csv(os.path.join(base_dir_cb, filename), index_col=None, header=0)
        df = df[df['dataset'] == ds_name]

        li.append(df)

    dfs.append(pd.concat(li, axis=0, ignore_index=True))

    df_method = dfs[0].copy()

    df_method = df_method[df_method['alg'] == alg]
    median_method = np.sqrt(df_method['test_fitness'])
    # print(median_method)

    df_gp = dfs[0].copy()

    df_gp = df_gp[df_gp['alg'] == "gp_500_0_0_0.0"]
    median_gp = np.sqrt(df_gp['test_fitness'])
    # print(median_gp)

    p_val = ranksums(x=median_method,
                     y=median_gp,
                     # alternative='less'
                     )

    return p_val


base_dir = "../results_less_ops"
ds_name = "airfoil"

method = "opgd"
variant = "c"
step_after = "5"
alg = f"{method}{variant}_500_0_{step_after}_0.01"

p_val = get_pval(base_dir, alg, ds_name, method=method)
print(p_val)
