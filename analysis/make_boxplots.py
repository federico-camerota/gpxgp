import os

import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rc('font', **{'size': 16, 'weight': 'bold'})
sns.set_theme(style="whitegrid")
# sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 1.5})


def mk_bp(base_dir, path_bp, comp_budgets, method=None, is_log=0):
    if not os.path.exists(path_bp):
        os.mkdir(path_bp)
    if method and method not in ["gpgd", "opgd", "gpgda", "opgda", "gpgdc", "opgdc"]:
        print("the method you insert is not valid")
        return

    dfs = []
    for cb in comp_budgets:
        base_dir_cb = base_dir + f'/results{cb}'
        li = []
        for filename in os.listdir(base_dir_cb):
            df = pd.read_csv(os.path.join(base_dir_cb, filename), index_col=None, header=0)
            df = df[(df['lr'] == 0.01) | (df['lr'] == 0.0)]
            df = df[(df['epochs_after_evolution'] != 375) & (df['epochs_after_evolution'] != 334)]
            df = df[(df['epochs_in_evolution'] != 3) & (df['epochs_in_evolution'] != 2)]

            if method:
                df = df[df.alg.str.startswith(tuple([method, "gp_"]))]
            li.append(df)

        dfs.append(pd.concat(li, axis=0, ignore_index=True))

    # sns.set(rc={"figure.figsize": (15, 12)})

    for ds_name in ["yacht", "bioav", "slump", "toxicity", "airfoil", "concrete", "ppb", "parkinson"]:
        for index in range(len(comp_budgets)):
            df = dfs[index].copy()

            x_order = df['alg'].unique()
            x_order = list(sorted(x_order))

            x_order[1], x_order[2], x_order[3] = x_order[2], x_order[3], x_order[1]
            x_order[4], x_order[5], x_order[6] = x_order[5], x_order[6], x_order[4]

            df = df[df['dataset'] == ds_name]

            # df_median = df.groupby(['alg'])['test_fitness'].median()
            colors_g = ['#5f187f'] + ['#d3436e' for _ in range(3)] + ['#febb81' for _ in range(3)]
            colors_o = ['#5f187f'] + ['#37659e' for _ in range(3)] + ['#8bdab2' for _ in range(3)]

            colors = colors_g if method == "gpgd" else colors_o

            # colors = sns.color_palette("mako")
            # print(colors.as_hex())

            bp = sns.boxplot(df, y=df['alg'],
                             x=np.sqrt(df['test_fitness']),
                             showfliers=False,
                             order=x_order,
                             palette=colors,
                             saturation=0.75,
                             width=0.8,
                             linewidth=2
                             )

            # bp.tick_params(bottom=False)

            if method == "gpgd":
                my_patch1 = mpatches.Patch(color=colors_g[0], label='GP')
                my_patch2 = mpatches.Patch(color=colors_g[1], label='GPGD-A')
                my_patch3 = mpatches.Patch(color=colors_g[-1], label='GPGD-C')
                plt.legend(handles=[my_patch1, my_patch2, my_patch3])

            if method == "opgd":
                my_patch1 = mpatches.Patch(color=colors_o[0], label='GP')
                my_patch2 = mpatches.Patch(color=colors_o[1], label='OPGD-A')
                my_patch3 = mpatches.Patch(color=colors_o[-1], label='OPGD-C')
                plt.legend(handles=[my_patch1, my_patch2, my_patch3])

            if is_log:
                bp.set_xscale("log")

            bp.set_yticks(range(7))
            if method == "gpgd":
                bp.set_yticklabels([None, '1', '5', '10', '1', '5', '10'])
            if method == "opgd":
                bp.set_yticklabels([None, '1', '5', '10', '1', '5', '10'])

            bp.set_xlabel(None)
            bp.set_ylabel(None)

            # plt.title(f'{ds_name}: {comp_budgets[index]}_{method}')
            bp.tick_params(left=False, bottom=False)
            if is_log:
                plt.savefig(path_bp + f'/{ds_name}_{comp_budgets[index]}_{method}_log.png')
            else:
                plt.savefig(path_bp + f'/{ds_name}_{comp_budgets[index]}_{method}.png')
            plt.close()


comp_budgets = [500]
base_dir = "../results_less_ops"
path_bp = '../plots/paper/'
is_log = 0  # flag to decide if we want logarithmic plots
mk_bp(base_dir, path_bp, comp_budgets, method="gpgd", is_log=is_log)
mk_bp(base_dir, path_bp, comp_budgets, method="opgd", is_log=is_log)
