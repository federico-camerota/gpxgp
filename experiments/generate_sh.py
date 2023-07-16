import os
import shutil

algs = ['gp', 'gpgda', 'opgda']
comp_budgets = [50, 100, 200, 500]
setups = [(1, 0), (2, 0), (5, 0), (10, 0)]
lr = [0.1, 0.01, 0.001]
datasets = [d for d in os.listdir('../datasets') if os.path.isdir(os.path.join('../datasets', d))]
save_dir = 'results_by_lr'

base_file = '../jobs/run.sh'

for cb in comp_budgets:
    for alg in algs:
        if alg == 'gp':
            file_name = f'jobs/run_gp_{cb}.sh'
            shutil.copyfile(base_file, file_name)
            with open(file_name, 'a') as f:
                for ds in datasets:
                    f.write(f'python3 main.py gp {ds} {save_dir} hyperparams_yaml/gp_params.yaml {cb} 0 0 0\n')
        else:
            for l in lr:
                for setup in setups:
                    file_name = f'jobs/run_{alg}_{cb}_{setup[0]}_{setup[1]}_{l}.sh'
                    shutil.copyfile(base_file, file_name)
                    with open(file_name, 'a') as f:
                        for ds in datasets:
                            f.write(
                                f'python3 main.py {alg} {ds} {save_dir} hyperparams_yaml/{alg}_params.yaml {cb} {setup[0]} {setup[1]} {l}\n')
