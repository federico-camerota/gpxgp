import os.path
import shutil

import yaml


def make_fair_params(comp_budget, gd_in_evo_per_iter, gd_after_evo_per_iter):
    gp_params = {'n_iter': comp_budget, 'epochs_in_evolution': 0, 'epochs_after_evolution': 0}

    in_evo_iters = comp_budget // (gd_in_evo_per_iter + 1)
    in_evo_params = {'n_iter': in_evo_iters, 'epochs_in_evolution': gd_in_evo_per_iter, 'epochs_after_evolution': 0}

    after_evo_iters = comp_budget // (gd_after_evo_per_iter + 1)
    after_evo_params = {'n_iter': after_evo_iters, 'epochs_in_evolution': 0,
                        'epochs_after_evolution': comp_budget - after_evo_iters}

    return gp_params, in_evo_params, after_evo_params


def update_yml_hyperparams(file, params):
    with open(file, 'r') as f:
        file_params = yaml.safe_load(f)

    file_params['gp'].update(params)

    new_file = f'{os.path.splitext(file)[0]}_{params["n_iter"]}_{params["epochs_in_evolution"]}_{params["epochs_after_evolution"]}.yml'
    with open(new_file, 'w') as f:
        yaml.safe_dump(file_params, f)

    return new_file
