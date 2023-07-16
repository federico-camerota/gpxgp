import enum
import os.path
from multiprocessing import Pool, cpu_count

import numpy as np
import itertools

import pandas as pd
import torch

from utils.dataset import make_train_test_ds
from utils.utils import set_seeds, make_fit_fun_gp, make_fit_fun_gpgd


def run_experiment(run_params, fit_type, alg, alg_params, verbose=False, dump_params=False):
    set_seeds(seed=run_params['seed'])

    gp_params = alg_params['gp']
    opcodes = enum.Enum('opcodes', gp_params['enum_set'])

    name_ds = run_params['name_ds']
    X_train, y_train, X_test, y_test = make_train_test_ds(name_ds, run_params['split'])

    x_train_list = [x_i for x_i in X_train.T]
    x_test_list = [x_i for x_i in X_test.T]
    y_train.requires_grad_()

    if fit_type == 'gp':
        train_fit_fun = make_fit_fun_gp(x_train_list, y_train, opcodes, reduction='mean')
        test_fit_fun = make_fit_fun_gp(x_test_list, y_test, opcodes, reduction='mean')
    else:
        train_fit_fun = make_fit_fun_gpgd(x_train_list, y_train, reduction='mean')
        test_fit_fun = make_fit_fun_gpgd(x_test_list, y_test, reduction='mean')

    solver = alg(opcodes, gp_params)
    best_prg, best_val = solver.fit(train_fit_fun, verbose)

    with torch.no_grad():
        test_fitness = test_fit_fun(best_prg)

    if dump_params:
        return best_prg, best_val.item(), test_fitness.item(), solver.dump_params()
    else:
        return best_prg, best_val.item(), test_fitness.item()


def run_exp_on_dataset(exp_params, fit_type, alg, alg_params, logger=None, verbose_level=0, dump_params=True, save_dir=''):
    run_params = {'name_ds': exp_params['dataset']}
    np.random.seed(exp_params['split_seed'])
    splits = np.random.choice(exp_params['max_splits'], exp_params['n_splits'])

    best_prgs, train_fitness, test_fitness = [], [], []

    with Pool(cpu_count()) as p:
        input = itertools.product([logger], [exp_params], [run_params], splits, exp_params['seeds'], [fit_type], [alg],
                                  [alg_params], [verbose_level], [dump_params])
        results = p.starmap(exp_fun, input)

    for out in results:
        train_fitness.append(out['train_fitness'])
        test_fitness.append(out['test_fitness'])

    if verbose_level > 0:
        print(
            f'{exp_params["run_name"]} (dataset: {exp_params["dataset"]}) -> train avg: {np.mean(train_fitness):.5f} test avg: {np.mean(test_fitness):.5f}')

    if dump_params is not None:
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(save_dir, f'{exp_params["run_name"]}__{exp_params["dataset"]}'))

    return best_prgs, train_fitness, test_fitness


def exp_fun(logger, exp_params, run_params, split, seed, fit_type, alg, alg_params, verbose_level, dump_params):
    run_params['split'] = split
    run_params['seed'] = seed

    if dump_params:
        best_prg, train_fit, test_fit, alg_params = run_experiment(run_params, fit_type, alg, alg_params,
                                                   verbose=True if verbose_level > 2 else False, dump_params=dump_params)
        run_params.update(alg_params)
    else:
        best_prg, train_fit, test_fit = run_experiment(run_params, fit_type, alg, alg_params,
                                                                   verbose=True if verbose_level > 2 else False, dump_params=dump_params)

    if logger is not None:
        neptune_run_prefix = f'{exp_params["dataset"]}/{exp_params["run_name"]}'
        logger.log_metric(neptune_run_prefix + '/train_fit', train_fit)
        logger.log_metric(neptune_run_prefix + '/test_fit', test_fit)

    if verbose_level > 1:
        print(
            f'{exp_params["run_name"]} (dataset: {exp_params["dataset"]}, seed: {run_params["seed"]}, split {run_params["split"]}) -> train: {train_fit:.5f} test: {test_fit:.5f}')

    out = {'alg': exp_params['run_name'], 'dataset': exp_params['dataset'], 'train_fitness': train_fit, 'test_fitness': test_fit}
    out.update(run_params)
    return out
