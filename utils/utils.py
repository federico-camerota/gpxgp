import random
import enum

import numpy as np
import torch
import yaml

from gp.prog_eval import ProgramEvaluator


def read_yaml_hyperparams(filename):
    with open(filename, 'r') as f:
        hyperparams = yaml.safe_load(f)

    return hyperparams


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_fit_fun_gp(x, y, opcodes, reduction='mean'):
    pg_eval = ProgramEvaluator(opcodes)
    if reduction == 'mean':
        return lambda prg: torch.mean((y - pg_eval(x, prg)) ** 2)
    if reduction == 'sum':
        return lambda prg: torch.sum((y - pg_eval(x, prg)) ** 2)


def make_fit_fun_gpgd(x, y, reduction='mean'):
    """
    fitness function for gpgd routine
    """
    if reduction == 'mean':
        return lambda prg: torch.mean((y - prg(x)) ** 2)
    elif reduction == 'sum':
        return lambda prg: torch.sum((y - prg(x)) ** 2)
    else:
        return lambda prg: torch.mean((y - prg(x)) ** 2)

