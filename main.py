import argparse
import os.path

from experiments.run_exp import run_exp_on_dataset
from experiments.utils import make_fair_params, update_yml_hyperparams
from gp.gp import LinearGP
from gp.gpgd.linear_gpgd import LinearGPGD
from gp.opgd.linear_opgd import LinearOPGD
from utils.utils import read_yaml_hyperparams

parser = argparse.ArgumentParser()
parser.add_argument('alg', type=str)
parser.add_argument('dataset', type=str)
parser.add_argument('save_dir', type=str)
parser.add_argument('hyperparams_file', type=str)
parser.add_argument('comp_budget', type=int)
parser.add_argument('e_in_evo', type=int)
parser.add_argument('e_after_evo', type=int)
parser.add_argument('lr', type=float)
args = parser.parse_args()

exp_params = {'split_seed': 123, 'n_splits': 30, 'max_splits': 100, 'seeds': [123, 794, 377, 366, 431],
              'dataset': args.dataset, 'run_name': f'{args.alg}_{args.comp_budget}_{args.e_in_evo}_{args.e_after_evo}_{args.lr}'}

gp_params, in_evo_params, after_evo_params = make_fair_params(args.comp_budget, args.e_in_evo, args.e_after_evo)

if args.alg == 'gp':
    alg = LinearGP
    fit_type = 'gp'
    hyperparams_file = update_yml_hyperparams(args.hyperparams_file, gp_params)
elif args.alg == 'gpgda':
    alg = LinearGPGD
    fit_type = 'gpgd'
    hyperparams_file = update_yml_hyperparams(args.hyperparams_file, in_evo_params)
elif args.alg == 'gpgdc':
    alg = LinearGPGD
    fit_type = 'gpgd'
    hyperparams_file = update_yml_hyperparams(args.hyperparams_file, after_evo_params)
elif args.alg == 'opgda':
    alg = LinearOPGD
    fit_type = 'gpgd'
    hyperparams_file = update_yml_hyperparams(args.hyperparams_file, in_evo_params)
elif args.alg == 'opgdc':
    alg = LinearOPGD
    fit_type = 'gpgd'
    hyperparams_file = update_yml_hyperparams(args.hyperparams_file, after_evo_params)
else:
    raise ValueError()

logger = None

save_dir = os.path.join(args.save_dir, f'results{args.comp_budget}')
if not os.path.isdir(args.save_dir):
    os.mkdir(args.save_dir)
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

gp_hyperparams = read_yaml_hyperparams(hyperparams_file)
gp_hyperparams['gp']['lr'] = args.lr
gp_best, gp_train_fit, gp_test_fit = run_exp_on_dataset(exp_params, fit_type=fit_type, alg=alg,
                                                        alg_params=gp_hyperparams, logger=logger, save_dir=save_dir)
