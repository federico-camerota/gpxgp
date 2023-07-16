import copy
import multiprocessing
import random

import torch

GP_DEFAULT_PARAMS = {
    'p_m': 0.2,
    'min_con': -5,
    'max_con': 5,
    'pop_size': 100,
    'n_iter': 100,
    'dim_prg': 10,
    'max_dim_prg': 10,
    'p_rand_op': 0.5,
    'rand_op_min': -5,
    'rand_op_max': 5,
    'tournament_size': 4
}


class LinearGP:
    def __init__(self, opcodes, params=None):
        # Initialize parameters with default values
        self._params = copy.deepcopy(GP_DEFAULT_PARAMS)
        # Update parameters with user provided values
        if params is not None:
            self._params.update(params)

        self._opcodes = opcodes

    def _random_op(self):
        if random.random() < self._params['p_rand_op']:  # 0.5
            op = random.randint(self._params['rand_op_min'], self._params['rand_op_max'])
        else:
            op = random.choice(list(self._opcodes))

        return op

    def _random_program(self):
        return [self._random_op() for _ in range(self._params['dim_prg'])]

    def _tournament_selection(self, fitness, pop):
        tournament = random.choices(range(len(pop)), k=self._params['tournament_size'])
        return pop[min(tournament, key=lambda x: fitness[x])]

    def _two_points_crossover(self, x, y):
        k1 = random.randint(0, len(x) - 1)
        k2 = random.randint(k1, len(x) - 1)
        h1 = random.randint(0, len(y) - 1)
        h2 = random.randint(h1, len(y) - 1)
        of1 = copy.deepcopy(x[0:k1]) + copy.deepcopy(y[h1:h2]) + copy.deepcopy(x[k2:])
        of2 = copy.deepcopy(y[0:h1]) + copy.deepcopy(x[k1:k2]) + copy.deepcopy(y[h2:])

        if len(of1) > self._params['max_dim_prg']:
            of1 = of1[:self._params['max_dim_prg']]
        if len(of2) > self._params['max_dim_prg']:
            of2 = of2[:self._params['max_dim_prg']]

        return of1, of2

    def _mutate_op(self, b):
        if random.random() < self._params['p_m']:
            return self._random_op()
        else:
            return b

    def _mutation(self, x):
        mutated_prg = [self._mutate_op(b) for b in x]
        return mutated_prg

    def fit(self, fit, verbose=False):

        pop = [self._random_program() for _ in range(0, self._params['pop_size'])]  # 10

        best = self._random_program()  # []
        fit_best = fit(best)
        pop.append(best)

        pool = multiprocessing.Pool()
        fitness = torch.Tensor(pool.map(fit, pop))

        for i in range(0, self._params['n_iter']):

            pop = self._fit_iteration(fit, pop, fitness)
            pop.append(best)
            fitness = torch.Tensor(pool.map(fit, pop))

            candidate_best = torch.argmin(fitness)
            if fitness[candidate_best] < fit_best:
                best = pop[candidate_best]
                fit_best = fitness[candidate_best]


            if verbose:
                print(f'Best fitness at iteration {i}: {fit_best}')

        return best, fit_best

    def _fit_iteration(self, fit, pop, fitness):

        selected = [self._tournament_selection(fitness, pop) for _ in range(0, self._params['pop_size'])]
        pairs = zip(selected, selected[1:] + [selected[0]])
        offsprings = []
        for x, y in pairs:
            of1, of2 = self._two_points_crossover(x, y)
            offsprings.append(of1)
            offsprings.append(of2)

        pop = [self._mutation(x) for x in offsprings]

        return pop

    def dump_params(self):
        return self._params
