from gp.gpgd.gpgd_program import GPGDProgram
from gp.learnable_gp import LearnableLinearGP


class LinearGPGD(LearnableLinearGP):
    def __init__(self, opcodes, params=None):
        super().__init__(opcodes, params)

    def _two_points_crossover(self, x, y):
        of1, of2 = super()._two_points_crossover(x, y)
        return GPGDProgram(self._opcodes, of1), GPGDProgram(self._opcodes, of2)

    def _mutation(self, x):
        mutated_prg = super()._mutation(x)
        return GPGDProgram(self._opcodes, mutated_prg)

    def _random_program(self):
        rand_prg = super()._random_program()
        return GPGDProgram(self._opcodes, rand_prg)
