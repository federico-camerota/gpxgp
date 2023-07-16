from gp.learnable_gp import LearnableLinearGP
from gp.opgd import operators
from gp.opgd.opgd_program import OPGDProgram


class LinearOPGD(LearnableLinearGP):
    def __init__(self, opcodes, params=None):
        super().__init__(opcodes, params)

        self._op_dict = {'const': operators.ConstantOP}
        try:
            self._op_dict[self._opcodes.PLUS] =  operators.SumOP
        except AttributeError:
            pass
        try:
            self._op_dict[self._opcodes.TIMES] = operators.ProductOP
        except AttributeError:
            pass
        try:
            self._op_dict[self._opcodes.MINUS] = operators.SubtractionOP
        except AttributeError:
            pass
        try:
            self._op_dict[self._opcodes.DIVIDE] = operators.DivisionOP
        except AttributeError:
            pass
        try:
            self._op_dict[self._opcodes.DUP] = operators.DuplicationOP
        except AttributeError:
            pass
        try:
            self._op_dict[self._opcodes.SWAP] = operators.SwapOP
        except AttributeError:
            pass
        try:
            self._op_dict[self._opcodes.NOP] = operators.NopOP
        except AttributeError:
            pass

    def _two_points_crossover(self, x, y):
        of1, of2 = super()._two_points_crossover(x, y)
        return OPGDProgram(of1), OPGDProgram(of2)

    def _mutation(self, x):
        mutated_prg = super()._mutation(x)
        return OPGDProgram(mutated_prg)

    def _random_program(self):
        rand_prg = super()._random_program()
        return OPGDProgram(rand_prg)

    def _random_op(self):
        rand_op = super()._random_op()

        if isinstance(rand_op, int):
            return self._op_dict['const'](float(rand_op))
        else:
            return self._op_dict[rand_op]()
