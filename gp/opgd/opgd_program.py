import itertools

from gp.opgd.prog_eval import opgd_prg_eval


class OPGDProgram:
    def __init__(self, prg):
        super().__init__()
        self.__prg = prg

    def __call__(self, x):
        return opgd_prg_eval(self.__prg, x)

    def __getitem__(self, item):
        return self.__prg[item]

    def __len__(self):
        return len(self.__prg)

    def parameters(self):
        return itertools.chain.from_iterable([op.parameters() for op in self.__prg])
