from gp.prog_eval import ProgramEvaluator


class GPGDProgram:
    def __init__(self, opcodes, prg):
        self.__prg_eval = ProgramEvaluator(opcodes)
        self.__prg = prg

    def __call__(self, x):
        return self.__prg_eval(x, self.__prg)

    def __getitem__(self, item):
        return self.__prg[item]

    def __len__(self):
        return len(self.__prg)

    def parameters(self):
        return self.__prg_eval.parameters()
