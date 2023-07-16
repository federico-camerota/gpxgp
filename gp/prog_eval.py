import torch
import torch.nn as nn


class ProgramEvaluator(nn.Module):

    def __init__(self, opcodes):
        super().__init__()
        self._sum_w1 = nn.parameter.Parameter(torch.tensor([1.]))
        self._sum_w2 = nn.parameter.Parameter(torch.tensor([1.]))
        self._prod_w1 = nn.parameter.Parameter(torch.tensor([1.]))
        self._prod_w2 = nn.parameter.Parameter(torch.tensor([1.]))
        self._div_w1 = nn.parameter.Parameter(torch.tensor([1.]))
        self._div_w2 = nn.parameter.Parameter(torch.tensor([1.]))
        self._diff_w1 = nn.parameter.Parameter(torch.tensor([1.]))
        self._diff_w2 = nn.parameter.Parameter(torch.tensor([1.]))
        # self._mod_w1 = nn.parameter.Parameter(torch.tensor([1.]))
        # self._mod_w2 = nn.parameter.Parameter(torch.tensor([1.]))
        #self._dup_w = nn.parameter.Parameter(torch.tensor([1.]))
        #self._swap_w1 = nn.parameter.Parameter(torch.tensor([1.]))
        #self._swap_w2 = nn.parameter.Parameter(torch.tensor([1.]))

        self._const_w = nn.parameter.Parameter(torch.tensor([1.]))

        self._opcodes = opcodes

    def __call__(self, x, program):
        """
        Function that takes a program and variables as input and returns the result of performing the described program
        :param x: variables for the program
        :param program: program considered
        :return: function that emulates the program
        """
        try:
            stack = x.copy()

            if type(stack) is not list:
                stack = stack.tolist()
            while program:
                op = program[0]
                program = program[1:]
                if op == self._opcodes.PLUS:
                    op1 = stack.pop()
                    op2 = stack.pop()
                    stack.append(self._sum_w1 * op1 + self._sum_w2 * op2)
                elif op == self._opcodes.MINUS:
                    op1 = stack.pop()
                    op2 = stack.pop()
                    stack.append(self._diff_w1 * op1 - self._diff_w2 * op2)
                elif op == self._opcodes.TIMES:
                    op1 = stack.pop()
                    op2 = stack.pop()
                    stack.append(self._prod_w1 * op1 * self._prod_w2 * op2)
                elif op == self._opcodes.DIVIDE:
                    op1 = stack.pop()
                    op2 = stack.pop()
                    stack.append(self._div_w1 * op1 / self._div_w2 * op2)
                # elif op == self._opcodes.MOD:
                #     op1 = stack.pop()
                #     op2 = stack.pop()
                #     stack.append(self._mod_w1 * op1 % self._mod_w2 * op2)
                #elif op == self._opcodes.DUP:
                #    tmp = stack.pop()
                #    stack.append(self._dup_w * tmp)
                #    stack.append(self._dup_w * tmp)
                #elif op == self._opcodes.SWAP:
                #    tmp1 = stack.pop()
                #    tmp2 = stack.pop()
                #    stack.append(self._swap_w1 * tmp1)
                #    stack.append(self._swap_w2 * tmp2)
                #elif op == self._opcodes.NOP:
                #    pass
                else:
                    stack.append(self._const_w * op)
            if stack:
                return stack.pop()  # in the last value of the stack is collected the value of the fitness
            else:
                return torch.tensor([10 ** 6])

        except IndexError as ie:
            if ie.args[0] == 'pop from empty list':
                return torch.tensor([10 ** 6])
            else:
                raise ie