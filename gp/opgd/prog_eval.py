import torch


def opgd_prg_eval(program, x):
    """
    Function that takes a program and variables as input and returns the result of performing the described program
    :param x: variables for the program
    :param program: program considered
    :return: program evaluation at given inputs
    """
    try:
        stack = x.copy()

        if type(stack) is not list:
            stack = stack.tolist()
        while program:
            op = program[0]
            program = program[1:]

            op_args = [stack.pop() for _ in range(op.n_args())]
            ret = op(*op_args)

            if isinstance(ret, tuple):
                stack.extend(ret)
            elif ret is not None:
                stack.append(ret)

        if stack:
            return stack.pop()  # in the last value of the stack is collected the value of the fitness
        else:
            return torch.tensor([10 ** 6])

    except IndexError as ie:

        if ie.args[0] == 'pop from empty list':
            return torch.tensor([10 ** 6])
        else:
            raise ie
