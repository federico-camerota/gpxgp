from abc import ABC, abstractmethod

import torch
from torch import nn


class Operator(ABC):
    @abstractmethod
    def n_args(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def parameters(self):
        pass


class ConstantOP(Operator):
    def __init__(self, val):
        super().__init__()
        self._const = val
        self.__w = nn.parameter.Parameter(torch.tensor([1.]))

    def __call__(self, *args, **kwargs):
        return self.__w * self._const

    def n_args(self):
        return 0

    def parameters(self):
        return [self.__w]


class SumOP(Operator):
    def __init__(self):
        super().__init__()
        self.__w1 = nn.parameter.Parameter(torch.tensor([1.]))
        self.__w2 = nn.parameter.Parameter(torch.tensor([1.]))

    def __call__(self, *args, **kwargs):
        return self.__w1 * args[0] + self.__w2 * args[1]

    def n_args(self):
        return 2

    def parameters(self):
        return [self.__w1, self.__w2]


class SubtractionOP(Operator):
    def __init__(self):
        super().__init__()
        self.__w1 = nn.parameter.Parameter(torch.tensor([1.]))
        self.__w2 = nn.parameter.Parameter(torch.tensor([1.]))

    def __call__(self, *args, **kwargs):
        return self.__w1 * args[0] - self.__w2 * args[1]

    def n_args(self):
        return 2

    def parameters(self):
        return [self.__w1, self.__w2]


class ProductOP(Operator):
    def __init__(self):
        super().__init__()
        self.__w1 = nn.parameter.Parameter(torch.tensor([1.]))
        self.__w2 = nn.parameter.Parameter(torch.tensor([1.]))

    def __call__(self, *args, **kwargs):
        return self.__w1 * args[0] * self.__w2 * args[1]

    def n_args(self):
        return 2

    def parameters(self):
        return [self.__w1, self.__w2]


class DivisionOP(Operator):
    def __init__(self):
        super().__init__()
        self.__w1 = nn.parameter.Parameter(torch.tensor([1.]))
        self.__w2 = nn.parameter.Parameter(torch.tensor([1.]))

    def __call__(self, *args, **kwargs):
        return self.__w1 * args[0] / self.__w2 * args[1]

    def n_args(self):
        return 2

    def parameters(self):
        return [self.__w1, self.__w2]


class DuplicationOP(Operator):
    def __init__(self):
        super().__init__()
        self.__w1 = nn.parameter.Parameter(torch.tensor([1.]))
        self.__w2 = nn.parameter.Parameter(torch.tensor([1.]))

    def __call__(self, *args, **kwargs):
        return self.__w1 * args[0], self.__w2 * args[0]

    def n_args(self):
        return 1

    def parameters(self):
        return [self.__w1, self.__w2]


class SwapOP(Operator):
    def __init__(self):
        super().__init__()
        self.__w1 = nn.parameter.Parameter(torch.tensor([1.]))
        self.__w2 = nn.parameter.Parameter(torch.tensor([1.]))

    def __call__(self, *args, **kwargs):
        return self.__w1 * args[1], self.__w2 * args[0]

    def n_args(self):
        return 2

    def parameters(self):
        return [self.__w1, self.__w2]


class NopOP(Operator):
    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        pass

    def n_args(self):
        return 0

    def parameters(self):
        return []
