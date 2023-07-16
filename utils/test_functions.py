import torch


def sum_of_squares(x):
    return torch.sum(x ** 2, dim=-1)


def all_sum(x):
    return torch.sum(x, dim=-1)


def ugly(x):
    return x[:, 0] ** 2 + torch.sum(x[:, 1:5], dim=-1) ** 2 * torch.sum(x[:, 6:12], dim=-1) ** 3 - torch.prod(x[:, 13:],
                                                                                                              dim=-1)


TEST_FUNS = {'sum_of_squares': sum_of_squares,
             'all_sum': all_sum,
             'ugly': ugly}
