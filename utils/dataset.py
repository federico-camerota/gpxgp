import torch


def make_ds(path):
    with open(path) as f:
        lines = f.readlines()

    variables = int(lines[0].strip())
    lines.pop(0)
    instances = int(lines[0].strip())
    lines.pop(0)

    X = torch.zeros(variables, instances)
    y = torch.zeros(instances)

    for index_line in range(len(lines)):
        list_instance = lines[index_line].split()
        for index_el in range(len(list_instance) - 1):
            X[index_el, index_line] = float(list_instance[index_el])
        y[index_line] = float(list_instance[-1])

    return X.T, y


def make_train_test_ds(name_ds, number_test):
    path_train = "datasets/" + name_ds + "/train" + str(number_test)
    path_test = "datasets/" + name_ds + "/test" + str(number_test)

    X_train, y_train = make_ds(path_train)
    X_test, y_test = make_ds(path_test)

    return X_train, y_train, X_test, y_test