import math
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import MLP, NAC, NALU

NORMALIZE = True
NUM_LAYERS = 2
HIDDEN_DIM = 2
LEARNING_RATE = 1e-3
NUM_ITERS = int(1e5)
RANGE = [5, 10]
ARITHMETIC_FUNCTIONS = {
    'add': lambda x, y: x + y,
    'sub': lambda x, y: x - y,
    'mul': lambda x, y: x * y,
    'div': lambda x, y: x / y,
    'squared': lambda x, y: torch.pow(x, 2),
    'root': lambda x, y: torch.sqrt(x),
}


def generate_data(num_train, num_test, dim, num_sum, fn, support):
    data = torch.FloatTensor(dim).uniform_(*support).unsqueeze_(1)
    X, y = [], []
    for i in range(num_train + num_test):
        idx_a = random.sample(range(dim), num_sum)
        idx_b = random.sample([x for x in range(dim) if x not in idx_a], num_sum)
        a, b = data[idx_a].sum(), data[idx_b].sum()
        X.append([a, b])
        y.append(fn(a, b))
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y).unsqueeze_(1)
    indices = list(range(num_train + num_test))
    np.random.shuffle(indices)
    X_train, y_train = X[indices[num_test:]], y[indices[num_test:]]
    X_test, y_test = X[indices[:num_test]], y[indices[:num_test]]
    return X_train, y_train, X_test, y_test


def train(model, optimizer, data, target, num_iters):
    for i in range(num_iters):
        out = model(data)
        loss = F.mse_loss(out, target)
        mea = torch.mean(torch.abs(target - out))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print("\t{}/{}: loss: {:.7f} - mea: {:.7f}".format(
                i+1, num_iters, loss.item(), mea.item())
            )


def test(model, data, target):
    with torch.no_grad():
        out = model(data)
        return torch.abs(target - out)



def main():
    save_dir = './results/'

    models = [
        MLP(
            num_layers=NUM_LAYERS,
            in_dim=2,
            hidden_dim=HIDDEN_DIM,
            out_dim=1,
            activation='relu6',
        ),
        MLP(
            num_layers=NUM_LAYERS,
            in_dim=2,
            hidden_dim=HIDDEN_DIM,
            out_dim=1,
            activation='none',
        ),
        NAC(
            num_layers=NUM_LAYERS,
            in_dim=2,
            hidden_dim=HIDDEN_DIM,
            out_dim=1,
        ),
        NALU(
            num_layers=NUM_LAYERS,
            in_dim=2,
            hidden_dim=HIDDEN_DIM,
            out_dim=1
        ),
    ]

    results = {}
    for fn_str, fn in ARITHMETIC_FUNCTIONS.items():
        results[fn_str] = []

        # dataset
        X_train, y_train, X_test, y_test = generate_data(
            num_train=500, num_test=50,
            dim=100, num_sum=5, fn=fn,
            support=RANGE,
        )

        # random model
        random_mse = []
        for i in range(100):
            net = MLP(
                num_layers=NUM_LAYERS, in_dim=2,
                hidden_dim=HIDDEN_DIM, out_dim=1,
                activation='relu6',
            )
            mse = test(net, X_test, y_test)
            random_mse.append(mse.mean().item())
        results[fn_str].append(np.mean(random_mse))

        # others
        for net in models:
            optim = torch.optim.RMSprop(net.parameters(), lr=LEARNING_RATE)
            train(net, optim, X_train, y_train, NUM_ITERS)
            mse = test(net, X_test, y_test).mean().item()
            results[fn_str].append(mse)

    with open(save_dir + "interpolation.txt", "w") as f:
        f.write("Relu6\tNone\tNAC\tNALU\n")
        for k, v in results.items():
            rand = results[k][0]
            mses = [100.0*x/rand for x in results[k][1:]]
            if NORMALIZE:
                f.write("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(*mses))
            else:
                f.write("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(*results[k][1:]))


if __name__ == '__main__':
    main()
