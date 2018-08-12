import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from models import MLP

TRAIN_RANGE = [-5, 5]
TEST_RANGE = [-20, 20]
LEARNING_RATE = 1e-2
NUM_ITERS = int(1e4)
NON_LINEARITIES = [
    'hardtanh', 'sigmoid',
    'relu6', 'tanh',
    'tanhshrink', 'hardshrink',
    'leakyrelu', 'softshrink',
    'softsign', 'relu',
    'prelu', 'softplus',
    'elu', 'selu',
]


def train(model, optimizer, data, num_iters):
    for i in range(num_iters):
        out = model(data)
        loss = F.mse_loss(out, data)
        mea = torch.mean(torch.abs(data - out))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print("\t{}/{}: loss: {:.3f} - mea: {:.3f}".format(
                i+1, num_iters, loss.item(), mea.item())
            )


def test(model, data):
    with torch.no_grad():
        out = model(data)
        return torch.abs(data - out)


def main():
    save_dir = './imgs/'

    TRAIN_RANGE[-1] += 1
    TEST_RANGE[-1] += 1

    # datasets
    train_data = torch.arange(*TRAIN_RANGE).unsqueeze_(1).float()
    test_data = torch.arange(*TEST_RANGE).unsqueeze_(1).float()

    # train
    all_mses = []
    for non_lin in NON_LINEARITIES:
        print("Working with {}...".format(non_lin))
        mses = []
        for i in range(100):
            net = MLP(4, 1, 8, 1, non_lin)
            optim = torch.optim.RMSprop(net.parameters(), lr=LEARNING_RATE)
            train(net, optim, train_data, NUM_ITERS)
            mses.append(test(net, test_data))
        all_mses.append(torch.cat(mses, dim=1).mean(dim=1))
    all_mses = [x.numpy().flatten() for x in all_mses]

    # plot
    fig, ax = plt.subplots(figsize=(8, 7))
    x_axis = np.arange(-20, 21)
    for i, non_lin in enumerate(NON_LINEARITIES):
        ax.plot(x_axis, all_mses[i], label=non_lin)
    plt.grid()
    plt.legend(loc='best')
    plt.ylabel('Mean Absolute Error')
    plt.savefig(save_dir + 'extrapolation.png', format='png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
