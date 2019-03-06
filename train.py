import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torchvision.utils import save_image

import toydataset as dataset
from npmodel import NPModel


def get_args():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


def train(model, epoch, log_interval):
    model.train()
    # train_loss = 0
    for bdx in enumerate(range()):
        data = data.to(device)
        optimizer.zero_grad()
        y_hatT, loss = model(data)
        loss.backward()
        # train_loss += loss.item()
        optimizer.step()
        if bdx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, bdx * len(data), len(train_loader.dataset),
                       100. * bdx / len(train_loader),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


if __name__ == "__main__":
    from toydataset import GPCurvesReader
    args = get_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    params = {
        "xC_size": 111,
        "yC_size": 111,
        "xT_size": 333,
        "yT_size": 333,
        "z_size": 66,
        "embed_layers": [32, 64, 28],
        "expand_layers": [32, 64],
    }
    model = NPModel(**params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_gpr = GPCurvesReader(batch_size=8, max_num_context=50, testing=False)
    test_gpr = GPCurvesReader(batch_size=8, max_num_context=50, testing=True)


    for epoch in range(1, args.epochs + 1):
        train(model, epoch, args.log_interval)
        # test(epoch)
        # with torch.no_grad():
        #     sample = torch.randn(64, 20).to(device)
        #     sample = model.decode(sample).cpu()
        #     save_image(sample.view(64, 1, 28, 28),
        #                'results/sample_' + str(epoch) + '.png')
