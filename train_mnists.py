import argparse

import torch
from torch import optim

import utils
from npmodel import NPModel
from trainer import Trainer, TrainParameters


def get_args():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=30, metavar='N',
                        help='input batch size for training (default: 30)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--gpu', type=int, default=1, metavar='N',
                        help='gpu number (default: 1), if no-cuda, ignore this option')
    parser.add_argument('--seed', type=int, default=777, metavar='S',
                        help='random seed (default: 777)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status (default: 10)')
    parser.add_argument('--dataset', type=str, default="mnist", choices=["mnist", "fashion", "kmnist"], metavar='S',
                        help='dataset name like "mnist", "fashion-mnist", default: "mnist"')
    parser.add_argument('--fix-iter', type=int, default=-1, metavar='N',
                        help='the number of training a fixed batch, if negative, using whole data (default: -1)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='F',
                        help='learning rate (default: 1e-4)')
    args = parser.parse_args()
    args.cuda = (args.gpu >= 0) and torch.cuda.is_available()
    return args


if __name__ == "__main__":
    args = get_args()
    utils.print_params(args, locals())
    device = torch.device(f"cuda:{args.gpu}" if args.cuda else "cpu")
    if args.seed >= 0:
        torch.manual_seed(args.seed)

    train_params = TrainParameters(
        batch_size=args.batch_size, env_name="main",
        log_interval=args.log_interval, max_epoch=args.epochs,
        seed=args.seed, fix_iter=args.fix_iter,
        device=device
    )
    trainer = Trainer(train_params)
    xC_dim, yC_dim, xT_dim, yT_dim = trainer.get_dims()

    hidden_size = 128
    model_params = dict(
        xC_size=xC_dim,
        yC_size=yC_dim,
        xT_size=xT_dim,
        yT_size=yT_dim,
        z_size=hidden_size,
        embed_layers=[hidden_size]*3,
        latent_encoder_layers=[hidden_size]*1,
        deterministic_layers=[hidden_size]*4,
        decoder_layers=[hidden_size]*2 + [yT_dim],
        use_deterministic_path=False,
        dec_f=torch.sigmoid,
    )

    model = NPModel(**model_params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    trainer.run_train(model, optimizer)
