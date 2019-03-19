import argparse
import pathlib
import collections
import traceback

import torch
from torch import optim

import utils
from npmodel import NPModel
from mnists import NPMnistReader, NPBatches, save_yimages


def get_args():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--gpu', type=int, default=1, metavar='N',
                        help='gpu number (default: 1), if no-cuda, ignore this option.')
    parser.add_argument('--seed', type=int, default=777, metavar='S',
                        help='random seed (default: 777)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dataset', type=str, default="mnist", choices=["mnist", "fashion"], metavar='S',
                        help='dataset name like "mnist", "fashion-mnist", default: "mnist"')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


class Trainer(object):
    def __init__(self, train_params):
        self.train_params = train_params

        batch_size, device = train_params.batch_size, train_params.device
        seed = train_params.seed
        params = dict(
            shuffle=True, seed=seed,
            mnist_type="mnist", fix_iter=10000,
            device=device,
        )
        self.train_reader = NPMnistReader(batch_size=batch_size, testing=False, **params)
        self.test_reader = NPMnistReader(batch_size=batch_size, testing=True, **params)

        self.model = None
        self.optimizer = None

        self.plotter = utils.VisdomLinePlotter(env_name=self.train_params.env_name)
        self.loss_meter = utils.AverageMeter()

    def get_dims(self):
        itm = next(iter(trainer.train_reader))
        return itm.dims()   # xC_size, yC_size, xT_size, yT_size

    def run_train(self, model, optimizer):
        for epoch in range(1, args.epochs + 1):
            self.train_epoch(epoch, model, optimizer)

    def train_epoch(self, epoch, model, optimizer):
        # hold as local variable
        train_params = self.train_params

        # keep on self
        self.model = model
        self.optimizer = optimizer

        # set to train mode
        model.train()

        test_itr = iter(self.test_reader)
        for b, train_itm in enumerate(self.train_reader):
            train_batches = train_itm.batches()   # xC, yC, xT, yT
            optimizer.zero_grad()
            yhatT, sgm, loss = model(*train_batches)
            loss.backward()

            def loss_closure():
                return loss
            optimizer.step(loss_closure)
            self.loss_meter.update(loss.item())

            if epoch % train_params.log_interval == 0 and b == 0:
                self.save_images(epoch, "train", train_itm)

        # by epoch
        try:
            print(f"Train Epoch {epoch:05d}/{train_params.max_epoch:05d}: {self.loss_meter.avg:.6f}")

            # if visdom server running, plot loss values
            self.plotter.plot("epoch", "loss", "train", "Epoch - Loss", [epoch], [self.loss_meter.avg], reset=False)
        except Exception as e:
            print(traceback.format_exc(chain=e))
        finally:
            self.loss_meter.reset()     # by every epoch

        if epoch % train_params.log_interval == 0:
            try:
                test_itm = next(test_itr)
            except StopIteration:
                test_itr = iter(self.test_reader)
                test_itm = next(test_itr)
            self.save_images(epoch, "test", test_itm)

    def save_images(self, epoch: int, name: str, batch_item: NPBatches):
        assert name in ["train", "test"]
        batches = batch_item.batches()
        file_name = f"img/{name}-{epoch:05d}_{batch_item.idx:05d}.png"
        p = pathlib.Path(file_name)
        p.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            yhatT, sgm = self.model.predict(*batches[:3])
            xC, yC, xT, yT = batches
            img_yC = self.train_reader.convert_to_img(xC, yC)
            img_yT = self.train_reader.convert_to_img(xT, yT)
            img_yhat = self.train_reader.convert_to_img(xT, yhatT)
            save_yimages(img_yC, img_yT, img_yhat, file_name)


if __name__ == "__main__":
    args = get_args()
    device = torch.device(f"cuda:{args.gpu}" if args.cuda else "cpu")
    if args.seed >= 0:
        torch.manual_seed(args.seed)

    batch_size = 30
    TrainParameters = collections.namedtuple(
        "TrainParameters", ("batch_size", "env_name", "log_interval", "max_epoch", "seed", "device")
    )
    train_params = TrainParameters(
        batch_size=batch_size, env_name="main",
        log_interval=args.log_interval, max_epoch=args.epochs,
        seed=args.seed,
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
    )

    model = NPModel(**model_params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    trainer.run_train(model, optimizer)
