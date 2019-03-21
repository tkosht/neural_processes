import collections
import pathlib
import traceback
import torch

import utils
from mnists import NPMnistReader, NPBatches, save_yimages


TrainParameters = collections.namedtuple(
    "TrainParameters",
    ("batch_size", "env_name", "log_interval", "max_epoch", "seed", "fix_iter", "device")
)


class Trainer(object):
    def __init__(self, train_params):
        self.train_params = train_params

        batch_size = train_params.batch_size
        seed = train_params.seed
        fix_iter = train_params.fix_iter
        device = train_params.device
        params = dict(
            shuffle=True, seed=seed,
            mnist_type="mnist", fix_iter=fix_iter,
            device=device,
        )
        self.train_reader = NPMnistReader(batch_size=batch_size, testing=False, **params)
        self.test_reader = NPMnistReader(batch_size=batch_size, testing=True, **params)

        self.model = None
        self.optimizer = None

        self.plotter = utils.VisdomLinePlotter(env_name=self.train_params.env_name)
        self.loss_meter = utils.AverageMeter()

    def get_dims(self):
        itm = next(iter(self.train_reader))
        return itm.dims()   # xC_size, yC_size, xT_size, yT_size

    def run_train(self, model, optimizer):
        for epoch in range(1, self.train_params.max_epoch + 1):
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
            train_itr = iter(self.train_reader)
            train_itm = next(train_itr)
            self.save_images(epoch, "train", train_itm)

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

