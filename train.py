import argparse
import torch
import torch.utils.data
from torch import nn, optim

import utils
from npmodel import NPModel
from toydataset import GPCurvesReader, plot_functions


def get_args():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--gpu', type=int, default=1, metavar='N',
                        help='gpu number (default: 1), if no-cuda, ignore this option.')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


def train(model, optimizer, epoch, npcfg):
    trainset = npcfg.trainset
    # trainset, _ = make_dataset(npcfg.train_gpr)
    testset, _ = make_dataset(npcfg.test_gpr)

    model.train()
    # train_loss = 0
    xC, yC, xT, yT = trainset
    optimizer.zero_grad()
    yhatT, sgm, loss = model(xC, yC, xT, yT)
    loss.backward()
    def losser():
        return loss
    optimizer.step(losser)
    loss_meter.update(loss.item())
    predicted_list.append(yhatT)

    if epoch % 1000 == 0:
        model.check_kl_collapse(xC, yC, xT, yT)

    if epoch % 100 == 0:
        try:

            # if visdom server running, plot loss values
            plotter.plot("epoch", "loss", "train", "Epoch - Loss", [epoch], [loss_meter.avg], reset=False)
        except Exception as e:
            print(e)
        finally:
            loss_meter.reset()

    if epoch % npcfg.log_interval == 0:
        print(f"Train Epoch {epoch}/{npcfg.max_epoch}: {loss.item():.6f}")
        # file_name = f"img/train-{epoch:05d}.png"
        # plot_functions(file_name, *trainset, yhatT, sgm)

        file_name = f"img/test-{epoch:05d}.png"
        import pathlib
        p = pathlib.Path(file_name)
        p.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            yhatT, sgm = model.predict(*testset[:3])
        plot_functions(file_name, *testset, yhatT, sgm)


def make_dataset(gpr):
    data = gpr.generate_curves()
    ((context_x, context_y), target_x) = data.query
    target_y = data.target_y
    dataset = [context_x, context_y, target_x, target_y]
    for idx, d in enumerate(dataset):
        dataset[idx] = d.to(device)
    xC_size = context_x.shape[-1]
    yC_size = context_y.shape[-1]
    xT_size = target_x.shape[-1]
    yT_size = target_y.shape[-1]
    # xC_size = numpy.array(context_x.shape[-2:]).prod()
    # yC_size = numpy.array(context_y.shape[-2:]).prod()
    # xT_size = numpy.array(target_x.shape[-2:]).prod()
    # yT_size = numpy.array(target_y.shape[-2:]).prod()
    sizes = [xC_size, yC_size, xT_size, yT_size]
    return dataset, sizes


if __name__ == "__main__":
    args = get_args()
    device = torch.device(f"cuda:{args.gpu}" if args.cuda else "cpu")

    batch_size = 16
    train_gpr = GPCurvesReader(batch_size=batch_size, max_num_context=50, testing=False)
    test_gpr = GPCurvesReader(batch_size=1, max_num_context=50, testing=True)
    trainset, train_sizes = make_dataset(train_gpr)
    xC_size, yC_size, xT_size, yT_size = train_sizes

    import collections
    NPTrainConfig = collections.namedtuple(
        "NPTrainConfig", ("trainset", "train_gpr", "test_gpr", "log_interval", "max_epoch")
    )
    npcfg = NPTrainConfig(
        trainset=trainset, train_gpr=train_gpr, test_gpr=test_gpr,
        log_interval=args.log_interval, max_epoch=args.epochs
    )

    hidden_size = 128
    model_params = {
        "xC_size": xC_size,
        "yC_size": yC_size,
        "xT_size": xT_size,
        "yT_size": yT_size,
        "z_size": hidden_size,
        "embed_layers": [hidden_size]*4,
        "encoder_layers": [hidden_size]*4,
        "expand_layers": [hidden_size]*2 + [yT_size],
    }
    model = NPModel(**model_params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    global plotter
    # plotter = utils.VisdomLinePlotter(env_name='np')
    plotter = utils.VisdomLinePlotter(env_name='main')

    global loss_meter
    loss_meter = utils.AverageMeter()

    global predicted_list
    predicted_list = []

    for epoch in range(1, args.epochs + 1):
        train(model, optimizer, epoch, npcfg)

