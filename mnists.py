import numpy
import itertools
import torch
import torch.utils.data
import collections
from matplotlib import pyplot as plt
from torchvision.datasets import FashionMNIST, MNIST, EMNIST, KMNIST


NPDescription = collections.namedtuple(
    "NPDescription",
    ("query", "target_y", "num_total_points", "num_context_points")
)


class NPBatches(object):
    def __init__(self, xC, yC, xT, yT, nC: int, nT: int, idx: int):
        self.xC = xC
        self.yC = yC
        self.xT = xT
        self.yT = yT
        self.nC = nC
        self.nT = nT
        self.idx = idx

    def dims(self):
        sizes = []
        for d in self.batches():   # d in xC, yC, xT, yT
            sz = d.shape[-1]
            sizes.append(sz)
        return sizes

    def batches(self):
        return [self.xC, self.yC, self.xT, self.yT]

    def desc(self):
        query = ((self.xC, self.yC), self.xT)
        return NPDescription(
            query=query,
            target_y=self.yT,
            num_total_points=self.nT,
            num_context_points=self.nC
        )


class NPMnistReader(object):
    def __init__(self, batch_size, testing=False, shuffle=False,seed=777,
                 mnist_type="mnist", fix_iter=-1, device=torch.device("cpu")):
        self._batch_size = batch_size
        self._testing = testing
        self._shuffle = shuffle
        self._seed = seed
        self._device = device
        mnist_classes = {
            "mnist": MNIST,
            "fashion": FashionMNIST,
            "emnist": EMNIST,
            "kmnist": KMNIST,
        }
        self.mnist = mnist_classes[mnist_type](root="mnist", train=(not testing), download=True)
        self.fix_iter = fix_iter
        self.img_shape = (-1, -1)
        self.n_data = -1
        self.cur = -1
        self.cnt = -1
        self.rnds = numpy.random.RandomState(seed)

    @property
    def batch_size(self):
        return self._batch_size

    def __iter__(self):
        if self._shuffle:
            if self.fix_iter > 0:
                self.rnds = numpy.random.RandomState(self._seed)
            shuffled = self.rnds.permutation(len(self.mnist.data))
            self.mnist.data = self.mnist.data[shuffled]
        self.img_shape = self.mnist.data.shape[1:]
        self.n_data = self.mnist.data.shape[0]
        self.cur = -self._batch_size
        self.cnt = -1
        return self

    def __next__(self) -> NPBatches:
        num_target = self.img_shape.numel()
        num_context = torch.randint(num_target//2, num_target, size=(1,))[0]
        num_context = num_context.numpy()
        if self.fix_iter <= 0:
            self.cur += self._batch_size
            if self.cur >= self.n_data:
                raise StopIteration()
        else:
            self.cur = 0
            self.cnt += 1
            self.rnds = numpy.random.RandomState(self._seed)
            if self.cnt >= self.fix_iter:
                raise StopIteration()
        target_y = self.mnist.data[self.cur:self.cur+self._batch_size]
        target_y = target_y.type(torch.FloatTensor) / 255.
        target_y = target_y.reshape(self._batch_size, -1).unsqueeze(-1)

        def get_interval(n):
            return numpy.linspace(0, n-1, n) / (n/2) - 1    # scale to [-1 1)

        intervals = [get_interval(self.img_shape[0]), get_interval(self.img_shape[1])]
        target_x = itertools.product(intervals[0], intervals[1])
        target_x = torch.Tensor(list(target_x)).unsqueeze(0).repeat(self._batch_size, 1, 1)
        # pixcel index: (target_x[0, -1, :]+1)*14 -> (27, 27)

        indices = self.rnds.choice(num_target, num_context)
        context_x = target_x[:, indices]
        context_y = target_y[:, indices]

        context_x = context_x.to(self._device)
        context_y = context_y.to(self._device)
        target_x = target_x.to(self._device)
        target_y = target_y.to(self._device)
        idx = self.cur // self.batch_size
        return NPBatches(context_x, context_y, target_x, target_y, nC=num_context, nT=num_target, idx=idx)

    def restore_to_grid_indices(self, x):
        d = x.detach().clone()
        d[:, :, 0] = (d[:, :, 0] + 1) * self.img_shape[0]/2
        d[:, :, 1] = (d[:, :, 1] + 1) * self.img_shape[1]/2
        return d.data.cpu().numpy().astype(numpy.int32)

    def convert_to_img(self, x, y):
        B, N, D = y.shape
        if N >= self.img_shape.numel():
            # the case of target_y
            assert N == self.img_shape.numel()
            d = y.reshape(B, *self.img_shape) * 255.
            return d.data.cpu().numpy().astype(numpy.uint8)

        # the case of context_y
        d = torch.zeros(B, *self.img_shape)
        indices = self.restore_to_grid_indices(x)
        for b in range(B):
            for n in range(N):
                i, j = indices[b, n]
                d[b, i, j] = y[b, n, 0]
        d *= 255.
        return d.data.cpu().numpy().astype(numpy.uint8)


def save_yimages(img_c, img_t, img_p=None, img_file="y.png"):
    plt.clf()
    B = img_c.shape[0]
    n_row = 2 if img_p is None else 3
    for b in range(B):
        plt.subplot(B, n_row, b*n_row+1)
        show_image(img_c[b])
        plt.subplot(B, n_row, b*n_row+2)
        show_image(img_t[b])
        if n_row < 3:
            continue
        plt.subplot(B, n_row, b*n_row+3)
        show_image(img_p[b])
    plt.savefig(img_file)
    return


def show_image(img):
    plt.imshow(img, cmap='gray')


if __name__ == "__main__":
    import pathlib
    device = torch.device("cuda:1")

    train_npr = NPMnistReader(batch_size=8, testing=False, device=device)
    test_npr = NPMnistReader(batch_size=8, testing=True, device=device)

    data_list = [(train_npr, "train"), (test_npr, "test")]

    for npr, name in data_list:
        def run_batch(b, itm: NPBatches):
            context_x, context_y, target_x, target_y = itm.batches()
            img_c = npr.convert_to_img(context_x, context_y)
            img_t = npr.convert_to_img(target_y, target_y)
            p = pathlib.Path(f"img_fm/{name}_{b:05d}.png")
            p.parent.mkdir(parents=True, exist_ok=True)
            save_yimages(img_c, img_t, img_file=str(p))
            return

        itm = next(iter(npr))
        with torch.no_grad():
            run_batch(99999, itm)

        n = dict(train=60000, test=10000)
        assert npr.n_data == n[name]

        for b, itm in enumerate(npr):
            run_batch(b, itm)
            if b >= 1:
                break

        for b, _ in enumerate(npr):
            pass
        assert b + 1 == npr.n_data // npr.batch_size

        print(f"data {name}: OK")

