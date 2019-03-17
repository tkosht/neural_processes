import numpy
import itertools
import torch
import torch.utils.data
import collections
from matplotlib import pyplot as plt
from torchvision.datasets import FashionMNIST


NPFashionMnistDescription = collections.namedtuple(
    "NPFashionMnistDescription",
    ("query", "target_y", "num_total_points", "num_context_points")
)


class NPFasihonMnistReader(object):
    def __init__(self, batch_size, max_num_context, testing=False, device=torch.device("cpu")):
        self._batch_size = batch_size
        self.max_num_context = max_num_context
        self._testing = testing
        self._device = device
        self.fmnist = FashionMNIST(root="fmnist", train=(not testing), download=True)
        self.img_shape = (-1, -1)
        self.n_data = -1
        self.cur = -1

    @property
    def batch_size(self):
        return self._batch_size

    def __iter__(self):
        self.img_shape = self.fmnist.data.shape[1:]
        self.n_data = self.fmnist.data.shape[0]
        self.cur = -self._batch_size
        return self

    def __next__(self):
        num_target = self.img_shape.numel()
        num_context = torch.randint(num_target//2, num_target, size=(1,))[0]
        num_context = num_context.numpy()
        self.cur += self._batch_size
        if self.cur >= self.n_data:
            raise StopIteration()

        target_y = self.fmnist.data[self.cur:self.cur+self._batch_size]
        target_y = target_y.type(torch.FloatTensor) / 255.
        target_y = target_y.reshape(self._batch_size, -1).unsqueeze(-1)

        def get_interval(n):
            return numpy.linspace(0, n-1, n) / (n/2) - 1    # scale to [-1 1)

        intervals = [get_interval(self.img_shape[0]), get_interval(self.img_shape[1])]
        target_x = itertools.product(intervals[0], intervals[1])
        target_x = torch.Tensor(list(target_x)).unsqueeze(0).repeat(self._batch_size, 1, 1)
        # pixcel index: (target_x[0, -1, :]+1)*14 -> (27, 27)

        indices = numpy.random.choice(num_target, num_context)
        context_x = target_x[:, indices]
        context_y = target_y[:, indices]

        context_x = context_x.to(self._device)
        context_y = context_y.to(self._device)
        target_x = target_x.to(self._device)
        target_y = target_y.to(self._device)

        query = ((context_x, context_y), target_x)

        return NPFashionMnistDescription(
            query=query,
            target_y=target_y,
            num_total_points=target_x.shape[1],
            num_context_points=num_context)

    def restore_to_grid_indices(self, x):
        d = x.detach().clone()
        d[:, :, 0] = (d[:, :, 0] + 1) * self.img_shape[0]/2
        d[:, :, 1] = (d[:, :, 1] + 1) * self.img_shape[1]/2
        return d.data.cpu().numpy().astype(numpy.int32)

    def convert_img(self, x, y):
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


def save_yimages(img_c, img_t, img_file="y.png"):
    plt.clf()
    B = img_c.shape[0]
    for b in range(B):
        plt.subplot(8, 2, b*2+1)
        show_image(img_c[b])
        plt.subplot(8, 2, b*2+2)
        show_image(img_t[b])
    plt.savefig(img_file)
    return


def show_image(img):
    plt.imshow(img, cmap='gray')


if __name__ == "__main__":
    import pathlib
    device = torch.device("cuda:1")

    train_npr = NPFasihonMnistReader(batch_size=8, max_num_context=50, testing=False, device=device)
    test_npr = NPFasihonMnistReader(batch_size=8, max_num_context=50, testing=True, device=device)

    data_list = [(train_npr, "train"), (test_npr, "test")]

    for npr, name in data_list:
        def run_batch(b, itm):
            ((context_x, context_y), target_x) = itm.query
            target_y = itm.target_y
            img_c = npr.convert_img(context_x, context_y)
            img_t = npr.convert_img(target_y, target_y)
            p = pathlib.Path(f"img_fm/{name}_{b:05d}.png")
            p.parent.mkdir(parents=True, exist_ok=True)
            save_yimages(img_c, img_t, str(p))
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
        assert b +1 == npr.n_data // npr.batch_size

        print(f"data {name}: OK")

