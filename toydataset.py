import torch
import torch.utils.data
import collections
from matplotlib import pyplot as plt


NPRegressionDescription = collections.namedtuple(
    "NPRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points")
)


class GPCurvesReader(object):
    def __init__(self, batch_size, max_num_context, x_size=1, y_size=1,
                 l1_scale=.6, sigma_scale=1.0, random_kernel_parameters=True, testing=False,
                 device=torch.device("cpu")):
        self._batch_size = batch_size
        self.max_num_context = max_num_context
        self.x_size = x_size
        self.y_size = y_size
        self._l1_scale = l1_scale
        self._sigma_scale = sigma_scale
        self._random_kernel_parameters = random_kernel_parameters
        self._testing = testing
        self._device = device
        assert self._l1_scale > 0.1

    @staticmethod
    def _gaussian_kernel(xdata, l1, sigma_f, sigma_noise=2e-2):
        num_total_points = xdata.shape[1]

        xdata1 = xdata.unsqueeze(1)     # (B, 1, num_total_points, x_size)
        xdata2 = xdata.unsqueeze(2)     # (B, num_total_points, 1, x_size)
        diff = xdata1 - xdata2          # (B, num_total_points, num_total_points, x_size)

        norm = (diff[:, None, :, :, :] / l1[:, :, None, None, :]) **2
        norm = norm.sum(-1)
        kernel = (sigma_f **2)[:, :, None, None] * torch.exp(-0.5 * norm)
        kernel += (sigma_noise**2) * torch.eye(num_total_points)
        return kernel

    def generate_curves(self):
        num_context = torch.randint(3, self.max_num_context, size=(1,))[0]
        if self._testing:
            num_target = 400
            num_total_points = num_target
            x_values = torch.arange(-2., 2., 1./100).unsqueeze(0).repeat(self._batch_size, 1)
            x_values = x_values.unsqueeze(-1)
        else:
            num_target = torch.randint(0, self.max_num_context - num_context, size=(1,))[0]
            num_total_points = num_context + num_target
            x_values = torch.empty(self._batch_size, num_total_points, self.x_size).uniform_(-2, 2) # may be leaked !?

        if self._random_kernel_parameters:
            l1 = torch.empty(self._batch_size, self.y_size, self.x_size).uniform_(0.1, self._l1_scale)
            sigma_f = torch.empty(self._batch_size, self.y_size).uniform_(0.1, self._sigma_scale)
        else:
            l1 = torch.ones([self._batch_size, self._y_size, self._x_size]) * self._l1_scale
            sigma_f = torch.ones([self._batch_size, self._y_size]) * self._sigma_scale

        kernel = self._gaussian_kernel(x_values, l1, sigma_f)
        cholesky = torch.cholesky(kernel.type(torch.DoubleTensor)).type(torch.FloatTensor)

        eps = torch.normal(torch.zeros([self._batch_size, self.y_size, num_total_points, 1]))
        y_values = torch.matmul(cholesky, eps)     # multivariate normal
        y_values = y_values.squeeze(-1).transpose(1, 2)

        if self._testing:
            target_x = x_values
            target_y = y_values
            idx = torch.randperm(num_target)
            idx_tensor = idx[:num_context]
            context_x = x_values.index_select(1, idx_tensor)
            context_y = y_values.index_select(1, idx_tensor)
        else:
            target_x = x_values[:, :num_target + num_context, :]
            target_y = y_values[:, :num_target + num_context, :]

            # Select the observations
            context_x = x_values[:, :num_context, :]
            context_y = y_values[:, :num_context, :]

        context_x = context_x.to(self._device)
        context_y = context_y.to(self._device)
        target_x = target_x.to(self._device)
        target_y = target_y.to(self._device)
        query = ((context_x, context_y), target_x)

        return NPRegressionDescription(
            query=query,
            target_y=target_y,
            num_total_points=target_x.shape[1],
            num_context_points=num_context)


def saveimg_xy(name, context_x, context_y, target_x, target_y):
    plt.clf()
    plt.plot(target_x[0].cpu().numpy(), target_y[0].cpu().numpy(), 'k:', linewidth=2)
    plt.plot(context_x[0].cpu().numpy(), context_y[0].cpu().numpy(), 'ko', markersize=10)
    plt.yticks([-2, 0, 2], fontsize=16)
    plt.xticks([-2, 0, 2], fontsize=16)
    plt.ylim([-2, 2])
    plt.grid(False)
    ax = plt.gca()
    plt.savefig(f"{name}.png")


def save_functions(file_name, context_x, context_y, target_x, target_y, pred_y, std):
    plt.clf()
    xc = context_x[0, :, 0].cpu().numpy()
    yc = context_y[0, :, 0].cpu().numpy()
    xt = target_x[0, :, 0].cpu().numpy()
    yt = target_y[0, :, 0].cpu().numpy()
    yht = pred_y[0, :, 0].cpu().numpy()
    sgm = std[0, :, 0].cpu().numpy()

    plt.plot(xt, yht, 'b', linewidth=2)
    plt.plot(xt, yt, 'k:', linewidth=2)
    plt.plot(xc, yc, 'ko', markersize=10)
    plt.fill_between( xt, yht - sgm, yht + sgm, alpha=0.2, facecolor='#65c9f7', interpolate=True)

    # Make the plot pretty
    plt.yticks([-2, 0, 2], fontsize=16)
    plt.xticks([-2, 0, 2], fontsize=16)
    plt.ylim([-2, 2])
    plt.grid(False)
    ax = plt.gca()
    # plt.show()
    plt.savefig(f"{file_name}")


if __name__ == "__main__":
    device = torch.device("cuda:1")

    train_gpr = GPCurvesReader(batch_size=8, max_num_context=50, testing=False, device=device)
    test_gpr = GPCurvesReader(batch_size=8, max_num_context=50, testing=True, device=device)

    data_list = [(train_gpr, "train"), (test_gpr, "test")]

    for gpr, name in data_list:
        data = gpr.generate_curves()
        ((context_x, context_y), target_x) = data.query
        target_y = data.target_y
        saveimg_xy(name, context_x, context_y, target_x, target_y)

