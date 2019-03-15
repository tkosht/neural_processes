import numpy
from visdom import Visdom


class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main', port=8097, server="localhost"):
        self.viz = Visdom(port=port, server=server)
        self.env = env_name
        self.plotted_windows = {}

    def plot(self, x_label, y_label, legend_name, title_name, x, y, reset=False):
        # X=x if hasattr(x, "__iter__") else numpy.array([x]),
        # Y=y if hasattr(y, "__iter__") else numpy.array([y]),
        params = dict(
            X=x,
            Y=y,
            env=self.env,
            name=legend_name,
        )
        extra = dict(opts=dict(legend=[legend_name],
                               title=title_name,
                               xlabel=x_label,
                               ylabel=y_label,
                               ))
        if y_label in self.plotted_windows:  # just at the first time
            _extra = dict(
                win=self.plotted_windows[y_label],
            )
            if not reset:
                _extra = dict(
                    win=self.plotted_windows[y_label],
                    update='append',
                )
            extra.update(_extra)
        params.update(extra)
        self.plotted_windows[y_label] = self.viz.line(**params)

    def scatter(self, x, y, y_label, legend_name, color=(0, 0, 0)):
        color = numpy.array(color).reshape(-1, 3)
        win = self.plotted_windows[y_label]
        self.viz.scatter(X=x, Y=y,
                         opts=dict(markersize=10, markercolor=color,),
                         name=legend_name,
                         update='append',
                         win=win)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = -1
        self.avg = -1
        self.sum = -1
        self.count = -1
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

