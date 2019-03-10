import numpy
from visdom import Visdom


class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main', port=8097, server="localhost"):
        self.viz = Visdom(port=port, server=server)
        self.env = env_name
        self.plotted_windows = {}

    def plot(self, x_label, y_label, data_name, title_name, x, y):
        params = dict(
            X=numpy.array([x]),
            Y=numpy.array([y]),
            env=self.env,
            name=data_name,
        )
        if y_label in self.plotted_windows:  # just at the first time
            extra = dict(
                win=self.plotted_windows[y_label],
                update='append',
            )
        else:
            extra = dict(opts=dict(legend=[data_name],
                                   title=title_name,
                                   xlabel=x_label,
                                   ylabel=y_label,
                                   ))
        params.update(extra)
        self.plotted_windows[y_label] = self.viz.line(**params)

