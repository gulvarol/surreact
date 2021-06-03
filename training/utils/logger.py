# A simple torch style logger
# (C) Wei YANG 2017

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

__all__ = ["Logger", "LoggerMonitor", "savefig"]


def savefig(fname, dpi=None):
    dpi = 150 if dpi == None else dpi
    plt.savefig(fname, dpi=dpi)


def plot_overlap(logger, names=None):
    names = logger.names if names == None else names
    numbers = logger.numbers
    for _, name in enumerate(names):
        x = np.arange(len(numbers[name]))
        plt.plot(x, np.asarray(numbers[name]))
    return [logger.title + "(" + name + ")" for name in names]


class Logger(object):
    """Save training process to log file with simple plot function."""

    def __init__(self, fpath, title=None, resume=False, save_json=True):
        self.file = None
        self.resume = resume
        self.title = "" if title == None else title
        self.save_json = save_json
        if self.save_json:
            self.json_path = os.path.splitext(fpath)[0] + ".json"
        if fpath is not None:
            if resume:
                self.file = open(fpath, "r")
                name = self.file.readline()
                self.names = name.rstrip().split("\t")
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split("\t")
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, "a")
                if self.save_json:
                    json_data = open(self.json_path).read()
                    self.figures = json.loads(json_data)
                    # self.json_file.close()
            else:
                self.file = open(fpath, "w")
                if self.save_json:
                    self.figures = {}

    def set_names(self, names):
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write("\t")
            self.numbers[name] = []
            if self.save_json and name not in self.figures.keys():
                if len(name.split("_")) > 1:
                    fig_id = name.split("_")[1]  # take 'loss' if it is 'val_loss'
                else:
                    fig_id = name
                self.figures[fig_id] = {}
                self.figures[fig_id]["data"] = []
                self.figures[fig_id]["layout"] = {"title": fig_id}
        self.file.write("\n")
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), "Numbers do not match names"
        for index, num in enumerate(numbers):
            self.file.write("{0:.3f}".format(num))
            self.file.write("\t")
            self.numbers[self.names[index]].append(num)
        self.file.write("\n")
        self.file.flush()
        if self.save_json:
            for index, num in enumerate(numbers):
                if len(self.names[index].split("_")) > 1:
                    plot_id = self.names[index].split("_")[
                        0
                    ]  # take 'val' if it is 'val_loss'
                    fig_id = self.names[index].split("_")[
                        1
                    ]  # take 'loss' if it is 'val_loss'
                else:
                    plot_id = self.names[index]
                    fig_id = self.names[index]
                fig_data = self.figures[fig_id]["data"]
                plot = None
                for k, v in enumerate(fig_data):
                    if v["name"] == plot_id:
                        plot = v

                if plot is None:
                    plot = {"name": plot_id, "x": [], "y": []}
                    fig_data.append(plot)

                # Epoch
                plot["x"].append(numbers[0])
                # Value
                plot["y"].append(num)

            self.json_file = open(self.json_path, "w")
            self.json_file.write(json.dumps(self.figures))
            self.json_file.close()

    def plot(self, names=None):
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + "(" + name + ")" for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()


class LoggerMonitor(object):
    """Load and visualize multiple logs."""

    def __init__(self, paths):
        """paths is a distionary with {name:filepath} pair"""
        self.loggers = []
        for title, path in list(paths.items()):
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)

    def plot(self, names=None):
        plt.figure()
        plt.subplot(121)
        legend_text = []
        for logger in self.loggers:
            legend_text += plot_overlap(logger, names)
        plt.legend(legend_text, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.grid(True)


if __name__ == "__main__":
    # # Example
    # logger = Logger('test.txt')
    # logger.set_names(['Train loss', 'Valid loss','Test loss'])

    # length = 100
    # t = np.arange(length)
    # train_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
    # valid_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
    # test_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1

    # for i in range(0, length):
    #     logger.append([train_loss[i], valid_loss[i], test_loss[i]])
    # logger.plot()

    # Example: logger monitor
    paths = {
        "resadvnet20": "/home/wyang/code/pytorch-classification/checkpoint/cifar10/resadvnet20/log.txt",
        "resadvnet32": "/home/wyang/code/pytorch-classification/checkpoint/cifar10/resadvnet32/log.txt",
        "resadvnet44": "/home/wyang/code/pytorch-classification/checkpoint/cifar10/resadvnet44/log.txt",
    }

    field = ["Valid Acc."]

    monitor = LoggerMonitor(paths)
    monitor.plot(names=field)
    savefig("test.eps")
