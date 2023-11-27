from collections import OrderedDict

import numpy as np
import torch
from matplotlib import pyplot as plt


def accuracy_graph(trainer_, epoch_):
    markers = {"train": "o", "test": "s"}
    x = torch.arange(epoch_)
    plt.plot(
        x, trainer_.train_acc_list, marker=markers["train"], label="train", markevery=2
    )
    plt.plot(
        x, trainer_.test_acc_list, marker=markers["test"], label="test", markevery=2
    )
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc="lower right")

    plt.show()


def confusion_matrix(trainer_):
    plt.figure(figsize=(20, 20))
    plt.matshow(trainer_.confusion_matrix, fignum=1, cmap=plt.colormaps["Blues"])
    for (x, y), value in np.ndenumerate(trainer_.confusion_matrix):
        plt.text(x, y, f"{int(value) if value != 0 else ' '}", va="center", ha="center")
    plt.show()


def loss_graph(trainer_, epoch_: int):
    losses = trainer_.train_loss_list

    # step_per_epoch
    step_per_epoch = len(losses) // epoch_
    losses_per_epoch = []
    for i in range(epoch_):
        losses_per_epoch.append(
            np.mean(losses[step_per_epoch * i : step_per_epoch * (i + 1)])
        )
    epochs = range(epoch_)
    # step_per_mini
    step_per_mini = max(len(losses) // 100, 1)
    losses_per_mini = losses[::step_per_mini]
    mini_batches = range(len(losses_per_mini))

    plt.figure(figsize=(6, 6))
    plt.title("Train Loss", fontsize=20)
    plt.xlabel("mini_batch")
    plt.ylabel("loss")

    # losses_per_epoch
    plt.plot(mini_batches, losses_per_mini)
    ax2 = plt.twiny()
    ax2.plot(epochs, losses_per_epoch, "ro--", markevery=epoch_ // 5)

    plt.show()


def print_images(x_: OrderedDict):
    x_keys = list(x_.keys())
    x_values = list(x_.values())
    layer_count = len(x_)

    plt.figure(figsize=(20, layer_count * 3))

    fig = plt.figure(constrained_layout=True)
    sub_figures = fig.subfigures(nrows=layer_count, ncols=1)

    for row, sub_figure in enumerate(sub_figures):
        sub_figure.suptitle(f"{x_keys[row]}: {len(x_values[row])}*{len(x_values[row])}")

        # subplots for sub_figure
        axes = sub_figure.subplots(nrows=1, ncols=10)
        for col, axis in enumerate(axes):
            axis.imshow(x_values[row][col])
            axis.axis("off")

    plt.show()
