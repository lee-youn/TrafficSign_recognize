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
