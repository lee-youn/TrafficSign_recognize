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


def print_images(key_: str, x_: torch.Tensor):
    x = x_[0][:10].cpu().numpy()

    plt.figure(figsize=(20, 3))
    plt.suptitle(key_ + f": {x.shape[1]}*{x.shape[2 ]}", fontsize=40)

    for idx, img in enumerate(x):
        plt.subplot(1, 10, idx + 1)
        plt.imshow(img)
        plt.axis("off")

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
    step_per_mini = len(losses) // 100
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
