import os
import matplotlib.pyplot as plt
import torch
import numpy as np

from CNN import CNN
from common.trainer import Trainer

from data.load_data import load_data

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Available devices ", torch.cuda.device_count())
    print("Current cuda device ", torch.cuda.current_device())
    print(torch.cuda.get_device_name(device))

    N = 6400
    EPOCHS = 2
    BATCH_SIZE = 100
    LR = 0.001

    # <data shape>
    # train     (N,     3,48,48)
    # validation(N,     3,48,48)
    # test      (7766,  3,48,48)
    # N=none 일 경우 각 90601, 31063, 7766개

    x_train, y_train, x_validation, y_validation, x_test, y_test = load_data(
        N, _print=True, device=device
    )

    # number of nodes in the previous layer
    # nodes_num = BATCH_SIZE*3*48*48
    # he = np.sqrt(2.0 / nodes_num)

    network = CNN(
        input_dim=(3, 48, 48),
        conv_param={"filter_num": 100, "filter_size": 5, "pad": 0, "stride": 1},
        hidden_size=100,
        output_size=43,
        weight_init_std=0.01,
        # 0.01, He
        device=device,
    )

    trainer = Trainer(
        # CNN
        network,

        # dataset
        x_train,
        y_train,
        x_test,
        y_test,

        # HyperParameters
        epochs=EPOCHS,
        mini_batch_size=BATCH_SIZE,
        optimizer="Adam",
        optimizer_param={"lr": LR},
        evaluate_sample_num_per_epoch=1000,
        # number of samples used in evaluating accuracy on every epoch.
        device=device,
        # CUDA device (GPU)
    )
    trainer.train()

    markers = {"train": "o", "test": "s"}
    x = torch.arange(EPOCHS)
    plt.plot(x, trainer.train_acc_list, marker="o", label="train", markevery=2)
    plt.plot(x, trainer.test_acc_list, marker="s", label="test", markevery=2)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc="lower right")
    plt.show()

    plt.matshow(trainer.confusionMatrix)
    for (x, y), value in np.ndenumerate(trainer.confusionMatrix):
        plt.text(x, y, f"{value}", va="center", ha="center")
    plt.show()
