import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import numpy as np

from common.etc import print_gpu_info
from common.plot import confusion_matrix, Accuracy_graph
from common.trainer import Trainer

from data.load_data import load_data
from models.SimpleConvNet import SimpleConvNet

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_gpu_info(device)

    # Hyper Parameters
    PARAMS = {
        "N": 64000,
        "EPOCHS": 2,
        "BATCH_SIZE": 64,
        "LR": 0.001,

        "CONV_FILTER_NUM": 100,
        "CONV_FILTER_SIZE": 5,
        "CONV_PADDING": 0,
        "CONV_STRIDE": 1,

        "HIDDEN_SIZE": 100
    }

    x_train, y_train, x_validation, y_validation, x_test, y_test = \
        load_data(_print=True, device=device)

    network = SimpleConvNet(
        conv_param={"filter_num": PARAMS["CONV_FILTER_NUM"],
                    "filter_size": PARAMS["CONV_FILTER_SIZE"],
                    "pad": PARAMS["CONV_PADDING"],
                    "stride": PARAMS["CONV_STRIDE"]},
        hidden_size=PARAMS["HIDDEN_SIZE"],
        weight_init_std=0.01,
        device=device,
    )

    trainer = Trainer(
        network=network,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,

        epochs=PARAMS["EPOCHS"],
        mini_batch_size=PARAMS["BATCH_SIZE"],
        optimizer="Adam",
        optimizer_param={"lr": PARAMS["LR"]},
        evaluate_sample_num_per_epoch=1000,
        device=device,
    )
    trainer.train()

    # Accuracy Graph
    Accuracy_graph(trainer, PARAMS["EPOCHS"])

    confusion_matrix(trainer)
