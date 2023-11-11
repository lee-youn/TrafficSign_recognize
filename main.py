import os
import numpy as np
import matplotlib.pyplot as plt

from CNN import CNN
from common.trainer import Trainer

from data.load_data import load_data

if __name__ == "__main__":
    N = 6400
    EPOCHS = 10
    BATCH_SIZE = 64
    LR = 0.001

    x_train, y_train, x_validation, y_validation, x_test, y_test = load_data(N)

    network = CNN(
        input_dim=(3, 48, 48),
        conv_param={"filter_num": 30, "filter_size": 5, "pad": 0, "stride": 1},
        hidden_size=100,
        output_size=43,
        weight_init_std=0.1,
    )

    trainer = Trainer(
        network,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=EPOCHS,
        mini_batch_size=BATCH_SIZE,
        optimizer='Adam',
        optimizer_param={'lr': LR},
        evaluate_sample_num_per_epoch=1000
    )
    trainer.train()

    markers = {'train': 'o', 'test': 's'}
    x = np.arange(EPOCHS)
    plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
    plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()
