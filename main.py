import os
import numpy as np
import matplotlib.pyplot as plt

from CNN import CNN
from common.trainer import Trainer

from data.load_data import load_data
from dataset.mnist import load_mnist


if __name__ == "__main__":
    #총 데이터 :        129430
    #testdata :         7766
    #traindata:         90601개 중 N개
    #validataion data:  31063개 중 N개
    N = 6400
    # N = traindata[:n] (train data와 validation data를 앞에서부터 n개만 사용)
    EPOCHS = 10
    BATCH_SIZE = 64
    # 한 번에 계산하는 데이터 개수
    LR = 0.005

    x_train, y_train, x_validation, y_validation, x_test, y_test = load_mnist(flatten=False, normalize=False)
    # load_data(N=none, printflag=True)
    # load_mnist(flatten=True, normalize=False)
    
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
