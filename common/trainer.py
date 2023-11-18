import os
import sys

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.optimizer import *


class Trainer:
    """신경망 훈련을 대신 해주는 클래스"""

    def __init__(
        self,
        network,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=20,
        mini_batch_size=100,
        optimizer="SGD",
        optimizer_param={"lr": 0.01},
        evaluate_sample_num_per_epoch=None,
        verbose=True,
    ):
        self.network = network
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch
        self.verbose = verbose

        # optimizer
        optimizer_class_dict = {
            "sgd": SGD,
            "momentum": Momentum,
            "nesterov": Nesterov,
            "adagrad": AdaGrad,
            "rmsprpo": RMSprop,
            "adam": Adam,
        }
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)

        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0

        self.confusion_matrix = None
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []
        self.train_f1_list = []
        self.test_f1_list = []

    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.y_train[batch_mask]

        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)

        if self.current_epoch != 0:
            loss = self.network.loss(x_batch, t_batch).cpu().numpy()
            self.train_loss_list.append(loss)
            if self.verbose:
                print(f"train loss: {loss:0.5f}")

        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            x_train_sample, t_train_sample = self.x_train, self.y_train
            x_test_sample, t_test_sample = self.x_test, self.y_test
            if self.evaluate_sample_num_per_epoch is not None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.y_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.y_test[:t]

            train_result = self.network.accuracy_f1score(x_train_sample, t_train_sample, self.batch_size)
            train_acc, train_f1, *_ = train_result
            test_result = self.network.accuracy_f1score(x_test_sample, t_test_sample, self.batch_size)
            test_acc, test_f1, *_ = test_result

            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)
            self.train_f1_list.append(train_f1)
            self.test_f1_list.append(test_f1)

            if self.verbose:
                print(
                    f"=== epoch:{self.current_epoch}, train acc:{train_acc}, test acc:{test_acc}, "
                    + f"train f1score:{train_f1:0.5f}, test f1score:{test_f1:0.5f} ==="
                )
        self.current_iter += 1

    def train(self):
        for _ in range(self.max_iter):
            self.train_step()

        # accuracy -> accuracy_f1score 변경에 따른 수정.
        test_acc, test_f1score, self.confusion_matrix = self.network.accuracy_f1score(
            self.x_test, self.y_test
        )
        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print(f"acc:{test_acc:0.5f}, f1score:{test_f1score:0.5f}")
