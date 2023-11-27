import os
import sys
import time

sys.path.append(os.pardir)
from common.optimizer import *


class Trainer:
    def __init__(
        self,
        network,
        datas,
        epochs=20,
        mini_batch_size=100,
        optimizer="Adam",
        optimizer_param={"lr": 0.01},
        evaluate_sample_num_per_epoch=None,
        verbose=True,
        device="cpu",
    ):
        self.network = network
        (
            self.x_train,
            self.y_train,
            self.x_valid,
            self.y_valid,
            self.x_test,
            self.y_test,
        ) = datas
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch
        self.verbose = verbose
        self.device = device

        # optimizer
        optimizer_class_dict = {
            "adam": Adam,
        }
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)

        self.train_size = self.x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0
        self.progress = 0

        self.confusion_matrix = None
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []
        self.train_f1_list = []
        self.test_f1_list = []

        self.timestamp = [time.time()]

    def train_step(self, test_flg=False):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.y_train[batch_mask]

        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)

        if self.current_epoch != 0:
            loss = self.network.loss(x_batch, t_batch).cpu().numpy()
            self.train_loss_list.append(loss)
            if self.verbose:
                if int(self.current_iter / self.max_iter * 100) > self.progress:
                    self.timestamp.append(time.time())
                    delta_time = int(self.timestamp[-1] - self.timestamp[0])
                    print(
                        f"TimeStamp: {delta_time:<4}    "
                        f"progress: {self.progress + 1}%    "
                        f"train_loss: {loss:0.5f}"
                    )

        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            x_train_sample, t_train_sample = self.x_train, self.y_train
            x_test_sample, t_test_sample = self.x_valid, self.y_valid
            # testFlg True가 아니면 validation data로 시행.
            if test_flg:
                x_test_sample, t_test_sample = self.x_test, self.y_test

            if self.evaluate_sample_num_per_epoch is not None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.y_train[:t]
                x_test_sample, t_test_sample = (
                    self.x_valid[:t],
                    self.y_valid[:t],
                )
                if test_flg:
                    x_test_sample, t_test_sample = self.x_test[:t], self.y_test[:t]

            # 변수명은 test_result이나 testFlg==False 인 경우 validation.
            train_result = self.network.accuracy_f1score(
                x_train_sample, t_train_sample, self.batch_size
            )
            train_acc, train_f1, *_ = train_result
            test_result = self.network.accuracy_f1score(
                x_test_sample, t_test_sample, self.batch_size
            )
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
        self.progress = int(self.current_iter / self.max_iter * 100)
        self.current_iter += 1

    def train(self):
        for _ in range(self.max_iter):
            self.train_step()

        # accuracy -> accuracy_f1score 변경에 따른 수정.
        # Final Test Accuracy만 test Data 사용.
        test_acc, test_f1score, self.confusion_matrix = self.network.accuracy_f1score(
            self.x_test, self.y_test, is_final=True
        )
        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print(f"acc:{test_acc:0.5f}, f1score:{test_f1score:0.5f}")
