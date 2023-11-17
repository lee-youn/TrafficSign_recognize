# coding: utf-8
import sys
import os

import numpy as np
import torch

from common.layers import Convolution, Relu, Pooling, Affine, SoftmaxWithLoss
import pickle
from collections import OrderedDict

from models.CNN import CNN

sys.path.append(os.pardir)  # 부모 디렉터리 파일을 가져올 수 있도록 설정


class VGG16(CNN):
    """단순한 합성곱 신경망

    conv - relu - pool - affine - relu - affine - softmax

    Parameters
    ----------
    input_size : 입력 크기（MNIST의 경우엔 784）
    hidden_size_list : 각 은닉층의 뉴런 수를 담은 리스트（e.g. [100, 100, 100]）
    output_size : 출력 크기（MNIST의 경우엔 10）
    activation : 활성화 함수 - 'relu' 혹은 'sigmoid'
    weight_init_std : 가중치의 표준편차 지정（e.g. 0.01）
        'relu'나 'he'로 지정하면 'He 초깃값'으로 설정
        'sigmoid'나 'xavier'로 지정하면 'Xavier 초깃값'으로 설정
    """

    def __init__(
        self,
        input_dim=(3, 48, 48),
        conv_param={"filter_num": 30, "filter_size": 3, "pad": 0, "stride": 1},
        hidden_size=100,
        output_size=43,
        weight_init_std=0.1,
        device="cpu",
    ):
        super().__init__(
            input_dim, conv_param, hidden_size, output_size, weight_init_std, device
        )

        # filter_num = conv_param["filter_num"]
        # filter_size = conv_param["filter_size"]
        # filter_pad = conv_param["pad"]
        # filter_stride = conv_param["stride"]
        # input_size = input_dim[1]
        # conv_output_size = (
        #     input_size - filter_size + 2 * filter_pad
        # ) / filter_stride + 1
        # pool_output_size = int(
        #     filter_num * (conv_output_size / 2) * (conv_output_size / 2)
        # )
        filter_size = conv_param["filter_size"]
        filter_nums = [3]
        for i in range(1, 5):
            for _ in range(2 if i not in range(3, 6) else 3):
                filter_nums.append(2 ** (i + 5))
        filter_nums.extend([2**9] * 3)

        fc_hidden_size = [512, 4096, 4096]
        fc_output_size = [4096, 4096, 4096]

        # 가중치 초기화
        # self.params = {}
        # rgen = np.random.default_rng(43)
        # self.params["W1"] = weight_init_std * rgen.logistic(
        #     size=(filter_num, input_dim[0], filter_size, filter_size)
        # )
        # self.params["b1"] = np.zeros(filter_num)
        #
        # self.params["W2"] = weight_init_std * rgen.logistic(
        #     size=(pool_output_size, hidden_size)
        # )
        # self.params["b2"] = np.zeros(hidden_size)
        #
        # self.params["W3"] = weight_init_std * rgen.logistic(
        #     size=(hidden_size, output_size)
        # )
        # self.params["b3"] = np.zeros(output_size)

        # 가중치 초기화
        self.params = {}
        rgen = np.random.default_rng(43)
        for i in range(1, 14):
            self.params[f"W{i}"] = weight_init_std * rgen.logistic(
                size=(filter_nums[i], filter_nums[i - 1], filter_size, filter_size)
            )
            self.params[f"b{i}"] = np.zeros(filter_nums[i])

        i = 13
        for fhs, fos in zip(fc_hidden_size, fc_output_size):
            print(fhs, fos)
            i += 1
            self.params[f"W{i}"] = weight_init_std * rgen.logistic(size=(fhs, fos))
            self.params[f"b{i}"] = np.zeros(fos)

        # 가중치를 tensor로 변경
        for key, value in self.params.items():
            self.params[key] = torch.from_numpy(value).to(device)

        # 계층 생성
        self.layers = OrderedDict()

        idx_c = idx_p = 0
        for i in range(1, 6):
            for _ in range(2 if i not in range(3, 6) else 3):
                idx_c += 1
                self.layers[f"Conv{idx_c}"] = Convolution(
                    self.params[f"W{idx_c}"], self.params[f"b{idx_c}"], stride=1, pad=1
                )
                self.layers[f"Relu{idx_c}"] = Relu()

            idx_p += 1
            self.layers[f"Pool{idx_p}"] = Pooling(pool_h=2, pool_w=2, stride=2)

        # idx_c: 13
        for i in range(1, 4):
            self.layers[f"Affine{i}"] = Affine(
                self.params[f"W{idx_c+i}"], self.params[f"b{idx_c+i}"]
            )
            if i != 3:
                self.layers[f"Relu{idx_c+i}"] = Relu()
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """
        손실 함수를 구한다.

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    # accuracy, f1score를 return 하는 함수.
    def accuracy_f1score(self, x, t, batch_size=100):
        # x : data
        # t : label

        # one hot label -> normal label
        if t.ndim != 1:
            t = torch.argmax(t, dim=1)

        labels = self.output_size

        confusion_matrix = np.zeros((labels, labels))

        # range(train data 개수 / batch_size)
        for i in range(int(x.shape[0] / batch_size)):
            # i번째 batch의 data list
            tx = x[i * batch_size : (i + 1) * batch_size]
            # i번째 batch의 label list
            tt = t[i * batch_size : (i + 1) * batch_size].cpu().numpy()

            # 매 batch당 classification
            y = self.predict(tx).cpu().numpy()
            y = np.argmax(y, axis=1)

            # confusion matrix
            for j in range(len(y)):
                confusion_matrix[tt[j]][y[j]] += 1

        # accuracy(맞은 것 세기)
        accuracy = 0.0
        for i in range(labels):
            accuracy += confusion_matrix[i][i]
        accuracy = accuracy / x.shape[0]

        # precision
        precision = [0] * labels
        precision_devider = np.sum(confusion_matrix, axis=0)

        # TP + FP
        for i in range(labels):
            # 0으로 나누는 것 방지. 우선은 그냥 0으로 뒀음.
            if precision_devider[i] == 0:
                precision[i] = 0
            else:
                precision[i] = confusion_matrix[i][i] / precision_devider[i]
        precision_avg = np.mean(precision)

        # recall
        recall = [0] * labels
        recall_devider = np.sum(confusion_matrix, axis=1)

        # TP + FN
        for i in range(0, labels):
            # 0으로 나누는 것 방지.
            if recall_devider[i] == 0:
                recall[i] = 0
            else:
                recall[i] = confusion_matrix[i][i] / recall_devider[i]
        recall_avg = np.mean(recall)

        # f1 score
        f1score = 2 * precision_avg * recall_avg / (precision_avg + recall_avg)

        return accuracy, f1score, confusion_matrix

    def gradient(self, x, t):
        """기울기를 구한다(오차역전파법).

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블

        Returns
        -------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads["W1"], grads["b1"] = self.layers["Conv1"].dW, self.layers["Conv1"].db
        grads["W2"], grads["b2"] = self.layers["Affine1"].dW, self.layers["Affine1"].db
        grads["W3"], grads["b3"] = self.layers["Affine2"].dW, self.layers["Affine2"].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, "wb") as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, "rb") as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(["Conv1", "Affine1", "Affine2"]):
            self.layers[key].W = self.params["W" + str(i + 1)]
            self.layers[key].b = self.params["b" + str(i + 1)]
