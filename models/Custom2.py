# coding: utf-8
import sys
import os

import numpy as np
import torch

from common.layers import (
    Convolution,
    Relu,
    Pooling,
    Affine,
    SoftmaxWithLoss,
    BatchNormalization,
    Dropout,
)
import pickle
from collections import OrderedDict

from models.CNN import CNN

sys.path.append(os.pardir)  # 부모 디렉터리 파일을 가져올 수 있도록 설정


class Custom2(CNN):
    """배치 정규화 추가된 단순한 합성곱 신경망

    기존 Custom
    conv1 - bNorm1 - relu1 - conv2 - bNorm2 - relu2 - pool1 - affine1 - bNorm3 - relu3 - affine2 - softmax

    추가:
    conv1 - bNorm1 - relu1 - dropout1 - pool1 - conv2 - bNorm2 - relu2 - dropout2 - pool2 - conv3 - bnorm3 - relu3 - dropout3 - pool3 - affine1 - bNorm4 -  relu4 - affine2 - softmax
    이미지 사이즈:
    48(pad1)-        -       -        -  24   -24(pad1)-       -       -          -  12   -   10  -        -       -          -  5    ->


    Parameters
    ----------
    input_size : 입력 크기（MNIST의 경우엔 784）
    hidden_size_list : 각 은닉층의 뉴런 수를 담은 리스트（e.g. [100, 100, 100]）
    output_size : 출력 크기（MNIST의 경우엔 10）
    활성화 함수 - 'relu'
    weight_init_std : 가중치의 표준편차 지정（0.01）
    """

    def __init__(
        self,
        input_dim=(3, 48, 48),
        conv_num=3,
        conv_param=[
            {"filter_num": 10, "filter_size": 3, "pad": 1, "stride": 1},
            {"filter_num": 10, "filter_size": 3, "pad": 1, "stride": 1},
            {"filter_num": 10, "filter_size": 3, "pad": 0, "stride": 1},
        ],
        pool_param=[
            {"pool_h": 2, "pool_w": 2, "stride": 2, "pad": 0},
            {"pool_h": 2, "pool_w": 2, "stride": 2, "pad": 0},
            {"pool_h": 2, "pool_w": 2, "stride": 2, "pad": 0},
        ],
        dropout_ratio=[0.3, 0.3, 0.3],
        hidden_size=100,
        output_size=43,
        weight_init_std=0.1,
        device="cpu",
        visualize=False,
        batch_norm=True,
        dropout=True,
    ):
        super().__init__(
            input_dim,
            conv_param,
            hidden_size,
            output_size,
            weight_init_std,
            visualize=visualize,
            device=device,
        )

        filter_num = conv_param[0]["filter_num"]
        filter_size = conv_param[0]["filter_size"]
        filter_pad = conv_param[0]["pad"]
        filter_stride = conv_param[0]["stride"]
        input_size = input_dim[1]

        filter2_num = conv_param[1]["filter_num"]
        filter2_size = conv_param[1]["filter_size"]
        filter2_pad = conv_param[1]["pad"]
        filter2_stride = conv_param[1]["stride"]

        filter3_num = conv_param[2]["filter_num"]
        filter3_size = conv_param[2]["filter_size"]
        filter3_pad = conv_param[2]["pad"]
        filter3_stride = conv_param[2]["stride"]

        conv1_output_size = int(
            (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        )

        conv2_output_size = int(
            (conv1_output_size - filter2_size + 2 * filter2_pad) / filter2_stride + 1
        )

        conv3_output_size = int(
            (conv2_output_size - filter3_size + 2 * filter3_pad) / filter3_stride + 1
        )

        pool_output_size_original = int(
            filter3_num * (conv3_output_size / 2) * (conv3_output_size / 2)
        )

        # used for Affine1 layer.
        pool_output_size = 5 * 5 * filter3_num

        # 가중치 초기화
        self.params = {}
        rgen = np.random.default_rng(43)
        # Conv1
        self.params["W1"] = weight_init_std * rgen.logistic(
            size=(filter_num, input_dim[0], filter_size, filter_size)
        )
        self.params["b1"] = np.zeros(filter_num)
        # bNorm1
        self.params["bG1"] = 1.0
        self.params["bB1"] = 0
        # relu1 - none
        # pool1 - none

        # Conv2
        self.params["W2"] = weight_init_std * rgen.logistic(
            size=(filter2_num, filter_num, filter2_size, filter2_size)
        )
        self.params["b2"] = np.zeros(filter2_num)
        # bNorm2
        self.params["bG2"] = 1.0
        self.params["bB2"] = 0
        # relu2 - none
        # pool2 - none

        # Conv3
        self.params["W3"] = weight_init_std * rgen.logistic(
            size=(filter3_num, filter2_num, filter3_size, filter3_size)
        )
        self.params["b3"] = np.zeros(filter2_num)
        # bNorm3
        self.params["bG3"] = 1.0
        self.params["bB3"] = 0
        # relu3 - none
        # pool3 - none

        # Affine1
        self.params["W4"] = weight_init_std * rgen.logistic(
            size=(pool_output_size, hidden_size)
        )
        self.params["b4"] = np.zeros(hidden_size)
        # bNorm4
        self.params["bG4"] = 1.0
        self.params["bB4"] = 0
        # relu4 - none

        # Affine2
        self.params["W5"] = weight_init_std * rgen.logistic(
            size=(hidden_size, output_size)
        )
        self.params["b5"] = np.zeros(output_size)
        # Softmax - none
        # 가중치 초기화

        # From numpy to Tensor
        for key, value in self.params.items():
            if type(value) is np.ndarray:
                self.params[key] = torch.from_numpy(value).to(device)

        # 계층 생성
        self.layers = OrderedDict()
        for i in range(1, 1 + conv_num):
            # Conv
            self.layers[f"Conv{i}"] = Convolution(
                self.params[f"W{i}"],
                self.params[f"b{i}"],
                conv_param[i - 1]["stride"],
                conv_param[i - 1]["pad"],
            )
            # bNorm
            if batch_norm:
                self.layers[f"bNorm{i}"] = BatchNormalization(
                    self.params[f"bG{i}"],
                    self.params[f"bB{i}"],
                )
            # Relu
            self.layers[f"Relu{i}"] = Relu()
            # Dropout
            self.layers[f"Dropout{i}"] = Dropout(dropout_ratio=dropout_ratio[i - 1])
            # Pool
            self.layers[f"Pool{i}"] = Pooling(
                pool_h=pool_param[i - 1]["pool_h"],
                pool_w=pool_param[i - 1]["pool_w"],
                stride=pool_param[i - 1]["stride"],
                pad=pool_param[i - 1]["pad"],
            )

        # Affine1
        self.layers["Affine1"] = Affine(
            self.params["W4"],
            self.params["b4"],
        )
        self.layers["bNorm4"] = BatchNormalization(
            self.params["bG4"],
            self.params["bB4"],
        )
        self.layers["Relu4"] = Relu()
        self.layers["Affine2"] = Affine(
            self.params["W5"],
            self.params["b5"],
        )
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=True):
        x = x.to(self.device)
        for layer_name, layer_value in self.layers.items():
            if "bNorm" in layer_name:
                x = layer_value.forward(x, train_flg=train_flg)
            else:
                x = layer_value.forward(x)

        return x

    def gradient(self, x, t):
        """기울기를 구한다(오차역전파법).

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블

        Returns
        -------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1']、   grads['W2']、   ... 가중치
            grads['b1']、   grads['b2']、   ... 편향
            grads['bG1'],   grads['bB1'],   ... 배치 정규화 베타 감마
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
        grads["bG1"], grads["bB1"] = (
            self.layers["bNorm1"].dgamma,
            self.layers["bNorm1"].dbeta,
        )
        grads["W2"], grads["b2"] = self.layers["Conv2"].dW, self.layers["Conv2"].db
        grads["bG2"], grads["bB2"] = (
            self.layers["bNorm2"].dgamma,
            self.layers["bNorm2"].dbeta,
        )
        grads["W3"], grads["b3"] = self.layers["Conv3"].dW, self.layers["Conv3"].db
        grads["bG3"], grads["bB3"] = (
            self.layers["bNorm3"].dgamma,
            self.layers["bNorm3"].dbeta,
        )
        grads["W4"], grads["b4"] = self.layers["Affine1"].dW, self.layers["Affine1"].db
        grads["bG4"], grads["bB4"] = (
            self.layers["bNorm4"].dgamma,
            self.layers["bNorm4"].dbeta,
        )
        grads["W5"], grads["b5"] = self.layers["Affine2"].dW, self.layers["Affine2"].db

        return grads
