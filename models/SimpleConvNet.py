# coding: utf-8
import sys
import os
from collections import OrderedDict

import numpy as np
import torch

from common.layers import Convolution, Relu, Pooling, Affine, SoftmaxWithLoss
from models.CNN import CNN

sys.path.append(os.pardir)  # 부모 디렉터리 파일을 가져올 수 있도록 설정


class SimpleConvNet(CNN):
    """단순한 합성곱 신경망

    Conv1 - Relu1 - Pool1 - Affine1 - Relu2 - Affine2 - Softmax

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
        conv_param={"filter_num": 30, "filter_size": 5, "pad": 0, "stride": 1},
        hidden_size=100,
        output_size=43,
        weight_init_std=0.1,
        visualize=False,
        device="cpu",
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

        # 변수 정리
        filter_num = conv_param["filter_num"]
        filter_size = conv_param["filter_size"]
        filter_pad = conv_param["pad"]
        filter_stride = conv_param["stride"]
        input_size = input_dim[1]
        conv_output_size = int(
            (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        )
        pool_output_size = int(
            filter_num * (conv_output_size / 2) * (conv_output_size / 2)
        )

        # 가중치 초기화
        self.params = {}
        rgen = np.random.default_rng(43)
        # Conv1
        self.params["W1"] = weight_init_std * rgen.logistic(
            size=(filter_num, input_dim[0], filter_size, filter_size)
        )
        self.params["b1"] = np.zeros(filter_num)
        # Affine1
        self.params["W2"] = weight_init_std * rgen.logistic(
            size=(pool_output_size, hidden_size)
        )
        self.params["b2"] = np.zeros(hidden_size)
        # Affine2
        self.params["W3"] = weight_init_std * rgen.logistic(
            size=(hidden_size, output_size)
        )
        self.params["b3"] = np.zeros(output_size)

        # 가중치를 tensor로 변경
        for key, value in self.params.items():
            self.params[key] = torch.from_numpy(value).to(device)

        # 레이어 생성
        self.layers = OrderedDict()
        self.layers["Conv1"] = Convolution(
            self.params["W1"],
            self.params["b1"],
            conv_param["stride"],
            conv_param["pad"],
        )
        self.layers["Relu1"] = Relu()
        self.layers["Pool1"] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers["Affine1"] = Affine(self.params["W2"], self.params["b2"])
        self.layers["Relu2"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W3"], self.params["b3"])
        self.last_layer = SoftmaxWithLoss()

    # 부모 클래스에 포함된 함수들.
    # def predict(self, x):
    # def loss(self, x, t):
    # def accuracy_f1score(self, x, t, batch_size=100):
    # def gradient(self, x, t):
