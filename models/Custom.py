# coding: utf-8
import sys
import os

import numpy as np
import torch

from common.layers import Convolution, Relu, Pooling, Affine, SoftmaxWithLoss, BatchNormalization
import pickle
from collections import OrderedDict

from models.CNN import CNN

sys.path.append(os.pardir)  # 부모 디렉터리 파일을 가져올 수 있도록 설정


class Custom(CNN):
    """배치 정규화 추가된 단순한 합성곱 신경망
    
    기존:
    conv - relu - pool - affine - relu - affine - softmax

    추가:
    conv1 - bNorm1 - relu1 - pool1 - affine1 - bNorm2 - relu2 - affine2 - softmax

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
        conv_param={"filter_num": 30, "filter_size": 5, "pad": 0, "stride": 1},
        hidden_size=100,
        output_size=43,
        weight_init_std=0.1,
        device="cpu",
    ):
        super().__init__(
            input_dim, conv_param, hidden_size, output_size, weight_init_std, device
        )

        filter_num = conv_param["filter_num"]
        filter_size = conv_param["filter_size"]
        filter_pad = conv_param["pad"]
        filter_stride = conv_param["stride"]
        input_size = input_dim[1]
        conv_output_size = int((
            input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        )
        pool_output_size = int(
            filter_num * (conv_output_size / 2) * (conv_output_size / 2)
        )

        # 가중치 초기화
        self.params = {}
        rgen = np.random.default_rng(43)
        #Conv1
        self.params["W1"] = weight_init_std * rgen.logistic(
            size=(filter_num, input_dim[0], filter_size, filter_size)
        )
        self.params["b1"] = np.zeros(filter_num)
        #bNorm1
        self.params["bG1"] = 1.0
        self.params["bB1"] = 0
        #Affine1
        self.params["W2"] = weight_init_std * rgen.logistic(
            size=(pool_output_size, hidden_size)
        )
        self.params["b2"] = np.zeros(hidden_size)
        #bNorm2
        self.params["bG2"] = 1.0
        self.params["bB2"] = 0
        #Affine2
        self.params["W3"] = weight_init_std * rgen.logistic(
            size=(hidden_size, output_size)
        )
        self.params["b3"] = np.zeros(output_size)
        #가중치 초기화


        # 가중치를 tensor로 변경
        for key, value in self.params.items():
            if not ("bG" in key or "bB" in key):    #batchNorm 아닌 경우만.
                self.params[key] = torch.from_numpy(value).to(device)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers["Conv1"] = Convolution(
            self.params["W1"],
            self.params["b1"],
            conv_param["stride"],
            conv_param["pad"],
        )
        self.layers["bNorm1"] = BatchNormalization(self.params["bG1"], self.params["bB1"]) # bNorm 추가
        self.layers["Relu1"] = Relu()
        self.layers["Pool1"] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers["Affine1"] = Affine(self.params["W2"], self.params["b2"])
        self.layers["bNorm2"] = BatchNormalization(self.params["bG2"], self.params["bB2"]) # bNorm 추가
        self.layers["Relu2"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W3"], self.params["b3"])
        self.last_layer = SoftmaxWithLoss()


    def predict(self, x, train_flg=True):
        for layername, layervalue in self.layers.items():
            if "bNorm" in layername:
                x = layervalue.forward(x, train_flg=train_flg)
            else:
                x = layervalue.forward(x)
        
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
        grads["bG1"], grads["bB1"] = self.layers["bNorm1"].dgamma, self.layers["bNorm1"].dbeta
        grads["W2"], grads["b2"] = self.layers["Affine1"].dW, self.layers["Affine1"].db
        grads["bG2"], grads["bB2"] = self.layers["bNorm2"].dgamma, self.layers["bNorm2"].dbeta
        grads["W3"], grads["b3"] = self.layers["Affine2"].dW, self.layers["Affine2"].db

        return grads