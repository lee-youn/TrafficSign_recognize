# coding: utf-8
import torch


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int_)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = torch.zeros(x)
    grad[x >= 0] = 1
    return grad


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - torch.max(x, dim=0)[0]
        y = torch.exp(x) / torch.sum(torch.exp(x), dim=0)
        return y.T

    x = x - torch.max(x)[0]  # 오버플로 대책
    return torch.exp(x) / torch.sum(torch.exp(x))


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -torch.sum(torch.log(y[torch.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(x, t):
    y = softmax(x)
    return cross_entropy_error(y, t)
