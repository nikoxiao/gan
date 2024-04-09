# import libraries

import torch
import torch.nn as nn

import pandas
import matplotlib.pyplot as plt
import random
import numpy


def generate_real():
    real_data = torch.FloatTensor(
        [random.uniform(0.8, 1.0),
         random.uniform(0.0, 0.2),
         random.uniform(0.8, 1.0),
         random.uniform(0.0, 0.2)])
    return real_data

# 鉴别器类
class Discriminator(nn.Module):
    def __init__(self):
        # 初始化父类
        super().__init__()
        # 定义神经网络层
        self.model = nn.Sequential(
            nn.Linear(4, 3),
            nn.Sigmoid(),
            nn.Linear(3, 1),
            nn.Sigmoid
        )
        # 创建损失函数
        self.loss_function = nn.MSELoss()
        # 创建优化器，使用随机梯度下降
        self.optimiser = torch.optim.SGD(self.parameters(),lr=0.01)
        # 计数器与进程记录
        self.counter = 0
        self.progress = []