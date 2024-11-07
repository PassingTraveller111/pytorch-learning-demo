import torch.optim
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import ToTensor

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.model = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
    def forward(self, x):
        return self.model(x)

dataset = torchvision.datasets.CIFAR10('./dataSet/CIFAR10', download=True, transform=ToTensor(), train=False)
dataloader = DataLoader(dataset, batch_size=1)

myModule = MyModule()
loss = CrossEntropyLoss()
optim = torch.optim.SGD(myModule.parameters(), lr=0.001)
for epoch in range(20): # 进行20轮学习
    running_loss = 0.0 # 当前轮学习的损失清零
    for data in dataloader:
        image, target = data
        output = myModule(image)
        result_loss = loss(output, target)
        optim.zero_grad() # 将优化器中的所有参数的梯度置零。
        '''
        在进行反向传播计算梯度后，每次更新参数之前，需要先将参数的梯度清零。
        这是因为在默认情况下，PyTorch 会在每次调用.backward()时累积梯度，而不是覆盖之前的梯度值。
        如果不将梯度清零，那么在多次迭代中梯度会不断累加，导致参数更新出现错误的结果。
        '''
        result_loss.backward() # 反向传播
        optim.step() # 更新参数
        running_loss += result_loss
    print(running_loss)

