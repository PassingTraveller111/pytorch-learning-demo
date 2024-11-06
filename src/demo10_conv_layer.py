import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 加载 CIFAR-10 训练集
dataset = torchvision.datasets.CIFAR10(root='./dataSet/CIFAR10', train=False,
                                        download=True, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        # 初始化卷积操作
        # 采用彩色图像，所以通道是3。输出6通道。3*3大小的卷积核。移动间隔为1。
        self.conv1 = Conv2d(in_channels=3, out_channels= 6, kernel_size= 3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

myModule = MyModule()
print(myModule)
'''
MyModule(
  (conv1): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1))
)
'''

writer = SummaryWriter('./logs')
step = 0
for data in dataloader:
    imgs, targets = data
    output = myModule(imgs)
    print('input', imgs.shape)
    '''
    input torch.Size([64, 3, 32, 32])
    '''
    print('output', output.shape)
    '''
    output torch.Size([64, 6, 30, 30]) 6通道图片没法展示，要做尺寸变换
    '''
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)
    step+=1
