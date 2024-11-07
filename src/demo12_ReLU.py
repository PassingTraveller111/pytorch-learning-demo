import torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, 0.5],
                     [-1, 3]])
input = torch.reshape(input, (-1, 1, 2, 2))

class MyModuleReLU(nn.Module):
    def __init__(self):
        super(MyModuleReLU, self).__init__()
        self.ReLu = ReLU()

    def forward(self, input):
        return self.ReLu(input)

myModule = MyModuleReLU()
output = myModule(input)
print(output)
'''
tensor([[[[1.0000, 0.5000],
          [0.0000, 3.0000]]]])
'''


class MyModuleSigmoid(nn.Module):
    def __init__(self):
        super(MyModuleSigmoid, self).__init__()
        self.MyModuleSigmoid = Sigmoid()

    def forward(self, input):
        return self.MyModuleSigmoid(input)

dataset = torchvision.datasets.CIFAR10('./dataSet/CIFAR10', download=True, train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)
myModuleSigmoid = MyModuleSigmoid()
writer = SummaryWriter('./logs/Sigmoid')
step = 0
for data in dataloader:
    imgs ,targets = data
    writer.add_images('input', imgs, global_step=step)
    output = myModuleSigmoid(imgs)
    writer.add_images('outputs', output, step)
    step+=1
writer.close()





