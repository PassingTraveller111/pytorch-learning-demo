import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])
input = torch.reshape(input, (-1, 1, 5, 5))
print(input.shape)
'''
torch.Size([1, 1, 5, 5])
'''

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.MaxPool = MaxPool2d(kernel_size=3, ceil_mode=True)
    def forward(self, input):
        return self.MaxPool(input)

myModule = MyModule()
output = myModule(input)
print(output)
'''
tensor([[[[2, 3],
          [5, 1]]]])
'''

dataset = torchvision.datasets.CIFAR10('./dataSet/CIFAR10', transform=torchvision.transforms.ToTensor(), train=False)
dataloader = DataLoader(dataset, batch_size=64)

writer = SummaryWriter('./logs/logs_maxpool')
step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images("inputs", imgs, step)
    output = myModule(imgs)
    writer.add_images("outputs", output, step)
    step += 1

writer.close()