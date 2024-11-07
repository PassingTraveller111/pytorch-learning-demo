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
for data in dataloader:
    image, target = data
    output = myModule(image)
    print(output)
    result_loss = loss(output, target)
    result_loss.backward()
    print(result_loss)


