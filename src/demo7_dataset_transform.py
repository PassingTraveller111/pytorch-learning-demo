import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载 CIFAR-10 训练集
trainset = torchvision.datasets.CIFAR10(root='./dataSet/CIFAR10', train=True,
                                        download=True, transform=transform)

print(trainset)
'''
Dataset CIFAR10
    Number of datapoints: 50000
    Root location: ./dataSet/CIFAR10
    Split: Train
    StandardTransform
Transform: Compose(
               ToTensor()
               Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
           )
'''
print(trainset.classes)
'''
['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
'''

writer = SummaryWriter("./logs/CIFAR10")
for i in range(10):
    img, target = trainset[i]
    writer.add_image('train_set', img, i)
writer.close()


# 加载 CIFAR-10 测试集
testset = torchvision.datasets.CIFAR10(root='./dataSet/CIFAR10', train=False,
                                       download=True, transform=transform)

# 创建数据加载器
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)