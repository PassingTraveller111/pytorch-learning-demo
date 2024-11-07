import torchvision.datasets
from torch.nn import Linear
from torchvision.models import VGG16_Weights

# trainSet = torchvision.datasets.ImageNet('./dataSet/ImageNet', split='train', download=True, transform=torchvision.transforms.ToTensor())
# ImageNet这个数据集数据量太大了

vgg16 = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT) # 通过预置的参数进行权重配置,会通过配好的链接去下载参数，这些参数会被缓存在电脑本地
'''
pretrained参数已经被废弃
parameters:
weights (VGG16_Weights, optional) – The pretrained weights to use. See VGG16_Weights below for more details, and possible values. By default, no pre-trained weights are used.

progress (bool, optional) – If True, displays a progress bar of the download to stderr. Default is True.

**kwargs – parameters passed to the torchvision.models.vgg.VGG base class. Please refer to the source code for more details about this class.
'''

print(vgg16)
'''
下面则是这个训练好的vgg16模型
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    ......
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
'''
# 下面用CIFAR10数据集进行训练
train_data = torchvision.datasets.CIFAR10('./dataSet/CIFAR10', download=True, transform=torchvision.transforms.ToTensor(), train=True)

# 由于CIFAR10数据集有10种类别，而上面的vgg16模型有1000种输出，所以我们要进行迁移学习，改一下输出的类别
vgg16.classifier.add_module('add_linear', Linear(1000, 10))
print(vgg16)
'''
可以看到，新的vgg16模型多了一层线性层 add_linear
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    ......
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    ......
    (6): Linear(in_features=4096, out_features=1000, bias=True)
    (add_linear): Linear(in_features=1000, out_features=10, bias=True)
  )
)
'''