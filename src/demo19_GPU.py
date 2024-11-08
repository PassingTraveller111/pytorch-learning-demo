import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root='./dataSet/CIFAR10', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root='./dataSet/CIFAR10', train=False, transform=torchvision.transforms.ToTensor(), download=True)

# 长度
train_data_length = len(train_data)
test_data_length = len(test_data)

# 加载数据
train_dataLoader = DataLoader(train_data, batch_size=64)
test_dataLoader = DataLoader(test_data, batch_size=64)

# 指定使用mps
device = torch.device("cuda" if torch.cuda.is_available()
                      else ("mps" if (torch.backends.mps.is_built() and torch.backends.mps.is_available())
                      else "cpu"))
# 创建网络模型
myModule = MyModule()
myModule.to(device) # 将模型移动到mps设备上

# 创建损失函数
lossFn = nn.CrossEntropyLoss()
lossFn = lossFn.to(device) # 损失函数转移到mps上
# 创建优化器
learning_rate = 0.01
optim = torch.optim.SGD(myModule.parameters(), lr=learning_rate)
# 对于优化器，在初始化时指定其参数来自于已经转移到mps上的模型，这样优化器就会自动在mps设备上操作

# 设置训练的一些参数
# 记录当前训练次数
total_learning_step = 0
# 记录测试次数
total_test_step = 0
# 训练轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter('./logs/logs_train')

for step in range(epoch):
    print(f'训练第{step + 1}轮开始')
    train_loss = 0
    for data in train_dataLoader:
        images, targets = data
        images, targets = images.to(device), targets.to(device) # 将数据转移到GPU上
        outputs = myModule(images)
        loss = lossFn(outputs, targets)
        train_loss += loss
        # 优化器优化模型
        optim.zero_grad()
        loss.backward()
        optim.step()
        # 记录训练次数
        total_learning_step += 1
        if total_learning_step % 100 == 0:
            writer.add_scalar("train_loss", train_loss, total_learning_step)
            print(f'第{total_learning_step}次训练 训练集loss：{loss}')
    print(f'训练轮数{step + 1}整体训练集的loss值：{train_loss}')
    writer.add_scalar("train_total_loss", train_loss, step + 1)

    # 测试步骤开始
    test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataLoader:
            images, targets = data
            images, targets = images.to(device), targets.to(device)  # 将数据转移到GPU上
            outputs = myModule(images)
            loss = lossFn(outputs, targets)
            test_loss += loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
            if total_test_step % 100 == 0:
                writer.add_scalar("test_loss", test_loss, total_test_step)
                print(f'第{total_test_step}次测试集loss：{loss}')
            total_test_step += 1
    print(f'训练轮数{step + 1}测试集总体loss值：{test_loss}')
    print(f'整体测试集上的正确率：{total_accuracy / test_data_length}')
    writer.add_scalar("test_total_loss", test_loss, step + 1)
    writer.add_scalar('test_accuracy', total_accuracy / test_data_length, step + 1)
    # 保存当前训练模型
    torch.save(myModule, f'myModule_{step}.pth')
    print('模型已保存')

writer.close()
