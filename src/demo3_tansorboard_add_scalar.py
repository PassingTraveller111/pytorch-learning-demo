from torch.utils.tensorboard import SummaryWriter

# 将日志数据存储到logs目录下
writer = SummaryWriter('logs')

# y = 2x
for i in range(100):
    writer.add_scalar("y=2x", 2 * i, i)

writer.close()