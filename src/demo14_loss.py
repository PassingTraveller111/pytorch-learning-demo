import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

inputs = torch.tensor([1, 2, 3], dtype=torch.float)
targets = torch.tensor([1, 2, 5], dtype=torch.float)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3)) # 以便适应一些模型的输入

loss = L1Loss(reduction='sum') # 计算损失时将所有样本的损失值求和
result = loss(inputs, targets)

loss_mse = MSELoss()
result_mse = loss_mse(inputs, targets)

loss_ce = CrossEntropyLoss()
result_ce = loss_ce(inputs, targets)

print(result)
print(result_mse)
print(result_ce)