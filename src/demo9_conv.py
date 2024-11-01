import torch
import torch.nn.functional as F
# 输入
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0,1, 1]])
# 卷积核
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])
print(input.shape) # torch.Size([5, 5])
print(kernel.shape) # torch.Size([3, 3])

input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

print(input.shape) # torch.Size([1, 1, 5, 5])
print(kernel.shape) # torch.Size([1, 1, 3, 3])


'''
torch.nn.functional.conv2d

input：输入张量，形状通常为 (batch_size, in_channels, height, width)。
weight：卷积核张量，形状通常为 (out_channels, in_channels/groups, kernel_height, kernel_width)。
bias（可选）：偏置张量，形状为 (out_channels)。如果设置为 None，则不添加偏置。
stride（可选）：卷积的步长，可以是一个整数或一个包含两个整数的元组，分别表示高度和宽度方向上的步长。默认值为 1。
padding（可选）：填充大小，可以是一个整数或一个包含两个整数的元组，分别表示高度和宽度方向上的填充大小。默认值为 0。
dilation（可选）：膨胀系数，可以是一个整数或一个包含两个整数的元组，分别表示高度和宽度方向上的膨胀系数。默认值为 1。
groups（可选）：分组卷积的参数。将输入通道和输出通道分别分成 groups 组，每组分别进行卷积操作。默认值为 1。
'''
output_1 = F.conv2d(input, kernel, stride=1)
print(output_1)
'''
tensor([[[[10, 12, 12],
          [18, 16, 16],
          [13,  9,  3]]]])
'''
output_2 = F.conv2d(input, kernel, stride=2)
print(output_2)
'''
tensor([[[[10, 12],
          [13,  3]]]])
'''

output_padding1 = F.conv2d(input, kernel, stride=1, padding=1)
print(output_padding1)
'''
tensor([[[[ 1,  3,  4, 10,  8],
          [ 5, 10, 12, 12,  6],
          [ 7, 18, 16, 16,  8],
          [11, 13,  9,  3,  4],
          [14, 13,  9,  7,  4]]]])
'''
