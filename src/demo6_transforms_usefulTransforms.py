from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# 日志工具
writer = SummaryWriter('logs')

# 图像数据
img_path = '../images/testImg.png'
img = Image.open(img_path)

# ToTensor
trans_toTensor = transforms.ToTensor()
img_tensor = trans_toTensor(img)
writer.add_image('ToTensor', img_tensor)

# Normalize 进行归一化
trans_norm = transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
writer.add_image('Normalize', img_norm)

# resize
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img) # 输入需要是PIL.Image类型，返回也是PIL.Image
img_resize_tensor = trans_toTensor(img_resize)
writer.add_image('Resize', img_resize_tensor)

writer.close()
