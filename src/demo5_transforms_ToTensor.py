from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

img_path = '../hymenoptera_data/train/ants/6240329_72c01e663e.jpg'

img = Image.open(img_path)
transform = transforms.ToTensor()
tensor_img = transform(img)


writer = SummaryWriter('logs')
writer.add_image('test', tensor_img, 1)

writer.close()