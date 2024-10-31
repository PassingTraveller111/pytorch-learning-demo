from torch.utils.tensorboard import SummaryWriter
import cv2
import numpy as np
writer = SummaryWriter('logs')

test_img_path = '../hymenoptera_data/train/ants/0013035.jpg'

image = cv2.imread(test_img_path)

if image is not None:
    cv2.imshow('Image', image)
    cv2.waitKey(0) # 按任意键关闭图像
    cv2.destroyAllWindows()
else:
    print(f"无法读取图像文件：{test_img_path}")

# 注意可以接受的图片的类型 img_tensor (torch.Tensor, numpy.ndarray, or string/blobname): Image data
imgArr = np.array(image) # 这里使用numpy的array类型
print(imgArr.shape) # (512,768,3) 是 (H,W,C)类型的图像，这种类型的图像add_images方法需要进行dataformats

writer.add_image('test', imgArr, 1, dataformats='HWC')

writer.close()