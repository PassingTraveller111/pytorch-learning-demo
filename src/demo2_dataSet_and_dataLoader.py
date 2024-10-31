import os.path
from PIL import Image
from gmpy2 import trunc
from torch.utils.data import Dataset, DataLoader

print(help(Dataset))
'''
Help on class Dataset in module torch.utils.data.dataset:

class Dataset(typing.Generic)
 |  An abstract class (抽象类) representing a :class:`Dataset`.
 |  
 |  All datasets that represent a map from keys to data samples should subclass
 |  it. （所有的数据集都需要继承这个类） All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
 |  data sample for a given key. Subclasses could also optionally overwrite
 |  :meth:`__len__`, which is expected to return the size of the dataset by many
 |  :class:`~torch.utils.data.Sampler` implementations and the default options
 |  of :class:`~torch.utils.data.DataLoader`. Subclasses could also
 |  optionally implement :meth:`__getitems__`, for speedup batched samples
 |  loading. This method accepts list of indices of samples of batch and returns
 |  list of samples.
 |  
 |  .. note::
 |    :class:`~torch.utils.data.DataLoader` by default constructs an index
 |    sampler that yields integral indices.  To make it work with a map-style
 |    dataset with non-integral indices/keys, a custom sampler must be provided.
 |  
 |  Method resolution order:
 |      Dataset
 |      typing.Generic
 |      builtins.object
 |  
 |  Methods defined here:
 |  
 |  __add__(self, other: 'Dataset[_T_co]') -> 'ConcatDataset[_T_co]'
 |  
 |  __getitem__(self, index) -> +_T_co
 |  
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |  
 |  __dict__
 |      dictionary for instance variables (if defined)
 |  
 |  __weakref__
 |      list of weak references to the object (if defined)
 |  
 |  ----------------------------------------------------------------------
 |  Data and other attributes defined here:
 |  
 |  __orig_bases__ = (typing.Generic[+_T_co],)
 |  
 |  __parameters__ = (+_T_co,)
 |  
 |  ----------------------------------------------------------------------
 |  Class methods inherited from typing.Generic:
 |  
 |  __class_getitem__(params) from builtins.type
 |  
 |  __init_subclass__(*args, **kwargs) from builtins.type
 |      This method is called when a class is subclassed.
 |      
 |      The default implementation does nothing. It may be
 |      overridden to extend subclasses.

None
'''

'''
实现一个数据集类
'''

class myDataSet(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir # 数据集根地址
        self.label_dir = label_dir # 数据集标签地址 ants or bees
        self.path = os.path.join(root_dir, label_dir) # 数据集图片地址，用根目录和标签目录拼接
        self.img_dir = os.listdir(self.path)
    def __getitem__(self, index):
        img_name = self.img_dir[index]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label
    def __len__(self):
        return len(self.img_dir)

root_dir = '../hymenoptera_data/train'
ants_label_dir = 'ants'
bees_label_dir = 'bees'
ants_dataSet = myDataSet(root_dir, ants_label_dir)
bees_dataSet = myDataSet(root_dir, bees_label_dir)
# ants_dataSet.__getitem__(0)[0].show() # 展示蚂蚁数据集的第一张图片

'''
数据集可以拼接
这里用DataLoader加载数据
'''
train_dataSet = ants_dataSet + bees_dataSet
print(train_dataSet.__getitem__(0))
train_dataLoader = DataLoader(train_dataSet, shuffle=True)
print(train_dataLoader)
for batch in train_dataSet:
    print(batch) # (<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x375 at 0x12D0E6AF0>, 'ants')
