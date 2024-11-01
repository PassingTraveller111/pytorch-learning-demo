# PyTorch Learning Demo

## 一、参考课程

[Pytorch深度学习快速入门](https://www.bilibili.com/video/BV1hE411t7RN/?spm_id_from=333.337.search-card.all.click&vd_source=43478a57ace1753f8a10a9b342b5e5b6)

## 二、环境配置

### 1.anaconda介绍

Anaconda 是一个开源的 Python 发行版，主要有以下作用：

**一、方便的环境管理**

1. 多环境隔离
   - 允许用户创建多个独立的 Python 环境，每个环境可以有不同的 Python 版本以及不同的第三方库和依赖项。这对于同时处理多个不同项目非常有用，避免了不同项目之间的库冲突问题。
   - 例如，一个项目可能需要使用 Python 2.7 和特定版本的库，而另一个项目需要 Python 3.8 和另一组库，Anaconda 可以轻松为这两个项目创建独立的环境，确保它们可以独立运行而不互相干扰。
2. 环境快速创建和切换
   - 可以快速创建新的环境并安装所需的库。用户只需指定所需的 Python 版本和一些关键的库，Anaconda 就会自动下载并安装这些依赖项，大大节省了环境搭建的时间。
   - 切换环境也很方便，通过简单的命令就可以在不同的环境之间切换，方便开发人员在不同项目之间快速切换工作环境。

**二、丰富的包管理**

1. 大量预装库
   - Anaconda 预装了许多常用的科学计算、数据分析和机器学习库，如 NumPy、Pandas、Matplotlib、Scikit-learn 等。这使得用户在开始一个新的数据分析或机器学习项目时，无需手动安装这些基础库，可以直接开始编码。
   - 对于初学者来说，这减少了安装和配置的复杂性，让他们能够更快地进入实际的编程和数据分析工作。
2. 便捷的包安装和更新
   - 使用 Anaconda 的包管理工具 conda，可以方便地安装、更新和卸载各种 Python 包。conda 不仅可以管理 Python 包，还可以安装一些非 - Python 的软件和工具，这些软件和工具通常在数据科学和科学计算中会用到。
   - 例如，可以使用 conda 安装 R 语言、Jupyter Notebook 扩展、数据库驱动程序等。conda 还支持从不同的渠道安装包，包括 Anaconda 官方仓库、第三方仓库和本地安装包。

**三、集成开发工具**

1. Jupyter Notebook 支持
   - Anaconda 集成了 Jupyter Notebook，这是一个非常流行的交互式编程环境，特别适合数据分析、机器学习和教学。Jupyter Notebook 允许用户在浏览器中创建和分享包含代码、文本、图像、公式等的文档，方便进行数据分析的探索、可视化和结果展示。
   - Anaconda 使得启动 Jupyter Notebook 变得非常容易，并且可以在不同的环境中使用不同版本的 Python 和库来运行 Notebook，满足各种项目的需求。
2. 其他开发工具集成
   - 除了 Jupyter Notebook，Anaconda 还可以集成其他开发工具，如 Spyder（一个功能强大的 Python 集成开发环境）、VS Code（通过安装 Anaconda 扩展）等。这些工具提供了更丰富的开发功能，如代码编辑、调试、版本控制等，提高了开发效率。

### 2.anaconda安装

**下载地址：**[anaconda官网](https://anaconda.org/)

进入官网，点击download anaconda

![QQ_1730199530447](./images/QQ_1730270815069.png)

这一步可以用自己的邮箱注册账号，也可以点击skip直接跳过

![QQ_1730199636496](./images/QQ_1730270789940.png)

到这里就可以选择合适的版本进行下载，这里我是M2芯片的mac，所以选择`Download for Apple Silicon`

![QQ_1730199689863](./images/QQ_1730270718282.png)

下载完成以后就可以点击安装包进行安装

打开终端输入下面的命令：

```
conda list
```

下面的情况即是安装成功

![QQ_1730199856589](./images/QQ_1730270696547.png)

### 3.虚拟环境配置

在 Anaconda 中创建虚拟环境可以通过以下步骤进行：

**一、使用命令行创建虚拟环境**

1. 打开终端（Terminal）应用程序。
2. 使用 `conda create` 命令创建虚拟环境。以下是一些常用的参数和示例：
   - 创建一个名为 `pytorch` 的虚拟环境，并指定 Python 版本为 3.8：

```plaintext
     conda create -n pytorch python=3.8
```

![QQ_1730266498115](./images/QQ_1730270674207.png)

3. 按下回车键后，Anaconda 会显示将要安装的包和所需的空间等信息，并询问你是否继续。输入 `y` 并按下回车键以继续安装。

![QQ_1730266545443](./images/QQ_1730270647273.png)

**二、使用 Anaconda Navigator 创建虚拟环境**

1. 打开 Anaconda Navigator。它通常可以在应用程序文件夹中找到，或者通过在终端中输入 `anaconda-navigator` 来启动。
2. 在 Anaconda Navigator 的主界面中，点击左侧菜单栏中的 “Environments”（环境）选项。
3. 在 “Environments” 页面中，点击右上角的 “Create”（创建）按钮。
4. 在弹出的窗口中，为虚拟环境输入一个名称（例如 `myenv`），选择一个 Python 版本（如果需要特定版本），然后点击 “Create” 按钮。

**三、激活和使用虚拟环境**

1. 命令行激活：
   - 在终端中，输入以下命令来激活创建的虚拟环境（以 `pytorch` 为例）：

```plaintext
     conda activate pytorch
```

- 激活后，终端提示符前面会显示虚拟环境的名称（例如 `(myenv)`）。
- 现在我们尝试激活之前创建的环境，从最左边的括号里的内容就可以看到，我们的环境从base切到了新建的pytorch

![QQ_1730266628473](./images/QQ_1730270623152.png)

1. 在 Anaconda Navigator 中激活：
   - 在 Anaconda Navigator 的 “Environments” 页面中，找到并选择要激活的虚拟环境，然后点击右侧的 “Play” 按钮，选择 “Open Terminal”（打开终端），这将打开一个已经激活了该虚拟环境的终端窗口。

现在你可以在激活的虚拟环境中安装、使用和管理特定的包，而不会影响其他环境或系统的全局 Python 安装。当你完成在虚拟环境中的工作后，可以使用 `conda deactivate` 命令来退出虚拟环境。

### 4.安装Pytorch

网站地址：[Pytorch官网](https://pytorch.org/)

![QQ_1730267604743](./images/QQ_1730270604724.png)

由于这里要求使用python3.9以上的版本，我们删除前面安装的环境，重新配置一个：

先切回base环境：

```
conda activate base
```

删除环境:

```
conda remove --name pytorch --all
```

![QQ_1730267303878](./images/QQ_1730270582919.png)

改为使用python3.9:

![QQ_1730267449800](./images/QQ_1730270559454.png)

最后，我们根据官网提供的安装指令，去安装pytorch，这里我选择用conda去安装：

![QQ_1730267658822](./images/QQ_1730270470499.png)

由于我使用的是M2芯片的笔记本，不支持CUDA，但是pytorch有做支持，所以直接选择Default即可

![QQ_1730267797284](./images/QQ_1730270439630.png)

检查是否安装成功：

如果是windows笔记本，可以通过下面的指令检测是否安装成功，cuda是否支持

![QQ_1730268464543](./images/QQ_1730270409257.png)

如果是M芯片的Mac笔记本，可以通过下面的命令检测是否支持GPU。如果这两个输出都为`True`，则表示 Mac 的 MPS 可用，可以在 PyTorch 中使用 GPU 加速计算。

```python
import torch
print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())
```

### 5.pycharm中使用对应的环境

用pycharm新建工程时，使用anaconda3新建
![QQ_1730269810341](./images/72d7af7d136bd1adaf85039b6480d052.png)

已有的工程也可以在设置/项目/解释器中更改，当前的anaconda会默认是base环境，如果要改成我们之前创建的pytorch环境，就需要

点击添加解释器/添加本地解释器

![QQ_1730275604019](./images/QQ_1730285089081.png)

然后最左边切到Conda环境，单选使用现有环境，点击下拉框就可以使用我们之前创建好的环境

![QQ_1730275642072](./images/QQ_1730285056159.png)

## 三、两大工具函数

### 1.`dir`

在 Python 中，`dir()` 函数是一个非常有用的内置函数。

**一、作用**

`dir()` 函数用于返回一个对象的所有属性和方法的列表（包括从父类继承来的属性和方法）。如果没有传入参数，它会返回当前作用域中的名称列表。

**二、用法示例**

1. 对模块使用：

```python
   import math
   print(dir(math))
```

这将列出 `math` 模块中的所有属性和方法，比如 `pi`、`sin`、`cos` 等。

1. 对类使用：

```python
   class MyClass:
       def __init__(self):
           self.x = 10
       def my_method(self):
           pass

   obj = MyClass()
   print(dir(obj))
```

这会列出 `MyClass` 实例对象的属性和方法，包括从 `object` 类继承来的方法，如 `__init__`、`__str__` 等，以及自定义的属性 `x` 和方法 `my_method`。

1. 没有参数时：

```python
   def my_function():
       pass

   print(dir())
```

如果在这个函数所在的作用域中调用 `dir()`，它会列出当前作用域中的名称，比如可能包括内置函数名、已导入的模块名、自定义的函数名等。

**三、用途**

1. 探索对象的功能：当你使用一个不熟悉的对象时，可以通过 `dir()` 快速了解它提供了哪些属性和方法。
2. 调试和开发：在调试过程中，可以检查对象是否具有特定的属性或方法，以帮助确定问题所在。
3. 动态编程：可以根据对象的属性和方法动态地调用它们，实现更加灵活的编程。

### 2.`help`

在 Python 中，`help()` 函数是一个用于获取帮助信息的内置函数。

**一、作用**

调用 `help()` 函数可以显示对象的文档字符串（如果有）以及关于该对象的一些帮助信息，包括对象的属性和方法的描述。

**二、用法示例**

1. 对模块使用：

```python
   import math
   help(math)
```

这将显示 `math` 模块的帮助信息，包括模块的文档字符串（如果有）以及模块中定义的函数和常量的描述。

1. 对函数使用：

```python
   def my_function():
       """This is a custom function."""
       pass

   help(my_function)
```

会显示关于 `my_function` 的帮助信息，包括函数的文档字符串。

1. 对类使用：

```python
   class MyClass:
       """This is a custom class."""
       def __init__(self):
           pass

   help(MyClass)
```

显示 `MyClass` 类的帮助信息，包括类的文档字符串以及类的方法的描述。

1. 对方法使用：

```python
   class MyClass:
       def my_method(self):
           """This is a custom method."""
           pass

   obj = MyClass()
   help(obj.my_method)
```

显示 `my_method` 方法的帮助信息。

**三、用途**

1. 学习新的模块、函数、类或方法：当你遇到一个不熟悉的 Python 内置模块或第三方库时，使用 `help()` 可以快速了解其功能和用法。
2. 查看文档字符串：可以方便地查看开发者为函数、类等编写的文档说明，以便更好地理解代码的意图和使用方式。
3. 调试和理解代码：在调试过程中，通过查看帮助信息可以更好地理解代码的行为和各个部分的作用。

## 四、Pytorch如何加载数据

### 1.`pytorch`中的`DataSet`类与`DataLoader`类

#### 1.1 Dataset类

在 PyTorch 中，`Dataset` 类是一个抽象类，用于表示数据集。它的主要目的是为数据加载提供一个统一的接口，使得数据可以方便地被模型使用。

以下是关于 `Dataset` 类的一些重要特点和用法：

1. `__init__()`：初始化方法，通常用于设置数据集的路径、预处理参数等。
2. `__len__()`：必须实现的方法，返回数据集的大小，即数据集中样本的数量。
3. `__getitem__()`：必须实现的方法，接受一个索引作为参数，返回数据集中对应索引的样本。

#### 1.2 DataLoader类

在 PyTorch 中，`DataLoader` 是一个用于数据加载的实用工具类。它可以将数据集（通常是一个实现了 `Dataset` 类的对象）包装起来，提供高效的数据加载和批处理功能。

1. `dataset`：这是一个必须的参数，是一个实现了 `Dataset` 类的对象，表示要加载的数据。
2. `batch_size`：指定每个批次的大小。例如，如果 `batch_size=32`，则每次从数据集中取出 32 个样本组成一个批次。
3. `shuffle`：如果设置为 `True`，在每个 epoch 开始时会随机打乱数据的顺序。这有助于提高模型的泛化能力。
4. `num_workers`：指定用于数据加载的子进程数量。增加这个值可以提高数据加载的速度，但也会消耗更多的系统资源。
5. `drop_last`：如果数据集的大小不能被 `batch_size` 整除，当设置为 `True` 时，会丢弃最后一个不完整的批次。

#### 1.3 使用示例

```python
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

data = [1, 2, 3, 4, 5, 6, 7, 8]
dataset = CustomDataset(data)
dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=2)

for batch in dataloader:
    print(batch)
```

#### 1.4 作用和优势

1. 高效数据加载：`DataLoader` 使用多进程（如果 `num_workers` 大于 0）来并行加载数据，从而加快数据加载速度，特别是对于大型数据集。
2. 批处理：自动将数据分成批次，方便模型进行批量处理。这对于需要批量输入的深度学习模型非常重要。
3. 数据打乱：通过设置 `shuffle=True`，可以在每个 epoch 中随机打乱数据的顺序，有助于模型更好地学习数据的分布，提高泛化能力。
4. 灵活性：可以方便地调整各种参数，如批次大小、数据加载的并行度等，以适应不同的数据集和模型需求。

### 2.蚂蚁蜜蜂分类数据集

#### 2.1 数据集下载链接

[下载链接](https://download.pytorch.org/tutorial/hymenoptera_data.zip)

#### 2.2 结构讲解

![](./images/QQ_1730287988011.png)

### 3.使用Dataset和DataLoader加载数据集

#### 3.1 基于Dataset抽象类实现一个数据集类

```python
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
```

#### 3.2 创建数据集

```python
root_dir = '../hymenoptera_data/train'
ants_label_dir = 'ants'
bees_label_dir = 'bees'
ants_dataSet = myDataSet(root_dir, ants_label_dir)
bees_dataSet = myDataSet(root_dir, bees_label_dir)
# ants_dataSet.__getitem__(0)[0].show() # 展示蚂蚁数据集的第一张图片
# 拼接数据集为训练集
train_dataSet = ants_dataSet + bees_dataSet
print(train_dataSet.__getitem__(0))
```

#### 3.3 用DataLoader进行装载

```python
train_dataLoader = DataLoader(train_dataSet, shuffle=True)
print(train_dataLoader)
for batch in train_dataSet:
    print(batch) # (<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x375 at 0x12D0E6AF0>, 'ants')
```

## 五、TensorBoard

### 1.简介

TensorBoard 是一个用于可视化和分析 TensorFlow 运行过程和结果的工具。它可以帮助你更好地理解、调试和优化机器学习模型。

### 2.安装

PyTorch 本身并不自带 TensorBoard。

但是，可以通过安装 `torch.utils.tensorboard` 模块来在 PyTorch 中使用 TensorBoard。

**安装指令**：

```
pip install tensorboard
```

**导入**：

```python
from torch.utils.tensorboard import SummaryWriter
```

### 3.`SummaryWriter`的使用方法

**一、导入模块并创建`SummaryWriter`对象**

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='runs/experiment_name')
```

这里的`log_dir`参数指定了保存事件文件的目录路径。可以根据不同的实验设置不同的目录名称。

**二、在训练过程中记录数据**

1. **标量数据（Scalars）**：

   可以记录训练过程中的损失、准确率等标量指标。

   以下是`add_scalar`方法的详细用法：

   ```python
   add_scalar(tag, scalar_value, global_step=None, walltime=None)
   ```

   - `tag`：字符串类型，表示数据的名称标签。这个标签将在 TensorBoard 中用于区分不同的数据系列。例如，可以使用`'loss'`表示损失值，`'accuracy'`表示准确率等。
   - `scalar_value`：要记录的标量值，可以是整数、浮点数等数值类型。
   - `global_step`：可选参数，通常是一个整数，表示当前的训练步数、迭代次数或 epoch 数等。这个参数用于在横坐标上显示数据的位置。如果不提供这个参数，数据将在 TensorBoard 中以无顺序的方式显示。
   - `walltime`：可选参数，通常是一个浮点数，表示记录数据的时间戳。如果不提供这个参数，将使用当前时间。

   以下是一个示例用法：

   ```python
   from torch.utils.tensorboard import SummaryWriter
   
   writer = SummaryWriter('runs/experiment')
   
   for epoch in range(10):
       loss = epoch * 0.1
       writer.add_scalar('training_loss', loss, epoch)
   
   writer.close()
   ```

   在这个例子中，每次循环都会将当前的 epoch 数和对应的损失值记录到 TensorBoard 中，使用`'training_loss'`作为标签。这样在 TensorBoard 中就可以看到随着 epoch 的增加，损失值的变化情况。

2. **图像数据（Images）**：

可以记录训练过程中的图像数据，例如输入图像、生成的图像等。

以下是`add_image`方法的详细用法：

```python
add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')
```

- `tag`：字符串类型，表示图像的名称标签。这个标签将在 TensorBoard 中用于区分不同的图像系列。
- `img_tensor`：一个形状为`(C, H, W)`或`(B, C, H, W)`的张量，表示单个图像或一批图像。其中`C`表示图像的通道数，`H`表示图像的高度，`W`表示图像的宽度，`B`表示批次大小。图像数据的值应该在`[0, 1]`或`[0, 255]`范围内。
- `global_step`：可选参数，通常是一个整数，表示当前的训练步数、迭代次数或 epoch 数等。这个参数用于在横坐标上显示图像的位置。如果不提供这个参数，图像将在 TensorBoard 中以无顺序的方式显示。
- `walltime`：可选参数，通常是一个浮点数，表示记录图像的时间戳。如果不提供这个参数，将使用当前时间。
- `dataformats`：可选参数，字符串类型，表示图像数据的格式。默认值为`'CHW'`，表示通道（Channel）、高度（Height）、宽度（Width）的顺序。如果图像数据的形状是`(B, H, W, C)`，则可以设置`dataformats='BHWC'`。

以下是一个示例用法：

```python
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np

writer = SummaryWriter('runs/experiment')

# 创建一个形状为 (3, 32, 32) 的随机图像张量
img_tensor = torch.randn(3, 32, 32)

# 将图像数据添加到 TensorBoard
writer.add_image('random_image', img_tensor)

writer.close()
```

在这个例子中，创建了一个随机的图像张量，并将其添加到 TensorBoard 中，使用`'random_image'`作为标签。这样在 TensorBoard 中就可以看到这个图像。如果要添加一批图像，可以将`img_tensor`的形状设置为`(B, C, H, W)`。

3. **直方图（Histograms）**：

可以记录张量的直方图，例如权重、偏置等参数的分布情况。

```python
   weights =...  # 模型的权重张量
   writer.add_histogram('weights', weights, epoch)
```

4. **嵌入向量（Embeddings）**：

可以记录高维数据的低维嵌入，例如词向量或图像特征向量。

```python
   embeddings =...  # 一个 Tensor 嵌入向量数据
   labels =...  # 对应的标签数据
   writer.add_embedding(embeddings, metadata=labels, global_step=epoch)
```

**三、关闭`SummaryWriter`对象**

在训练结束后，应该关闭`SummaryWriter`对象以确保所有数据都被正确写入事件文件。

```python
writer.close()
```

**四、启动 TensorBoard 进行可视化**

在命令行中运行以下命令来启动 TensorBoard：

```plaintext
tensorboard --logdir=runs/experiment_name
```

其中`runs/experiment_name`是你在创建`SummaryWriter`对象时指定的日志目录路径。然后在浏览器中打开给出的网址，就可以查看 TensorBoard 中的可视化结果了。

### 4.使用例子

#### 4.1 以`y=2x`函数为例，运行代码

```python
from torch.utils.tensorboard import SummaryWriter

# 将日志数据存储到logs目录下
writer = SummaryWriter('logs')

# y = 2x
for i in range(100):
    writer.add_scalar("y=2x", 2 * i, i)

writer.close()
```

当前目录下就会创建`logs`目录，用来存储日志文件

![QQ_1730361030438](images/QQ_1730361078331.png)

#### 4.2 启动tensorboard服务

终端输入：

```
tensorboard --logdir=src/logs
```

可以看到，tensorboard服务被运行在本地的6006端口

![QQ_1730361200165](images/QQ_1730362807515.png)

用浏览器访问端口

![QQ_1730361261086](images/QQ_1730362840346.png)

### 5.`add_image`方法

前面简单介绍了`add_image`方法，需要注意的是`add_image`的`img_tensor`可以接受的类型有` (torch.Tensor, numpy.ndarray, or string/blobname)` 

#### 5.1 `opencv`

`opencv-python`库是一个功能强大的计算机视觉库，可以用于处理图像数据。以下是一些常见的处理图像数据的方法：

**一、安装和导入**

首先，确保你已经安装了`opencv-python`库。可以使用以下命令进行安装：

```plaintext
pip install opencv-python
```

然后，在你的 Python 代码中导入该库：

```python
import cv2
```

**二、读取和显示图像**

1. 读取图像：

   使用`cv2.imread()`函数读取图像文件。该函数接受图像文件的路径作为参数，并返回一个表示图像的 NumPy 数组。

```python
   image = cv2.imread('path/to/image.jpg')
```

1. 显示图像：

   使用`cv2.imshow()`函数显示图像。该函数接受一个窗口名称和图像数组作为参数，并在指定的窗口中显示图像。

```python
   cv2.imshow('Image', image)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
```

`cv2.waitKey(0)`函数等待用户按下任意键，然后`cv2.destroyAllWindows()`函数关闭所有打开的窗口。

**三、图像基本操作**

1. 访问像素值：

   可以使用索引来访问图像中的像素值。图像是一个三维数组，其中第一个维度表示行数（高度），第二个维度表示列数（宽度），第三个维度表示颜色通道（通常是蓝、绿、红三个通道）。

```python
   # 获取图像的高度、宽度和通道数
   height, width, channels = image.shape

   # 访问特定像素的值
   pixel_value = image[y, x]  # y 和 x 是像素的坐标
```

2. 修改像素值：

   可以使用索引来修改图像中的像素值。

```python
   image[y, x] = [255, 0, 0]  # 将特定像素设置为蓝色
```

**四、图像颜色空间转换**

1. BGR 转灰度：

   使用`cv2.cvtColor()`函数将 BGR 颜色空间的图像转换为灰度图像。

```python
   gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

2. BGR 转 HSV：

   使用`cv2.cvtColor()`函数将 BGR 颜色空间的图像转换为 HSV 颜色空间的图像。

```python
   hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
```

**五、图像滤波和模糊**

1. 均值滤波：

   使用`cv2.blur()`函数对图像进行均值滤波。该函数接受图像数组和核大小作为参数，并返回滤波后的图像。

```python
   blurred_image = cv2.blur(image, (5, 5))  # (5, 5) 是核的大小
```

2. 高斯滤波：

   使用`cv2.GaussianBlur()`函数对图像进行高斯滤波。该函数接受图像数组、核大小和标准差作为参数，并返回滤波后的图像。

```python
   gaussian_blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
```

**六、图像边缘检测**

Canny 边缘检测：

使用`cv2.Canny()`函数进行 Canny 边缘检测。该函数接受图像数组、低阈值和高阈值作为参数，并返回边缘检测后的图像。

```python
   edges = cv2.Canny(image, 100, 200)
```

**七、图像形态学操作**

1. 腐蚀和膨胀：

   使用`cv2.erode()`和`cv2.dilate()`函数进行腐蚀和膨胀操作。这些函数接受图像数组、核和迭代次数作为参数，并返回操作后的图像。

```python
   kernel = np.ones((5, 5), np.uint8)
   eroded_image = cv2.erode(image, kernel, iterations=1)
   dilated_image = cv2.dilate(image, kernel, iterations=1)
```

2. 开运算和闭运算：

   使用`cv2.morphologyEx()`函数进行开运算和闭运算。该函数接受图像数组、操作类型、核和迭代次数作为参数，并返回操作后的图像。

```python
   kernel = np.ones((5, 5), np.uint8)
   opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
   closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
```

**八、图像变换**

1. 缩放：

   使用`cv2.resize()`函数对图像进行缩放。该函数接受图像数组、目标大小和插值方法作为参数，并返回缩放后的图像。

```python
   resized_image = cv2.resize(image, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR)
```

2. 旋转：

   使用`cv2.getRotationMatrix2D()`和`cv2.warpAffine()`函数对图像进行旋转。首先，使用`cv2.getRotationMatrix2D()`函数创建一个旋转矩阵，然后使用`cv2.warpAffine()`函数对图像进行旋转。

```python
   center = (width // 2, height // 2)
   angle = 45
   scale = 1
   rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
   rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
```

**九、图像保存**

使用`cv2.imwrite()`函数将处理后的图像保存到文件中。该函数接受图像文件的路径和图像数组作为参数。

```python
cv2.imwrite('path/to/saved_image.jpg', processed_image)
```

这些只是`opencv-python`库中一些常见的图像处理方法。该库还提供了许多其他功能，如目标检测、特征提取等，可以根据具体需求进行进一步的探索和使用。

#### 5.2 add_image使用例子

模块导入

```python
from torch.utils.tensorboard import SummaryWriter
import cv2
import numpy as np
```

建立`writer`对象

```python
writer = SummaryWriter('logs')
```

创建`image`和`imgArr`‘对象

```python
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
```

使用`add_image`

```
print(imgArr.shape) # (512,768,3) 是 (H,W,C)类型的图像，这种类型的图像add_images方法需要进行dataformats
writer.add_image('test', imgArr, 1, dataformats='HWC')
writer.close()
```

运行以后，用`databoard`查看日志

![](images/QQ_1730364845923.png)

## 六、transforms

### 1.介绍

在 PyTorch 中，`torchvision.transforms`是一个用于图像变换的模块。它提供了一系列的函数和类，可以对图像进行各种操作，如调整大小、裁剪、归一化、数据增强等。

### 2.用法

以下是一些常见的图像变换操作：

**转换为张量（ToTensor）:**

在 PyTorch 中，`torchvision.transforms.ToTensor`是一个常用的图像变换方法。

**作用**：

`ToTensor`将 PIL 图像（或者 NumPy 数组表示的图像）转换为 PyTorch 的张量（tensor）。具体来说，它执行以下操作：

- 将图像的像素值从范围 [0, 255] 归一化到 [0, 1]。
- 将图像的维度从 (H, W, C)（Height, Width, Channels）转换为 (C, H, W)，其中 C 是通道数（通常是 3 表示 RGB 图像），H 是图像高度，W 是图像宽度。

**示例用法**：

```python
from torchvision import transforms
from PIL import Image

img = Image.open('your_image.jpg')
transform = transforms.ToTensor()
tensor_img = transform(img)
```

在图像数据加载和预处理管道中，`ToTensor`通常与其他图像变换方法一起使用，例如调整大小、裁剪、归一化等，以将原始图像数据转换为适合神经网络输入的格式。

![](images/QQ_1730371158725.png)

**调整大小（Resize）**：

在 PyTorch 中，`transforms.Resize`是用于调整图像大小的变换操作。

**用法示例**：

```python
from torchvision import transforms

# 将图像调整为 224x224 大小
resize_transform = transforms.Resize((224, 224))

image = some_image
resized_image = resize_transform(image)
```

**主要参数**：

- `size`：可以是一个整数，表示将图像调整为正方形，短边和长边都将调整为这个整数大小。也可以是一个包含两个整数的元组 `(height, width)`，分别指定调整后的图像高度和宽度。

`transforms.Resize`可以方便地对图像进行尺寸调整，以适应不同的模型输入要求或数据处理需求。需要注意的是，调整大小的方式（如插值方法）可能会影响图像的质量和外观。默认情况下，它使用双线性插值进行图像大小调整。

除了 resize 操作，transforms 还提供了哪些图像变换操作？

介绍一下 transforms 库中常用的图像增强操作。

如何调整 transforms.Resize 的插值方法？

**中心裁剪（CenterCrop）**：

```python
transform = transforms.CenterCrop((200, 200))
```

从图像中心裁剪出指定大小的区域。

**随机裁剪（RandomCrop）**：

```python
transform = transforms.RandomCrop((150, 150))
```

随机从图像中裁剪出指定大小的区域。

**归一化（Normalize）**：

在 PyTorch 中，`transforms.Normalize` 是一种数据归一化的转换操作。

`Normalize` 通常用于将图像的像素值归一化到特定的均值和标准差。其作用是使数据具有零均值和单位方差，这有助于模型更快地收敛并提高性能。

以下是 `Normalize` 的基本用法：

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
```

在上面的例子中，`Normalize` 将输入的图像张量归一化，使得每个通道的均值变为 `0.5`，标准差变为 `0.5`。

参数说明：

- `mean`：是一个包含每个通道均值的序列。例如对于三通道彩色图像，通常设置为 `(0.5, 0.5, 0.5)` 表示将图像的每个通道的值范围从 `[0, 1]` 转换到 `[-1, 1]`。
- `std`：是一个包含每个通道标准差的序列，与 `mean` 对应。

需要注意的是，`Normalize` 通常在将图像转换为张量（如 `transforms.ToTensor`）之后使用，并且应该根据具体的数据集和任务来调整 `mean` 和 `std` 的值。

对图像的每个通道进行归一化，通常用于将图像数据调整到适合神经网络输入的范围。

**随机水平翻转（RandomHorizontalFlip）**：

```python
transform = transforms.RandomHorizontalFlip()
```

以一定的概率随机水平翻转图像。

**随机垂直翻转（RandomVerticalFlip）**：

```python
transform = transforms.RandomVerticalFlip()
```

以一定的概率随机垂直翻转图像。

**组合多个变换**：

可以使用`transforms.Compose`将多个变换组合在一起：

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

这些变换在图像分类、目标检测等计算机视觉任务中非常常用，可以增加数据的多样性，提高模型的泛化能力。

### 3.为什么需要`toTensor`

在 PyTorch 中使用 `ToTensor` 主要有以下几个重要原因：

**一、数据格式统一**

1. 标准化输入格式
   - 神经网络通常要求输入具有特定的数据格式。将图像数据转换为张量可以确保所有输入数据具有相同的类型和形状，便于在模型中进行批量处理。例如，PyTorch 的神经网络模型通常期望输入是一批张量，其中每个张量代表一个样本。
   - 不同的图像格式（如 JPEG、PNG 等）在加载后可能具有不同的数据结构（如 PIL 图像对象），通过 `ToTensor` 可以将它们统一转换为 PyTorch 张量格式，方便后续的数据处理和模型训练。
2. 维度一致性
   - 图像在以常见的库（如 PIL）加载后通常以 `(H, W, C)`（高度、宽度、通道数）的顺序存储。而神经网络通常期望输入的维度顺序为 `(C, H, W)`。`ToTensor` 可以方便地进行这种维度顺序的转换，确保数据与神经网络的输入要求一致。

**二、数值范围调整**

1. 归一化像素值
   - 原始图像的像素值通常在 `[0, 255]` 的范围。将其转换为张量后，像素值被归一化到 `[0, 1]` 的范围。这对于神经网络的训练非常重要，因为较小的数值范围可以使模型的训练更加稳定，避免数值过大导致的梯度爆炸等问题。
   - 归一化后的像素值也使得不同图像之间的数值差异相对较小，有助于模型更好地学习图像的特征，而不是被不同图像之间的绝对像素值差异所主导。

**三、与 PyTorch 生态系统集成**

1. 方便使用 PyTorch 的工具和功能
   - PyTorch 提供了丰富的工具和功能来处理张量数据，如自动求导、并行计算等。将图像转换为张量后，可以直接使用这些功能，方便地进行模型的训练和优化。
   - 例如，在训练过程中，可以使用 PyTorch 的优化器来更新模型的参数，而优化器通常期望输入是张量。
2. 与其他 PyTorch 模块无缝集成
   - `ToTensor` 可以与其他 `torchvision.transforms` 中的图像变换方法无缝组合使用，构建强大的数据预处理管道。例如，可以先对图像进行裁剪、翻转等操作，然后再使用 `ToTensor` 将其转换为张量，方便地进行数据增强和预处理。

### 4.实践

运用之前的`add_image`的例子，转换成`tensor`以后，就不需要进行`dataformate`了。

```python
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
```

**Normalize**

```python
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

writer.close()
```

打开transboard查看日志

![](images/QQ_1730433157894.png)

**Resize**

```python
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

# resize
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img) # 输入需要是PIL.Image类型，返回也是PIL.Image
img_resize_tensor = trans_toTensor(img_resize)
writer.add_image('Resize', img_resize_tensor)

writer.close()
```

transBoard的效果

![](images/QQ_1730433526311.png)

## 七、dataSet与transform

### 1.使用示例

以下是使用`torchvision.transforms`处理 CIFAR-10 数据集的示例代码：

```python
import torch
import torchvision
import torchvision.transforms as transforms

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载 CIFAR-10 训练集
trainset = torchvision.datasets.CIFAR10(root='./dataSet/CIFAR10', train=True,
                                        download=True, transform=transform)

# 加载 CIFAR-10 测试集
testset = torchvision.datasets.CIFAR10(root='./dataSet/CIFAR10', train=False,
                                       download=True, transform=transform)

# 创建数据加载器
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)
```

在上述代码中：

1. 首先定义了一组数据转换操作，包括将图像转换为张量（`ToTensor`）并进行归一化（`Normalize`）。
2. 然后使用`torchvision.datasets.CIFAR10`加载 CIFAR-10 数据集，设置`root`参数指定数据存储路径，`train=True`表示加载训练集，`download=True`表示如果数据不存在则自动下载，`transform`参数应用定义好的转换操作。

![](images/QQ_1730446698532.png)

下面就是我们存放数据集的位置

![](images/QQ_1730446975529.png)

如果下载失败，也可以直接访问这个链接进行下载 [https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz ](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

3. 最后创建数据加载器`DataLoader`，可以方便地在训练和测试过程中批量加载数据。

### 2.数据集结构解析

下面是通过debug调出来的trainset的结构

![](images/QQ_1730447440579.png)

- `classes`: 类，即这个数据集有多少种标签，这个数据集是经典的10种分类，所以对应了10种标签
- `data`: 即数据
- `root`: 数据集存储位置
- `targets`: 每个数据对应的标签，targets[0]等于6，即第一个数据为第七个种类frog
