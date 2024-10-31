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

