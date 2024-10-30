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