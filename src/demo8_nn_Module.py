from torch import nn

class MyModule(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        '''
        首先调用了父类（nn.Module）的 __init__ 方法，这是一种良好的编程习惯，确保父类的初始化逻辑被正确执行。
        '''

    def forward(self, input):
        output = input + 1
        return output

newModule = MyModule()
print(newModule(1)) # 2