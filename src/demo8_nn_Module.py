from torch import nn

class MyModule(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = input + 1
        return output

newModule = MyModule()
print(newModule(1)) # 2