#### 标准格式

```python
class CustomNet(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        return output
```

其中在`__init__`中完成类的继承和各个网络层的定义，在`forward`中完成对输入`x`前向传播的过程。PyTorch中的网络层相关接口可以查阅PyTorch官方文档：https://pytorch.org/docs/stable/nn.html（多看文档，熟能生巧）。



