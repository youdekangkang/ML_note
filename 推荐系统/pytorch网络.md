[Toc]

#### 没见过的网络

##### torch.nn.LayerNorm

```python
class torch.nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None)
```

- normalized_shape ： 传入整数，则被看作只有一个整数的list，会对输入的最后一维进行归一化。传入list或torch.size，最会对输入的相应维度进行归一化。
- eps ：归一化时加在分母上防止除零
- elementwise_affine : 若False，则LayerNorm层中不含有任何可学习参数。若True，则会包含可学习参数weight和bias，用于放射变换，即对输入数据归一化到均值0方差1后，乘以weight，即bias。



#### 奇怪的函数

##### model.train() & model.eval()

```python
model.train()
model.eval()
```

1. model.train()作用是**启用batch normalization和drop out。**

如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()。model.train()是保证BN层能够用到每一批数据的均值和方差。对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。

2. mode.eval()能够使神经网路**沿用batch normalization的值，而不使用drop out**

如果模型中有BN层(Batch Normalization）和Dropout，在测试时添加model.eval()。model.eval()是保证BN层能够用全部训练数据的均值和方差，即测试过程中要保证BN层的均值和方差不变。对于Dropout，model.eval()是利用到了所有网络连接，即不进行随机舍弃神经元。

训练完train样本后，生成的模型model要用来测试样本。在model(test)之前，需要加上model.eval()，否则的话，有输入数据，即使不训练，它也会改变权值。这是model中含有BN层和Dropout所带来的的性质。







