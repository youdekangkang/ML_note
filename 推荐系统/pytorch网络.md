[Toc]

#### 没见过的网络

##### torch.nn.LayerNorm

```python
class torch.nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None)
```

- `normalized_shape` ： 传入整数，则被看作只有一个整数的list，会对输入的最后一维进行归一化。传入list或torch.size，最会对输入的相应维度进行归一化。
- `eps` ：归一化时加在分母上防止除零
- `elementwise_affine` : 若False，则LayerNorm层中不含有任何可学习参数。若True，则会包含可学习参数weight和bias，用于放射变换，即对输入数据归一化到均值0方差1后，乘以weight，即bias。

##### torch.nn.CrossEntropyLoss

```python
CLASS torch.nn.CrossEntropyLoss(weight: Optional[torch.Tensor] = None, ignore_index: int = -100, reduction: str = 'mean')
```

该损失函数结合了`nn.LogSoftmax()`和`nn.NLLLoss()`两个函数。它在做分类（具体几类）训练的时候是非常有用的。在训练过程中，对于每个类分配权值，可选的参数权值应该是一个1D张量。当你有一个不平衡的训练集时，这是是非常有用的。

- weight : 给定每个类手动调整缩放权重。 如果给定，则必须是大小为C维的张量.
- ignore_index : 指定一个目标值，该目标值将被忽略并且不会影响输入梯度。当`size_average` 为`True`时，损失是对non-ignored目标的平均值。
- reduction : 指定要应用于输出的缩减量：'none'| 'mean' | 'sum'。 'none':不应用reduction， 'mean':采用输出的加权平均值， 'sum': 对输出求和。注意：size_average 和reduce 正在弃用过程中，与此同时，指定这两个args中的任何一个将覆盖reduction。 默认值：'mean'。

##### torch.optim.Adam

```python
class torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)[source]
```

在[[1412.6980\] Adam: A Method for Stochastic Optimization (arxiv.org)](https://arxiv.org/abs/1412.6980)这个文章中被提出

这里的torch.optim是优化算法的包，所以这里调用的是Adam算法的接口

- params (iterable) – 待优化参数的iterable或者是定义了参数组的dict
- lr (float, 可选) – 学习率（默认：1e-3）
  - 较大的值（如 0.3）在学习率更新前会有更快的初始学习，而较小的值（如 1.0E-5）会令训练收敛到更好的性能。
- betas (Tuple[float, float], 可选) – 用于计算梯度以及梯度平方的运行平均值的系数（默认：0.9，0.999）
  - beta1：一阶矩估计的指数衰减率（如 0.9）
  - beta2：二阶矩估计的指数衰减率（如 0.999）。该超参数在稀疏梯度（如在 NLP 或计算机视觉任务中）中应该设置为接近 1 的数
- eps (float, 可选) – 为了增加数值计算的稳定性而加到分母里的项（默认：1e-8）
  - 该参数是非常小的数，其为了防止在实现中除以零
- weight_decay (float, 可选) – 权重衰减（L2惩罚）（默认: 0）

构建出优化器之后，使用step()方法对所有的参数进行更新：

```python
optimizer.step()
```





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



##### torch.unsqueeze() & torch.squeeze()

torch.squeeze() 这个函数主要对数据的维度进行压缩，去掉维数为1的的维度，比如是一行或者一列这种，一个一行三列（1,3）的数去掉第一个维数为一的维度之后就变成（3）行。squeeze(a)就是将a中所有为1的维度删掉。不为1的维度没有影响。a.squeeze(N) 就是去掉a中指定的维数为一的维度。还有一种形式就是b=torch.squeeze(a，N) a中去掉指定的定的维数为一的维度。

torch.unsqueeze()这个函数主要是对数据维度进行扩充。给指定位置加上维数为一的维度，比如原本有个三行的数据（3），在0的位置加了一维就变成一行三列（1,3）。a.squeeze(N) 就是在a中指定位置N加上一个维数为1的维度。还有一种形式就是b=torch.squeeze(a，N) a就是在a中指定位置N加上一个维数为1的维



