# 保存和加载模型的方法
```python
# 保存和加载整个模型
torch.save(model_object, 'model.pkl')
model = torch.load('model.pkl')
```

```python
# 仅保存和加载模型参数(推荐使用)
torch.save(model_object.state_dict(), 'params.pkl')
model_object.load_state_dict(torch.load('params.pkl'))
```

```python
# 保存DataPallel过的模型参数
 torch.save(model.module.state_dict(), PATH)
```



**加载预训练模型**

以resnet18为例

```python
from torchvision import models
from torch import nn
from torch import optim

resnet_model = models.resnet18(pretrained=True) 
# pretrained 设置为 True，会自动下载模型 所对应权重，并加载到模型中
# 也可以自己下载 权重，然后 load 到 模型中，源码中有 权重的地址。

# 假设 我们的 分类任务只需要 分 100 类，那么我们应该做的是
# 1. 查看 resnet 的源码
# 2. 看最后一层的 名字是啥 （在 resnet 里是 self.fc = nn.Linear(512 * block.expansion, num_classes)）
# 3. 在外面替换掉这个层
resnet_model.fc= nn.Linear(in_features=..., out_features=100)

# 这样就 哦了，修改后的模型除了输出层的参数是 随机初始化的，其他层都是用预训练的参数初始化的。

# 如果只想训练 最后一层的话，应该做的是：
# 1. 将其它层的参数 requires_grad 设置为 False
# 2. 构建一个 optimizer， optimizer 管理的参数只有最后一层的参数
# 3. 然后 backward， step 就可以了

# 这一步可以节省大量的时间，因为多数的参数不需要计算梯度
for para in list(resnet_model.parameters())[:-2]:
    para.requires_grad=False 

optimizer = optim.SGD(params=[resnet_model.fc.weight, resnet_model.fc.bias], lr=1e-3)


```

前一种方法只适合简单修改。有的时候我们往往要修改网络中的层次结构，这时只能用参数覆盖的方法，即自己先定义一个类似的网络，再将预训练中的参数提取到自己的网络中来。这里以resnet预训练模型举例。

```python
# coding=UTF-8
import torchvision.models as models
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

class CNN(nn.Module):

    def __init__(self, block, layers, num_classes=9):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        #新增一个反卷积层
        self.convtranspose1 = nn.ConvTranspose2d(2048, 2048, kernel_size=3, stride=1, padding=1, output_padding=0, groups=1, bias=False, dilation=1)
        #新增一个最大池化层
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        #去掉原来的fc层，新增一个fclass层
        self.fclass = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        #新加层的forward
        x = x.view(x.size(0), -1)
        x = self.convtranspose1(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fclass(x)

        return x

#加载model
resnet50 = models.resnet50(pretrained=True)
cnn = CNN(Bottleneck, [3, 4, 6, 3])
#读取参数
pretrained_dict = resnet50.state_dict()
model_dict = cnn.state_dict()
# 将pretrained_dict里不属于model_dict的键剔除掉
pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 更新现有的model_dict
model_dict.update(pretrained_dict)
# 加载我们真正需要的state_dict
cnn.load_state_dict(model_dict)
# print(resnet50)
print(cnn)
```

