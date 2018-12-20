# 训练测试 and 上GPU方法
假设我们已经有了模型 model， 做好了训练集和测试集的 loader(train_loader 和 test_loader,这里我们假设一个样本就是一张图和一个label)，在不上GPU

的情况下可以:

**train**

```python

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0004) #设定一个优化器
loss_func = torch.nn.CrossEntropyLosss() ##设置损失函数
for epoch  in range(total_epochs)：      
    model.train(True)   ##将模型调为训练模式，默认是True
    for step, (data, target) in enumerate(train_loader):
    	data, target = Variable(data)，Variable(target)
		 optimizer.zero_grad()
         outputs = model(data)
         _, preds = torch.max(outputs.data, 1) #概率最大的那一类就是预测结果
         loss = loss_func(outputs, target)
         loss.backward()
         optimizer.step()
```

**测试**

```python
model.train(False)
for step, (data, target) in enumerate(test_loader):
    outputs = model(data)
    _, preds = torch.max(outputs.data, 1)
```



**GPU**

```python
##单卡版
if use_gpu:
   model = model.cuda()
for epoch in range(total_epochs):
    for step, (x, y) in enumerate(train_data):
            if use_gpu:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data)，Variable(target)
            ##...后面过程同上面
```

```python
##多卡版
#主要就是用 DataParallel 重新包装一下model，使得模型可以并行训练
net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
# 然后拿net 去训练即可，训练过程写法就同上面了，这里device_ids 表示用那几张编号的GPU去训练，不写的话默认用所有GPU
```



**在pytorch 0.4中 有比较简单的写法**

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
将模型放上gpu，如果是但gpu，不需要中间if的两行
"""
model = Model()
if torch.cuda.device_count() > 1:
  model = nn.DataParallel(model，device_ids=[0,1,2])
model.to(device)

for epoch in range(total_epochs):
    for step, (x, y) in enumerate(train_data):
            data, target = data, target = data.to(device), target.to(device)
            ## ....下同前面
```



**其他**

在训练时，我们可以使用模型中的子网络，比如我们定义了这样一个alexnet:

```python
class AlexNet(nn.Module):
def __init__(self, num_classes=1000):
    super(AlexNet, self).__init__()
    self.features = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(64, 192, kernel_size=5, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(192, 384, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),
    )
    self.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(256 * 6 * 6, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, num_classes),
    )

def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), 256 * 6 * 6)
    x = self.classifier(x)
    # return nn.functional.log_softmax(x, dim=1)
    return x
```
它有两个子模块 features 和 classifier，假设我们现在只想让数据通过features这个子模块

在非多GPU的情况下，可以使用如下代码

```python
model = AlexNet()
output = model.features(input)
```

在多GPU时:

```python
model = AlexNet()
model = nn.DataParallel(model)
out = model.module.features(input)
```

注意是加多了一个module 而不是 model