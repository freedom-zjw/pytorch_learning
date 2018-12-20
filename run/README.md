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

