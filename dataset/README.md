# 加载数据集的方法
### 一、已经集合在torchvison.datasets里的数据集

torchvision.datasets 中包含了 如下数据集

* MNIST
* COCO（用于图像标注和目标检测）(Captioning and Detection
* LSUN Classification
* Imagenet-12
* CIFAR10 和 CIFAR100
* STL10

以MNIST数据集为例:

```python
import torchvision.datasets as dset
import torchvision.transforms as transforms

MNIST = dset.MNIST(root = '.', train=True, download=True,
                   transform = transforms.ToTensor()
                  )
```

参数说明: 

1. `root`：数据集存放目录

2. `train`：True = 训练集， False = 测试集

3. `download`：True = 从网上下载数据集，并把数据集放在root目录下如果数据集之前下载过，将处理过的数据(minist.py中有相关函数)放在`processed`文件夹下

4. `transform`:一个函数，将图片按照该函数进行转化，常用的有:

   ```python
   transforms.ToTensor() 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
   
   transforms.CenterCrop(size)将给定的PIL.Image进行中心切割，得到给定的size，size可以是tuple，(target_height, target_width)。size也可以是一个Integer，在这种情况下，切出来的图片的形状是正方形。
   
   transforms.RandomCrop(size) 切割中心点的位置随机选取。size可以是tuple也可以是Integer。
   
   transforms.Normalize(mean, std) 给定均值：(R,G,B) 方差：（R，G，B），将会把Tensor正则化。即：Normalized_image=(image-mean)/std。
   
   transforms.ToPILImage 将shape为(C,H,W)的Tensor或shape为(H,W,C)的numpy.ndarray转换成PIL.Image，值不变。
   
   transforms.Resize(h,w)  将图片缩放到(h,w)的大小
   
   
   此外我们通常用Compose 把多个步骤整合，比如:
   transforms.Compose([
   	transforms.CenterCrop(10),
   	transforms.ToTensor(),
   ])
   ```

5. 前面这样我们只得到了Dataset,但是并不能放入模型中去跑，还需要用DataLoader加载一下，如下：

   ~~~python
   import torch.utils.data.DataLoader as loader
   
   dateloader = loader(MNIST, batch_size=24, shuffle=True, num_work=4)
   ~~~

   其中`batch_size`是一次计算用的样本数，`shuffle`表示是否将样本打乱，`num_work`表示用多少个线程去加载。后面两种加载数据集的方法最后都要用loader，后面就不再重复了

### 二、用ImageFolder加载数据

ImageFolder是一个通用的数据加载器，数据集中的数据以以下方式组织:

```
root/dog/xxx.png
root/dog/xxy.png
root/dog/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/asd932_.png
```

那么我们就可以直接这样读入数据集:

```
dset.ImageFolder(root="root folder path", [transform, target_transform])
```

用此函数进行处理的时候，会自动会图片的label命名 0，1，3... 方便接下来的loss计算

class_names = image_datasets['train'].classes 可以会获得cat、dog 等组成的列表

```python
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
  }
```



###三、重写Dataset 类

`torch.utils.data.Dataset`是表示数据集的抽象类。您自定义的数据集应该继承`Dataset`并重写以下方法：

- `__len__` 使用`len(dataset)`将返回数据集的大小。
- `__getitem__` 支持索引，`dataset[i]`可以获取第`i`个样本

以我在玩的一个自己拍的reid数据集为例子，它的训练集是 sysu/train/whole  和 sysu/train/partial 分别是全身像和半身像，我这里是要将一张全身像和一张半身像捆绑在一起作为一个label，而label设成这两张照片是否是同一个人 在重写的类中`__init__`通常是用来保存数据路径，最好是用一个list存着，这样有索引，然后再`__getitem__`中实现读取第index个样本的方法。

```python
def default_loader(path):
    return Image.open(path).convert('RGB')

class MyTrainSet(torch.utils.data.Dataset):
    def __init__(self, img0, img1, label, loader=default_loader):
        # 1.Initialize file path or list of file names.
        self.img0 = img0
        self.img1 = img1
        self.label = label
        self.loader = loader
        self.transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor()
                        ])
    
    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (e.g. torchvision.Transform)
        # 3. Return a data pair (e.g. image and label)
        # 这里需要注意的是 第一步 read one data，是一个data
        img0 = self.transform(self.loader(self.img0[index]))
        img1 = self.transform(self.loader(self.img1[index]))
        label = self.label[index]
        return img0, img1, torch.from_numpy(np.array([label], dtype = np.float32)) 

    def __len__(self):
        # You should change 0 to the total size of your dataset
        return len(self.img0)
    

def get_train_data(dir, batch_size=4):
    whole_datapath = join(dir, "whole")
    partial_datapath = join(dir, "partial")
    category = os.listdir(whole_datapath)  # 有多少个种类
    kinds = len(category)
    img0 = []
    img1 = []
    label = []
    for i in range(kinds):  # 对whole的每一张图选择一个同类partial图片 和一个不同类partial图片捆绑
        whole_path = join(whole_datapath, category[i])
        whole_imgs = os.listdir(whole_path)
        partial_path = join(partial_datapath, category[i])
        partial_imgs = os.listdir(partial_path)
        for j in range(len(whole_imgs)):
            # 选择同类图
            img0.append(join(whole_path, whole_imgs[j]))
            idx = random.randint(1, len(partial_imgs))
            img1.append(join(partial_path, partial_imgs[idx - 1]))
            label.append(1)

            # 选择不同类图
            img0.append(join(whole_path, whole_imgs[j]))
            another_kinds = random.randint(1, kinds)
            while (another_kinds == j + 1):
                another_kinds = random.randint(1, kinds)
            another_path = join(partial_datapath, category[another_kinds - 1])
            another_imgs = os.listdir(another_path)
            idx = random.randint(1, len(another_imgs))
            img1.append(join(another_path, another_imgs[idx - 1]))
            label.append(0)      
    
    train_set = MyTrainSet(img0, img1, label)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size = batch_size,
        shuffle = False
    )

    return train_set, train_loader
```

这个写法就比较灵活多变了，不需要拘泥于一格，只要能实现`__init__` `__getitem__` `__len__` 这三个方法就好