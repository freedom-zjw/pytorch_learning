# FashionAI
## Environment
- Dependencies: 
  - [CUDA](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn) with GPU
  - [PyTorch](https://github.com/pytorch/pytorch) with packages ([torchvision](https://github.com/pytorch/vision)) installed



## Prepare

- 将数据集下载到datasets文件夹。训练集命名为base，测试集命名为rank，热身数据集命名为web

- 数据集下载地址：https://pan.baidu.com/s/1au14kMRKt_TuOVm80ruopw 

  密码：`vqsg`

## Usage
* 运行参数可用下面的命令查看

  ```bash
  python3 main.py --help
  ```

  默认运行resnet34网络结构，训练测试 'coat_length_labels'，batch_size=128，epochs=50, learning_rate=0.01, momentum=0, 使用GPU运行

* 用resnet50 跑 ‘’collar_design_labels“ 的示例。

  ```bash
  python3 main.py --model 'resnet50' --attribute 'collar_design_labels' --epochs 60 --batch-size 128 --lr 0.01 --momentum 0.5
  ```

  每个epoch训练出来的模型存储在save/[attribute]/[model]


