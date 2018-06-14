# pytorch_learning
learning about how to use pytorch

PyTorch is a deep learning framework for fast, flexible experimentation

### Documents

English ver:  https://pytorch.org/docs/stable/index.html

chinese ver: https://pytorch-cn.readthedocs.io/zh/latest/ <br>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;https://ptorch.com/docs/1/<br>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[index](docs/index.md)

### Installtion

   Base on python3

* **Windows**

  * Step1: Install ANACONDA: https://www.anaconda.com/download/

    ```shell
    Add anaconda to environment path:
    installpath\Anaconda3
    installpath\Anaconda3\Scripts
    installpath\Anaconda3\Library\bin
    ```

  * Step2: Install CUDA: https://developer.nvidia.com/cuda-downloads

  * Step3: Create a virtural environment for  pytorch

    ```shell
    conda create -n pytorch python=3.6
    ```

  * Step4: Activate your virtural environment

    ```shell
    activate pytorch
    ```

    PS: if you finish your work in the virtural environment and want to quit the ve, use the flowing command:

    ```
    deactivate pytorch
    ```

  * Step5: In the virtural environment:

    ```shell
    # for cuda8.0
    conda install pytorch -c pytorch 
    # for cuda9.0
    conda install pytorch cuda90 -c pytorch 
    ```

  * Step6: use the flowing python code to test:

    ~~~shell
    import torch
    print(torch.__version__)
    ~~~

  * Step7: install torchvision

    ```shell
    pip install torchvision
    ```

* **Linux**

  * Step1: install Anaconda 

  * Step2: Add Tsinghua Open Source Mirror

    ```shell
    conda config
    ```

    Use the above command to generate a configuration file `.condarc`. Then

    ```shell
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
    conda config --set show_channel_urls yes
    ```

    delete '-defaults', the content of .condarc just like:

    ```shell
    channels:
      - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
      - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
      - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
    show_channel_urls: true
    ```

  * Step3: install pytorch and torchvision:

    ```shell
    # for cuda8.0
    conda install pytorch torchvision -c pytorch
    # for cuda9.0
    conda install pytorch torchvision cuda90 -c pytorch
    ```

    ​

