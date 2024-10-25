## 1. Environment setup (x86)
- Docker pull
    - ` $ sudo docker pull nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04`
- 启动镜像
    - `$ sudo docker run --name yolo-v8-seg -itd  -v /home/ntueee/yangjianbing:/root/code -p 2015:22   -e NVIDIA_DRIVER_CAPABILITIES=compute,utility --gpus all --shm-size="12g" --restart=always nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04`
- Install ssh (Note that enter container first!)
    - `$ apt-get update`
    - `$ apt-get install vim`
    - `$ apt-get install openssh-server`
    - 设置root密码，后续登录会用到: `$ passwd`
    - 修改配置文件: `$ vim /etc/ssh/sshd_config`
        ``` 
        #PermitRootLogin prohibit-password
        PermitRootLogin yes
        UsePAM yes 修改为 no
        注释这一行PermitRootLogin prohibit-password
        添加一行PermitRootLogin yes
        UsePAM yes 修改为 no #禁用PAM
        ```
    - 重启ssh服务: `$ service ssh restart`
    - 添加开机启动
        - 新建`power_launch.sh`文件，放到根目录：`/root`下，`power_launch.sh`添加如下内容
            ```
            #!/bin/sh -e
            service ssh start &
            ```
        - 获取读写权限：`chmod 777 /root/power_launch.sh`
        - 编辑`~/.bashrc`: `vim ~/.bashrc`，在下面添加
            ```
            if [ -f /root/power_launch.sh ]; then
                    ./root/power_launch.sh
            fi
            ```
- Install pytorch 1.13
    - 创建python软链接：`$ ln -s /usr/bin/python3 /usr/bin/python` （注意python的版本，下载pytorch时要对应）
    - install pip package: `apt-get install pip`
    - 下载[cu116/torch-1.13.0%2Bcu116-cp38-cp38-linux_x86_64.whl](https://download.pytorch.org/whl/cu116/torch-1.13.0%2Bcu116-cp38-cp38-linux_x86_64.whl)，安装：`pip install torch-1.13.0+cu116-cp38-cp38-linux_x86_64.whl`
    - 下载[cu116/torchvision-0.14.0%2Bcu116-cp38-cp38-linux_x86_64.whl](https://download.pytorch.org/whl/cu116/torchvision-0.14.0%2Bcu116-cp38-cp38-linux_x86_64.whl)，安装：`pip install torchvision-0.14.0+cu116-cp38-cp38-linux_x86_64.whl`
    - 测试是否用的GPU：
        ```
        import torch
        flag = torch.cuda.is_available()
        if flag:
            print("CUDA is available")
        else:
            print("CUDA is not available")
        ```
- Install COCO API
    - pip install Cython
    - pip install pycocotools==2.0.0

- Install dependence
    - apt update && apt install -y libsm6 libxext6
    - apt-get install -y libxrender-dev
    - apt-get install libgl1-mesa-glx
    - pip install onnxslim
    - pip install onnxruntime-gpu
    - pip install pyyaml
    - pip install psutil
    - pip install pandas
    - 安装tensorRT参考[这里](../../02-Segmetation/yolact_trt/README.MD)

## [code](https://docs.ultralytics.com/tasks/segment/)

### 制作训练标签(.txt)
```
一、每个图像对应一个.txt标签, .txt标签的格式如下：
    <class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>
    <class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>
    解释：
    1、每一行代表图片中的一个目标，有多少个目标就有多少行
    2、<class-index>：类别标签，(e.g., 0 for person, 1 for car, etc.)
    3、<x1> <y1>为轮廓的坐标，注意：<x1> <y1>做归一化处理，即x/图像宽度，y/图像高度

二、数据集格式
    my_dataset/
    │
    ├── images/
    │   ├── train/
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   │   └── ...
    │   ├── val/
    │   │   ├── img3.jpg
    │   │   └── ...
    │   └── test/   # 可选
    │       └── img4.jpg
    │
    └── labels/
        ├── train/
        │   ├── img1.txt   # 对应 img1.jpg 的标签
        │   ├── img2.txt   # 对应 img2.jpg 的标签
        │   └── ...
        ├── val/
        │   ├── img3.txt   # 对应 img3.jpg 的标签
        │   └── ...
        └── test/  # 可选
            └── img4.txt   # 对应 img4.jpg 的标签

    数据集格式必须整理成这个样子

三、写一个.yaml文件，用来指向数据集，训练时，代码中直接载入这个.yaml文件，.yaml文件格式如下：

# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: /root/code/dataset/containercode # dataset root dir，就是上面的my_dataset路径
train: images/train # train images 
val: images/val # val images 
test: # test images (optional)

# Classes，若有多个类别，一次往下添加
names:
    0: person
    1: car
```
- coco标注格式的数据转化，运行[dataset_preprocess/coco_convert.py](dataset_preprocess/coco_convert.py)下的脚本即可
    - 启动指令示例：`python coco_convert.py --ann_dir /root/code/dataset/cornerline-ningbobeier/cornerline-ningbobeier/annotations/ --ann_name instances_Validation.json --label_save_dir /root/code/AiEngineering/02-Segmetation/Yolo-v8/datas --class_names cornerline.name --train_val val`
    - 参数说明：
        - --ann_dir coco标签的存储路径
        - --ann_name coco标签的名称
        - --label_save_dir `制作训练标签(.txt)`那一步中`labels`的存储路径
        - --class_names class的存储路径（`.name`文件）
        - --train_val 生成train or val文件夹
### Train
- 按照`制作训练标签(.txt)`中的第三步，写一个`dataset.yaml`文件
- 在`Yolo-v8`下新建`pre_trained_weights`文件夹：`mkdir pre_trained_weights`
- 下载[预训练模型](https://pan.baidu.com/s/1u8JTcb-qVNzRAwPe2oYxUw)(提取码：1234)，只要下载分割相关的（`yolov8n-seg.pt`, `yolov8s-seg.pt`, `yolov8m-seg.pt`, `yolov8l-seg.pt`, `yolov8x-seg.pt`, 模型的依次从小到大），并将其拷贝到`pre_trained_weights`文件夹
- 下载[check模型](https://pan.baidu.com/s/1u8JTcb-qVNzRAwPe2oYxUw)(提取码：1234)中的`yolov11n.pt`，并将其拷贝到`Yolo-v8/ultralytics`下。（代码会自动下载，如果没有网络的话，手动下载下）
- 找到`yolov8_train.py`文件
    - 在`line3`处载入相关的预训练模型
    - 在`line6`处载入`制作训练标签(.txt)`第三步中写的`dataset.yaml`文件
- 训练
    ```
    cd 02-Segmetation/Yolo-v8/ultralytics
    python yolov8_train.py
    ```
     训练结果会保存在`Yolo-v8ultralytics/runs`文件夹中
### Forward
- `cd Yolo-v8/ultralytics`
- `mkdir results`
- `python yolov8_predict.py`
    - 会在`results`文件加下生成带有预测结果的图片

### export onnx
- `cd Yolo-v8/ultralytics`
- 找到`yolov8_onnx.py`文件
- 在`line5`处载入相关的训练出的权重模型
- `python yolov8_onnx.py`
    - 注意`line8`处的`dynamic`参数，一般设置成`True`
- 导出结束后，会在`模型`所在的目录下生成相应`.onnx`文件
- 注意：导出文件时，看下输入size和输出size

### trt convert and inference
- 环境配置参考[这里](../../02-Segmetation/yolact_trt/README.MD)
- `cd Yolo-v8/ultralytics`
- 找到`yolov8_trt.py`文件的line293，
    - `batch_size, _, nHeight, nWidth = (1, 3, 640, 640)`, 这里的batch_size=1，也可以改成2， 3， 4...，需要注意的是batch_size>1时，对应的解码函数，需要重写下
- batch_size=1时，配置好对应的入口参数后，直接运行：
    - `python yolov8_trt.py`

- `trt`文件会存储在`trt文件夹`中，预测结果存在`img_out`文件夹中

