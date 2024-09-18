#  [Yolo-V8]()

![](./Figure/overview.png)


[参考文档]()

## 1. Network

## 2 Loss function

## 3 Annotation

## 4 [Code，参考的这个](https://github.com/bubbliiiing/yolov8-pytorch)
### 4.1 Environment setup (x86)
- Docker pull
    - ` $ sudo docker pull nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04`
- 启动镜像
    - `$ sudo docker run --name yolo-v8 -itd  -v /home/ntueee/yangjianbing:/root/code -p 2014:22   -e NVIDIA_DRIVER_CAPABILITIES=compute,utility --gpus all --shm-size="12g" --restart=always nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04`
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
    - pip install -r requirements.txt

### 4.2 Object Detection
#### 4.2.1 Train
- 制作训练标签 (.txt file)
    ```
    1、训练标签
    每行：image_path x_min,y_min,x_max,y_max,cls x_min,y_min,x_max,y_max,cls ... (注意：类别标签从0开始，如果有10个类，则类别为 0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    2、类别标签 (`.txt` file)，如果是10个类
    cls0
    cls1
    cls2
    cls3
    cls4
    cls5
    cls6
    cls7
    cls8
    cls9

    For example (class.txt):
    aerop
    bicyc
    bird
    boat
    bottl
    bus
    car
    cat
    chair
    cow
    ```

- COCO格式的训练标签生成，在`CenterNet/code/generate_train_label`目录下找到[coco_annotation.py](../CenterNet/code/generate_train_label/coco_annotation.py)，指令如下：
    ```
    python coco_annotation.py --ann_dir coco标签的存储路径 --ann_name coco标签的名称 --label_save_dir 生成训练标签的存储路径 --label_save_name 生成训练标签的存储名称 --class_names 生成coco中类别文件的存储名称 --images_dir 训练图像的存储目录

    For example:
    python coco_annotation.py --ann_dir /root/code/dataset/annotations --ann_name instances_train2017.json --label_save_dir /root/code/dataset/annotations --label_save_name instances_train2017.txt --class_names coco_classes.txt --images_dir /root/code/dataset/train2017
    ```

- 运行训练代码
    - 在`code`目录下新建`model_data`和`logs`文件夹：`mkdir model_data`, `mkdir logs`
    - 将[预训练模型](https://pan.baidu.com/s/1u8JTcb-qVNzRAwPe2oYxUw)(提取码：1234)拷贝到model_data。注意：里面有`yolov8_n.pth`（ 非常小的模型，参数和计算量都很少）,`yolov8_s.pth`（小型模型，适合于需要快速推理但仍能提供较好精度的任务）,`yolov8_m.pth`（中型模型，适中的计算开销和较高的精度）,`yolov8_l.pth`（大型模型，拥有更多的参数和更高的计算开销）,`yolov8_x.pth`（最大的模型，计算开销和内存占用最高），5个权重文件。
    - `cd code`
    - 找到`train.py`文件，找到参数`model_path`（line 97）和`phi`（line 110），加载预训练模型`model_path`时，要将参数`phi`做对应的修改
    - 找到`train.py`文件，找到参数`Unfreeze_batch_size`(line 188), 根据显存大小，选择合适的`batch_size`
    - 导入数据集：数据集的格式要与上述`制作训练标签`中描述的一致
        - 类别标签: line 77 `classes_path`（.txt）
        - 训练标签：line 249 `train_annotation_path` （.txt）
        - 验证标签：line 250 `val_annotation_path` (.txt)
    - 训练:`python train.py`

#### 4.2.2 预测
    - 修改[yolo.py](code/yolo.py)文件中的`phi`（line 42）参数，改成与训练时对应的模型
    - 修改[yolo.py](code/yolo.py)文件中的`model_path`(line 28)：权重路径
    - 修改[yolo.py](code/yolo.py)文件中的`classes_path`(line 29)：类别路径
    - 预测：`python predict.py`，然后输入需要预测的图片路径，最后会在`logs`文件夹下生成`output_image.png`图片

#### 4.2.3 Estimation


#### 4.2.4 convert to onnx
- 环境配置参考[这里](../../02-Segmetation/yolact_trt/README.MD)
- cd code
- 修改[yolo_detection_onnx.py](code/yolo_detection_onnx.py)中的`model_path`(line 28), `classes_path`(line 29)参数,改成与网络对应的模型权重与类别
- `python detection_onnx.py`，会在`model_data`文件夹下生成`yolo_detection_models.onnx`

#### 4.2.5 onnx convert to trt
- 单帧预测（batch size = 1）
- `cd code`
- `python detection_trt.py --onnxFile onnx 文件路径 --trtFile_save_dir 生成的trt文件存储目录 --trtFile_save_name 生成的trt文件存储名称 --FPMode FP32或FP16 （保存浮点数32位或16位）--images_dir 要预测图片的目录 --detect_save_dir 预测结果存储的目录 --class_file 类别标签（.txt）`



