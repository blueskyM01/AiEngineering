# zpmc_yolact
## Environment setup (x86)
- Docker pull
    - ` $ sudo docker pull nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04`
- 启动镜像
    - `$ sudo docker run --name yoloact -itd -v /home/ntueee/yangjianbing:/root/code -p 2003:22 -e NVIDIA_DRIVER_CAPABILITIES=compute,utility --gpus all --shm-size="12g" --restart=always nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04`
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
    - cd 02-Segmetation/yolact && pip install -r requirements.txt


## 一、数据、环境准备
- 数据集准备（coco格式）：







- 找到[data/config.py](data/config.py)文件，修改```line175处```，如下：
```
# 修改这里，使用自己的数据集
ZPMC_CORNERLINE_SEG_CLASSES = ('cornerline') # 你数据集中所有的label标签，如xxx_CLASSES = ('label1', 'label2', ...)

ZPMC_CORNERLINE_LABEL_MAP = {1:1}  # 对label进行映射。原因：假设'label1': 1, 'label2': 2, 'label3': 4, 这时候，就要映射为：ZPMC_CORNERLINE_LABEL_MAP = {1:1, 2:2, 4:3} 

zpmc_cornerline_segmentation_dataset = dataset_base.copy({
    'name': 'zpmc cornerline seg 2022',

    'train_images': '/software/dataset/ZPMC_FirstLoading_Seg_Temp/train',
    'valid_images': '/software/dataset/ZPMC_FirstLoading_Seg_Temp/val',
    
    'train_info': '/software/dataset/ZPMC_FirstLoading_Seg_Temp/annotations/train.json',
    'valid_info': '/software/dataset/ZPMC_FirstLoading_Seg_Temp/annotations/val.json',

    'class_names': ZPMC_CORNERLINE_SEG_CLASSES,
    'label_map': ZPMC_CORNERLINE_LABEL_MAP
})
```
- 训练镜像拉取
```
$ sudo docker pull 10.128.231.44/train/zpmc_yolact-torch1.13:v1.0
```
- 容器启动
```
$ sudo docker run --name ocr -itd -v /mnt/sda:/software -p 1999:22  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility --gpus all --shm-size="12g" --restart=always 10.128.231.44/train/zpmc_yolact-torch1.13:v1.0
```

## 二、训练
- 训练指令
```
$ python train.py --config=yolact_base_config --dataset=zpmc_cornerline_segmentation_dataset --log_folder=logs --save_folder=weights
参数解释
--config=yolact_base_config，这个参数保持不变
--dataset=zpmc_cornerline_segmentation_dataset，这个参数保持不变
--log_folder，存储log日志的文件目录
--save_folder，保存网络权重的目录

加载预训练模型
$ python train.py --config=yolact_base_config --resume xxx/yolact_base_xxx.pth --dataset=zpmc_cornerline_segmentation_dataset --log_folder=logs --save_folder=weights
--resume: 预训练模型权重(注意：是当前训练生成的模型哦)
```
- 训练的权重会存在[weights](weights)文件夹中
## 三、评估
- 生成测试结果
```
python zpmc_eval.py --trained_model weights/yolact_base_45.pth --config yolact_base_config --dataset zpmc_cornerline_segmentation_dataset --detect_save_dir results --class_name_path /software/yangjianbing/dataset/ZPMC_FirstLoading_Seg_Temp/annotations/FirstLanding.name --val_label_path /software/yangjianbing/dataset/ZPMC_FirstLoading_Seg_Temp/annotations/val_detection_parser.txt --val_img_dir /software/yangjianbing/dataset/ZPMC_FirstLoading_Seg_Temp/val
参数解释
--trained_model，模型权重
--config yolact_base_config，这个参数保持不变
--dataset zpmc_cornerline_segmentation_dataset，这个参数保持不变
--detect_save_dir，会在该指定的目录下生成detection-results、ground-truth和visualizes文件夹，detection-results、ground-truth下存有“.txt”文件，visualizes保存10张可视化图片
--class_name_path xxxx.name, 数据集的类别名称文件路径
--val_label_path， 测试集的label文件路径
--val_img_dir，测试集的图像存储目录
```
- 评估测试结果
    
    ★ 下载[mAP](http://10.128.231.44:7080/yangjianbing/map)
    ```
    $ cd 你想存储mAP的目录
    $ git clone http://10.128.231.44:7080/yangjianbing/map.git
    ```
    ★ 进入mAP目录，并在下面新建**input**文件夹
    ```
    $ cd mAP
    $ mkdir input
    ```
    ★ 将生成测试结果拷贝到mAP/input文件夹下
    ```
    $ rm -rf /xxxx/mAP/input/*
    $ mv /xxxx/zpmc_yolact_pytorch2onnx/results(这个改成你设置的--detect_save_dir)/detection-results /xxxx/zpmc_yolact_pytorch2onnx/results(这个改成你设置的--detect_save_dir)/ground-truth /xxxx/mAP/input
    ```
    ★ 测试精度
    ```
    $ python main.py
    ```
## 四、onnx生成
- 指令
```
python zpmc_torch2onnx.py --trained_model weights/yolact_base_45.pth --config yolact_base_config --dataset zpmc_cornerline_segmentation_dataset --image test_images/image_0000002313.jpeg:test_images/image_0000002313-out.jpeg
```
- 会在```onnx```文件夹下生成```yolact.onnx```文件

## 五、云上训练
- 需要修改的地方
    1. 增加入口参数data_dir、train_dir；
    2. --log_folder 
    3. --save_folder
    4. 找到[data/config.py](data/config.py)文件，修改```line175处```下面的数据集与标注label路径
    5. 预训练的权重```weights/resnet101_reducedfc.pth```，拷贝到--save_folder所指定的目录下
