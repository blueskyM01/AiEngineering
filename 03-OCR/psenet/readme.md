## 一、Environment setup (x86)
- Docker pull
    - ` $ sudo docker pull nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04`
- 启动镜像
    - `$ sudo docker run --name psenet -itd  -v /home/ntueee/yangjianbing:/root/code -p 2016:22 -e NVIDIA_DRIVER_CAPABILITIES=compute,utility --gpus all --shm-size="12g" --restart=always nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04`
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

## 二、数据集
### 2.1 集装箱箱号数据集下载
    [地址]()
### 2.2 预处理
- 代码下载
```
$ git clone http://10.128.231.44:7080/yangjianbing/image_ann_preprocess_tool.git
```

- coco数据格式的标注转为训练label
```
$ cd Image_ANN_Preprocess_Tool # 进入该project的目录
$ python ocr_dataset/container_number/zpmc_convert_detect.py --ann_dir 标注文件的存储目录 --ann_name 标注文件的名称 --label_save_dir 生成训练label的存储目录 --label_save_name 生成训练label的名称 --class_names 标注中的label名称的存储名称
```

## 三、训练
### 3.1 数据集
- 编译
    ```
    cd psenet
    chmod 777 compile.sh
    ./compile.sh
    ```
- 找到```dataset/psenet/psenet_ctw.py```文件
- 修改```line:17-20```，改成上面预处理得到的label路径
    ```
    ctw_train_data_dir = '/root/code/yangjianbing/dataset/ZPMC_Container_Number/detect/images/train'
    ctw_train_gt_dir = '/root/code/yangjianbing/dataset/ZPMC_Container_Number/detect/annotations/train.json'
    ctw_test_data_dir = '/root/code/yangjianbing/dataset/ZPMC_Container_Number/detect/images/val'
    ctw_test_gt_dir = '/root/code/yangjianbing/dataset/ZPMC_Container_Number/detect/annotations/val.json'
    ```

### 3.2 模型训练
- 在`psenet`下新建`pretrained`文件夹
- 下载[预训练模型](https://pan.baidu.com/s/1kDD5nuvkOL-tkuvrafytKA )(密码1234)，并拷贝到`pretrained`文件夹
- 训练指令
    ```
    $ python train.py # 预训练模型在models/backbone/resnet.py中的line211处加载参数   # 或
    $ python train.py --config config/psenet/psenet_r50_ctw.py --checkpoint checkpoints --resume xxx.pth.tar
    --config 参数配置文件
    --checkpoint 模型参数的存储目录
    --resume 加载之前训练好的模型
    ```
- 若出现`AttributeError: module 'numpy' has no attribute 'int'.`报错，修改
    ```
    找到dataset/psenet/psenet_ctw.py，
    将line63处bbox = [np.int(gt[i]) for i in range(num_ele)]，改成bbox = [int(gt[i]) for i in range(num_ele)]
    ```
- 梯度不下降解决方法
    ```
    找到  
    /root/code/yangjianbing/code/zpmc_psenet_ocr/config/psenet/psenet_r50_ctw.py，
    line47 修改学习率
    ```
- 训练生产的模型保存在```checkpoints```文件夹中，如```checkpoint_epoch_0.pth.tar```

## 四、预测
- 预测指令
```
$ python zpmc_inference.py --checkpoint checkpoints/checkpoint_epoch_0.pth.tar --img_dir /root/code/yangjianbing/dataset/ZPMC_Container_Number/detect/images/val
```
- 参数解释
    - --checkpoint：保存的权重
    - --img_dir：验证集图像存储的目录，必须设置成与ctw_test_data_dir相同

- 预测的结果保存在```outputs/show```

## 五、评估
- 启动指令
```
$ python zpmc_eval.py --checkpoint checkpoints/checkpoint_epoch_0.pth.tar --img_dir /root/code/yangjianbing/dataset/ZPMC_Container_Number/detect/images/val --ann_path /root/code/yangjianbing/dataset/ZPMC_Container_Number/detect/annotations/val.json --result_save_dir ocr_detect_result
```

- 参数解释
    - --checkpoint：保存的权重
    - --img_dir：验证集图像存储的目录，必须设置成与ctw_test_data_dir相同
    - --ann_path: 验证集标签，必须设置成与ctw_test_gt_dir相同
    - --result_save_dir：预测结果存储路径

- 精度测试  
    - 测试代码下载
    ```
    $ git clone http://10.128.231.44:7080/yangjianbing/map.git
    ```
    - 将预测结果，传到测试代码中
    ```
    $ rm -rf /xxxx/mAP/input/* # 先清空测试代码中原有的文件
    $ mv /xxxx/ocr_detect_result/detection-results /xxxx/ocr_detect_result/ground-truth /xxxx/mAP/input
    ```
    - 精度预测指令
    ```
    $ python main.py
    ```
## 六、covert to onnx
- 找到`models/psenet.py`，将line87处的`# return det_out`取消注释（注意：导出onnx文件后，一定要把这一行再注释掉，否则`推理`和`精度评估`时会报错）
- `python zpmc_onnx.py --config config/psenet/psenet_r50_ctw.py --checkpoint checkpoints/checkpoint_epoch_0.pth.tar --report_speed False --input_image_path /root/code/dataset/containercode/images/val/image_0000002754.jpg`
    - --config(配置文件): config/psenet/psenet_r50_ctw.py
    - --checkpoint（训练出的模型）
    - --report_speed：必须设置成False
    - --input_image_path（预测图片的路径）
- 生成的`psenet.onnx`文件保存在`onnx文件夹`下

## 七、trt
- [环境配置](../../02-Segmetation/yolact_trt/README.MD)
    - 将`models`文件夹下的`post_processing`文件夹，拷贝(编译好了再拷贝，若trt运行的python版本跟编译时的环境不一样，则需要再trt环境下重新编译)
    - 找到`zpmc_trt.py`，line13处，这里导入的`post_processing`包就是上面拷贝的`post_processing`。（直接拷贝`post_processing`文件夹的话，导入包时去掉前面的`models.`）
- `cd 03-OCR/psenet`
- `python zpmc_trt.py --onnxFile onnx/psenet.onnx --trtFile_save_dir trt --trtFile_save_name psenet16.trt --FPMode FP16 --images_dir /root/code/dataset/containercode/images/val --detect_save_dir result`
    - --onnxFile: 导出的onnx文件存储路径
    - --trtFile_save_dir：编译生成的trt文件的存储目录
    - --trtFile_save_name：编译生成的trt文件的名称
    - --FPMode：精度（FP32，FP16）
    - --images_dir：待预测图片存储的目录
    - --detect_save_dir：预测结果存储的目录



## 八、云上训练
- 增加两个入口参数```data_dir```、```train_dir```
- train.py中的```num_workers```，设置为8
- config/psenet/psenet_r50_ctw.py中的```batch_size```设为14
- models/backbone/resnet.py中的resnet50加载参数，改成：
```
# model.load_state_dict(load_url(model_urls['resnet50']), strict=False)
model.load_state_dict(torch.load('/ai/007dfc08923c/code/zpmc_psenet_ocr/pretrained/resnet50-imagenet.pth'), strict=False) # 改成你自己的云上路径
```
- dataset/psenet/psenet_ctw.py，改
```
改成云上路径
ctw_train_data_dir = '/root/code/yangjianbing/dataset/ZPMC_Container_Number/detect/images/train'
ctw_train_gt_dir = '/root/code/yangjianbing/dataset/ZPMC_Container_Number/detect/annotations/train.json'
ctw_test_data_dir = '/root/code/yangjianbing/dataset/ZPMC_Container_Number/detect/images/val'
ctw_test_gt_dir = '/root/code/yangjianbing/dataset/ZPMC_Container_Number/detect/annotations/val.json'
```