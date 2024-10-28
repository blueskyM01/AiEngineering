from cuda import cudart
import cv2
from datetime import datetime as dt
from glob import glob
import numpy as np
import os, sys
import tensorrt as trt
sys.path.append('/root/TensorRT-8.4.3.1/samples/python') 
import common
import datetime
import cv2
import argparse
from models.post_processing import pse

font = cv2.FONT_HERSHEY_SIMPLEX
def scale_aligned_short(img, short_size=736):
    h, w = img.shape[0:2]
    scale = short_size * 1.0 / min(h, w)
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img

def normalize(image, mean, std):
    # 将图像从 0-255 转换到 0-1 的范围
    image = image / 255.0
    # 使用广播机制进行归一化
    image = (image - mean) / std
    return image

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_results(out, img_meta, kernel_num, min_area, min_score):
    score = sigmoid(out[:, 0, :, :])

    kernels = out[:, :kernel_num, :, :] > 0
    text_mask = kernels[:, :1, :, :]
    kernels[:, 1:, :, :] = kernels[:, 1:, :, :] * text_mask

    score = score[0].astype(np.float32)
    kernels = kernels[0].astype(np.uint8)
    

    label = pse(kernels, min_area)

    # image size
    org_img_size = img_meta['org_img_size'][0]
    img_size = img_meta['img_size'][0]
    
    label_num = np.max(label) + 1
    label = cv2.resize(label, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)
    score = cv2.resize(score, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)



    scale = (float(org_img_size[1]) / float(img_size[1]),
                float(org_img_size[0]) / float(img_size[0]))

    bboxes = []
    scores = []
    for i in range(1, label_num):
        ind = label == i
        points = np.array(np.where(ind)).transpose((1, 0))

        if points.shape[0] < min_area:
            label[ind] = 0
            continue

        score_i = np.mean(score[ind])
        if score_i < min_score:
            label[ind] = 0
            continue

        
        binary = np.zeros(label.shape, dtype='uint8')
        binary[ind] = 1
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bbox = contours[0] * scale

        bbox = bbox.astype('int32')
        bboxes.append(bbox.reshape(-1))
        scores.append(score_i)
    return bboxes, scores

def zpmc_onnx2trt(onnxFile, trtFile_save_dir, trtFile_save_name, FPMode, images_dir, detect_save_dir):
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 << 30)

    if FPMode == 'FP32':
        print('Set FPMode = FP32!')
    elif FPMode == 'FP16':
        config.set_flag(trt.BuilderFlag.FP16)
        print('Set FPMode = FP16!')
    else:
        print('Please set FPMode = FP32 or FP16')
        exit()

    parser = trt.OnnxParser(network, logger)
    
    if not os.path.exists(detect_save_dir):
        os.mkdir(detect_save_dir)

    if not os.path.exists(onnxFile):
        print("Failed finding {}!".format(onnxFile))
        exit()
    print("Succeeded finding {}!".format(onnxFile))

    with open(onnxFile, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing {}!".format(onnxFile))
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing {}!".format(onnxFile))

    inputTensor = network.get_input(0)
    outputTensor0 = network.get_output(0)
    print('inputTensor:', inputTensor.name, inputTensor.shape)
    print('outputTensor0:', outputTensor0.name, outputTensor0.shape)



    batch_size, channel, nHeight, nWidth = inputTensor.shape
    profile.set_shape(inputTensor.name, (1, channel, nHeight, nWidth), (1, channel, nHeight, nWidth), (1, channel, nHeight, nWidth)) # 最小batch，常见batch，最大batch
    config.add_optimization_profile(profile)

    trtFile = os.path.join(trtFile_save_dir, trtFile_save_name)
    if not os.path.exists(trtFile):
        engineString = builder.build_serialized_network(network, config)
        if engineString == None:
            print("Failed building engine!")
            exit()
        print("Succeeded building engine!")

        if not os.path.exists(trtFile_save_dir):  # os模块判断并创建
            os.mkdir(trtFile_save_dir)
        with open(trtFile, "wb") as f:
            f.write(engineString)
        print("Generate {}!".format(trtFile))
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
        print('Loaded {}!'.format(trtFile))
    else:
         with open(trtFile, "rb") as f, trt.Runtime(logger) as runtime:
            engine =  runtime.deserialize_cuda_engine(f.read())
         print('Loaded {}!'.format(trtFile))


    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)

    image_names = os.listdir(images_dir)
    
    for image_name in image_names:
        # 读取图像
        image_path = os.path.join(images_dir, image_name)
        img_org = cv2.imread(image_path)
        img = img_org[:, :, [2, 1, 0]]
        img_meta = dict(
                org_img_size=np.array(img.shape[:2]).reshape(-1, 2)
            )
        img = scale_aligned_short(img, short_size=736)
        img_meta.update(dict(
                img_size=np.array(img.shape[:2]).reshape(-1, 2)
            ))

        img = normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1))
        imgs = img[np.newaxis, :, :, :]
        
        inputs[0].host = np.ascontiguousarray(imgs, dtype=np.float32)
        trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        # print(trt_outputs[0].shape)
        if_out_ = trt_outputs[0].reshape(outputTensor0.shape)
        bboxes, scores = get_results(out=if_out_, 
                                    img_meta=img_meta, 
                                    kernel_num=7, 
                                    min_area=16, 
                                    min_score=0.85)
        contours = []
        for score, bbox in zip(scores, bboxes):
            contour = []
            for i in range(0, len(bbox), 2):
                point = [int(bbox[i]), int(bbox[i+1])]
                contour.append(point)

            cv2.putText(img_org, str(score), (int(bbox[0]), int(bbox[1])-5), font, 0.5, (255, 255, 255), 2)
            
            contours.append(np.array(contour))
        cv2.drawContours(img_org, contours, -1, color=(255,0,0), thickness=2)
        cv2.imwrite(os.path.join(detect_save_dir, image_name), img_org)    
        print('save ', os.path.join(detect_save_dir, image_name))



    

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN')
    parser.add_argument('--onnxFile', default='onnx/psenet.onnx', type=str, help='')
    parser.add_argument('--trtFile_save_dir', default='trt', type=str, help='')
    parser.add_argument('--trtFile_save_name', default='psenet16.trt', type=str, help='')
    parser.add_argument('--FPMode', default='FP16', type=str, help='')
    parser.add_argument('--images_dir', default='/root/code/dataset/containercode/images/val', type=str, help='')
    parser.add_argument('--detect_save_dir', default='result', type=str, help='')
    args = parser.parse_args()

    zpmc_onnx2trt(onnxFile=args.onnxFile, trtFile_save_dir=args.trtFile_save_dir, trtFile_save_name=args.trtFile_save_name, FPMode=args.FPMode, 
                  images_dir=args.images_dir, detect_save_dir=args.detect_save_dir)