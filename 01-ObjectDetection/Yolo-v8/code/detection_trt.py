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

def non_max_suppression(prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
    #----------------------------------------------------------#
    #   Convert the format of the prediction to top-left and bottom-right corners.
    #   prediction  [batch_size, num_anchors, 85]
    #----------------------------------------------------------#
    box_corner = np.zeros_like(prediction)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        #----------------------------------------------------------#
        #   Get the maximum of class predictions.
        #   class_conf  [num_anchors, 1]    Class confidence
        #   class_pred  [num_anchors, 1]    Class
        #----------------------------------------------------------#
        class_conf = np.max(image_pred[:, 4:4 + num_classes], axis=1, keepdims=True)
        class_pred = np.argmax(image_pred[:, 4:4 + num_classes], axis=1, keepdims=True)

        #----------------------------------------------------------#
        #   Filter by confidence threshold
        #----------------------------------------------------------#
        conf_mask = (class_conf[:, 0] >= conf_thres)
        
        #----------------------------------------------------------#
        #   Filter predictions based on the confidence threshold
        #----------------------------------------------------------#
        image_pred = image_pred[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]
        if not image_pred.shape[0]:
            continue
        
        #-------------------------------------------------------------------------#
        #   detections  [num_anchors, 6]
        #   6 columns: x1, y1, x2, y2, class_conf, class_pred
        #-------------------------------------------------------------------------#
        detections = np.concatenate((image_pred[:, :4], class_conf, class_pred), axis=1)

        #------------------------------------------#
        #   Get all the unique classes from predictions
        #------------------------------------------#
        unique_labels = np.unique(detections[:, -1])

        for c in unique_labels:
            #------------------------------------------#
            #   Get all the predictions of a single class
            #------------------------------------------#
            detections_class = detections[detections[:, -1] == c]

            #------------------------------------------#
            #   Sort by confidence
            #------------------------------------------#
            detections_class = detections_class[detections_class[:, 4].argsort()[::-1]]
            
            # Non-Maximum Suppression (NMS)
            max_detections = []
            while len(detections_class) > 0:
                max_detections.append(detections_class[0])
                if len(detections_class) == 1:
                    break

                # Calculate IoUs
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                detections_class = detections_class[1:][ious < nms_thres]

            # Stack max detections
            max_detections = np.stack(max_detections)
            
            # Add max detections to outputs
            output[i] = max_detections if output[i] is None else np.concatenate((output[i], max_detections))
        
        if output[i] is not None:
            box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
            output[i][:, :4] = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
    return output

def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    #-----------------------------------------------------------------#
    #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
    #-----------------------------------------------------------------#
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)

    if letterbox_image:
        #-----------------------------------------------------------------#
        #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
        #   new_shape指的是宽高缩放情况
        #-----------------------------------------------------------------#
        new_shape = np.round(image_shape * np.min(input_shape/image_shape))
        offset  = (input_shape - new_shape)/2./input_shape
        scale   = input_shape/new_shape

        box_yx  = (box_yx - offset) * scale
        box_hw *= scale

    box_mins    = box_yx - (box_hw / 2.)
    box_maxes   = box_yx + (box_hw / 2.)
    boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes

def bbox_iou(box1, boxes2):
    """
    Calculate IoU of a single box with multiple boxes.
    box1: [x1, y1, x2, y2]
    boxes2: [[x1, y1, x2, y2], ...]
    """
    # Calculate intersection area
    inter_x1 = np.maximum(box1[0], boxes2[:, 0])
    inter_y1 = np.maximum(box1[1], boxes2[:, 1])
    inter_x2 = np.minimum(box1[2], boxes2[:, 2])
    inter_y2 = np.minimum(box1[3], boxes2[:, 3])
    
    inter_area = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = box1_area + boxes2_area - inter_area

    return inter_area / union_area

def get_classes(file_path):
    classes = []
    with open(file_path, 'r') as f:
        line = f.readline()
        while line:
            classes.append(line.rstrip('\n'))
            line = f.readline()
    return classes

def dist2bbox(distance, anchor_points, xywh=True, axis=-1):
    """Transform distance(ltrb) to box(xywh or xyxy) using numpy."""
    # Split distance into left-top and right-bottom parts
    lt, rb = np.split(distance, 2, axis)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return np.concatenate((c_xy, wh), axis)  # xywh bbox
    return np.concatenate((x1y1, x2y2), axis)  # xyxy bbox

def resize_image(image, size, letterbox_image):
    iw, ih = image.shape[1], image.shape[0]  # OpenCV image shape: (height, width, channels)
    w, h = size
    
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        
        # Resize the image using OpenCV
        image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
        
        # Create a new image with a gray background
        new_image = np.full((h, w, 3), 128, dtype=np.uint8)
        
        # Paste the resized image into the center of the new image
        top = (h - nh) // 2
        left = (w - nw) // 2
        new_image[top:top + nh, left:left + nw] = image_resized
    else:
        # Resize the image directly to the target size
        new_image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
    
    return new_image

def preprocess_input(image):
    image /= 255.0
    return image

def decode_box(dbox, cls, origin_cls, anchors, strides, input_shape):
    # dbox  batch_size, 4, 8400
    # cls   batch_size, 20, 8400

    # Get the center, width, and height coordinates
    anchors = np.expand_dims(anchors, axis=0)  # Equivalent to unsqueeze in PyTorch
    dbox = dist2bbox(dbox, anchors, xywh=True, axis=1) * strides
    
    # Concatenate dbox and cls (sigmoid activation)
    y = np.concatenate((dbox, 1 / (1 + np.exp(-cls))), axis=1)  # Sigmoid using NumPy
    
    # Transpose the array to match the shape (batch_size, 8400, 24)
    y = np.transpose(y, (0, 2, 1))
    
    # Normalize to the range 0~1
    input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
    y[:, :, :4] = y[:, :, :4] / input_shape
    
    return y


    

def zpmc_onnx2trt(onnxFile, trtFile_save_dir, trtFile_save_name, FPMode, images_dir, detect_save_dir, class_names):
    if not os.path.exists(detect_save_dir):
        os.makedirs(detect_save_dir)
    input_shape = [640, 640]
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
    outputTensor1 = network.get_output(1)
    outputTensor2 = network.get_output(2)
    outputTensor3 = network.get_output(3)
    outputTensor4 = network.get_output(4)
    outputTensor5 = network.get_output(5)
    outputTensor6 = network.get_output(6)



    print('inputTensor:', inputTensor.name, inputTensor.shape)
    print('outputTensor0:', outputTensor0.name, outputTensor0.shape) # dbox
    print('outputTensor1:', outputTensor1.name, outputTensor1.shape) # cls
    print('outputTensor2:', outputTensor2.name, outputTensor2.shape) # P3
    print('outputTensor3:', outputTensor3.name, outputTensor3.shape) # P4
    print('outputTensor4:', outputTensor4.name, outputTensor4.shape) # P5
    print('outputTensor5:', outputTensor5.name, outputTensor5.shape) # anchors
    print('outputTensor6:', outputTensor6.name, outputTensor6.shape) # strides



    batch_size, _, nHeight, nWidth  = inputTensor.shape
    profile.set_shape(inputTensor.name, (batch_size, _, nHeight, nWidth), (batch_size, _, nHeight, nWidth), (batch_size, _, nHeight, nWidth)) # 最小batch，常见batch，最大batch
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
    
    
    images_list = os.listdir(images_dir)
    print(len(images_list))
    counter = 0
    for image_name in images_list:
        counter += 1
        # print(counter, ':', image_name)
        src = cv2.imread(os.path.join(images_dir, image_name))
        image_shape = src.shape[0:2]
        starttime = datetime.datetime.now()  
        image = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        image_data = resize_image(image, input_shape, True)
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        
    #     image_data = cv2.resize(image, (input_shape[1], input_shape[0]))
    #     image_data = np.expand_dims(image_data, 0)
        inputs[0].host = np.ascontiguousarray(image_data, dtype=np.float32)
        trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)


        P3_ = trt_outputs[0].reshape(outputTensor2.shape) # (batch_size, x, 80, 80)
        P4_ = trt_outputs[1].reshape(outputTensor3.shape) # (batch_size, x, 40, 40)
        P5_ = trt_outputs[2].reshape(outputTensor4.shape) # (batch_size, x, 20, 20)
        classes = trt_outputs[3].reshape(outputTensor1.shape)
        dbox = trt_outputs[4].reshape(outputTensor0.shape)
        anchors = trt_outputs[5].reshape(outputTensor5.shape)
        strides = trt_outputs[6].reshape(outputTensor6.shape)
        # print('dbox: {}'.format(dbox.shape))
        # print('classes: {}'.format(classes.shape))
        # print('P3: {}'.format(P3_.shape))
        # print('P4: {}'.format(P4_.shape))
        # print('P5: {}'.format(P5_.shape))
        # print('anchors: {}'.format(anchors.shape))
        # print('strides: {}'.format(strides.shape))

        outputs_ = decode_box(dbox, classes, [P3_, P4_, P5_], anchors, strides, input_shape)
        results = non_max_suppression(outputs_, len(class_names), input_shape, 
                        image_shape, True, conf_thres = 0.7, nms_thres = 0.3)
        
        if results[0] is None: 
            cv2.imwrite(os.path.join(detect_save_dir, 'trt_'+image_name.split('/')[-1]), src)
            continue

        top_label   = np.array(results[0][:, 5], dtype = 'int32')
        top_conf    = results[0][:, 4]
        top_boxes   = results[0][:, :4]
        
        colors = [(0,0,255), (255,0,0), (0,0,0), (0, 255, 255)]     #红（人脸），蓝（四旋翼）， 黑(飞机)，黄（降落伞）  
        for i, c in list(enumerate(top_label)):
            predicted_class = class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image_shape[0], np.floor(bottom).astype('int32'))
            right   = min(image_shape[1], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            # print(label, top, left, bottom, right)
            
            cv2.rectangle(src, (int(left), int(top)), (int(right), int(bottom)), colors[int(c)], 2)
        cv2.imwrite(os.path.join(detect_save_dir, 'trt_'+image_name.split('/')[-1]), src)
        print('write ', 'trt_'+image_name.split('/')[-1])
    

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='YoloV8')
    parser.add_argument('--onnxFile', default='/root/code/AiEngineering/01-ObjectDetection/Yolo-v8/code/model_data/yolo_detection_models.onnx', type=str, help='')
    parser.add_argument('--trtFile_save_dir', default='/root/code/AiEngineering/01-ObjectDetection/Yolo-v8/code/trt', type=str, help='')
    parser.add_argument('--trtFile_save_name', default='yolov8_detection16.trt', type=str, help='')
    parser.add_argument('--FPMode', default='FP16', type=str, help='')
    parser.add_argument('--images_dir', default='/root/code/dataset/610/show/', type=str, help='')
    parser.add_argument('--detect_save_dir', default='/root/code/AiEngineering/01-ObjectDetection/Yolo-v8/code/datas/img_out', type=str, help='')
    parser.add_argument('--class_file', default='/root/code/AiEngineering/01-ObjectDetection/Yolo-v8/code/datas/610_classes.txt', type=str, help='')
    args = parser.parse_args()
    class_names = get_classes(args.class_file)
    zpmc_onnx2trt(onnxFile=args.onnxFile, trtFile_save_dir=args.trtFile_save_dir, trtFile_save_name=args.trtFile_save_name, FPMode=args.FPMode, 
                  images_dir=args.images_dir, detect_save_dir=args.detect_save_dir, class_names=class_names)