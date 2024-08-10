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
# from PIL import Image, ImageDraw, ImageFont



def get_classes(file_path):
    classes = []
    with open(file_path, 'r') as f:
        line = f.readline()
        while line:
            classes.append(line.rstrip('\n'))
            line = f.readline()
    return classes

def numpy_nms(boxes, scores, iou_threshold):
    """
    Perform Non-Maximum Suppression (NMS) using NumPy.

    Args:
    - boxes (np.ndarray): Array of bounding boxes with shape (N, 4), where N is the number of boxes.
                          Each box is represented as [x1, y1, x2, y2].
    - scores (np.ndarray): Array of scores with shape (N,) representing the confidence for each box.
    - iou_threshold (float): Intersection over Union (IoU) threshold for filtering.

    Returns:
    - List[int]: Indices of the boxes that are kept after NMS.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

def centernet_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
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

def postprocess(prediction, need_nms, image_shape, input_shape, letterbox_image, nms_thres=0.4):
    output = [None for _ in range(len(prediction))]
    
    #----------------------------------------------------------#
    #   预测只用一张图片，只会进行一次
    #----------------------------------------------------------#
    for i, image_pred in enumerate(prediction):
        detections      = prediction[i]
        if len(detections) == 0:
            continue
        #------------------------------------------#
        #   获得预测结果中包含的所有种类
        #------------------------------------------#
        # unique_labels   = detections[:, -1].cpu().unique()
        unique_labels   = np.unique(detections[:, -1])

        # if detections.is_cuda:
        #     unique_labels = unique_labels.cuda()
        #     detections = detections.cuda()

        for c in unique_labels:
            #------------------------------------------#
            #   获得某一类得分筛选后全部的预测结果
            #------------------------------------------#
            detections_class = detections[detections[:, -1] == c]
            if need_nms:
                #------------------------------------------#
                #   使用官方自带的非极大抑制会速度更快一些！
                #------------------------------------------#
                keep = numpy_nms(
                    detections_class[:, :4],
                    detections_class[:, 4],
                    nms_thres
                )
                max_detections = detections_class[keep]

                # #------------------------------------------#
                # #   按照存在物体的置信度排序
                # #------------------------------------------#
                # _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
                # detections_class = detections_class[conf_sort_index]
                # #------------------------------------------#
                # #   进行非极大抑制
                # #------------------------------------------#
                # max_detections = []
                # while detections_class.size(0):
                #     #---------------------------------------------------#
                #     #   取出这一类置信度最高的，一步一步往下判断。
                #     #   判断重合程度是否大于nms_thres，如果是则去除掉
                #     #---------------------------------------------------#
                #     max_detections.append(detections_class[0].unsqueeze(0))
                #     if len(detections_class) == 1:
                #         break
                #     ious = bbox_iou(max_detections[-1], detections_class[1:])
                #     detections_class = detections_class[1:][ious < nms_thres]
                # #------------------------------------------#
                # #   堆叠
                # #------------------------------------------#
                # max_detections = torch.cat(max_detections).data
            else:
                max_detections  = detections_class

            
            # output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))
            output[i] = max_detections if output[i] is None else np.concatenate((output[i], max_detections))

        if output[i] is not None:
            output[i]           = output[i]
            box_xy, box_wh      = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
            output[i][:, :4]    = centernet_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
    return output

def decode_bbox(pred_hms, pred_whs, pred_offsets, confidence, cuda):
    #-------------------------------------------------------------------------#
    #   当利用512x512x3图片进行coco数据集预测的时候
    #   h = w = 128 num_classes = 80
    #   Hot map热力图 -> b, 80, 128, 128, 
    #   进行热力图的非极大抑制，利用3x3的卷积对热力图进行最大值筛选
    #   找出一定区域内，得分最大的特征点。
    #-------------------------------------------------------------------------#
    # pred_hms = pool_nms(pred_hms)
    
    b, c, output_h, output_w = pred_hms.shape
    detects = []
    #-------------------------------------------------------------------------#
    #   只传入一张图片，循环只进行一次
    #-------------------------------------------------------------------------#
    for batch in range(b):
        #-------------------------------------------------------------------------#
        #   heat_map        128*128, num_classes    热力图
        #   pred_wh         128*128, 2              特征点的预测宽高
        #                                           在预测过程的前处理以及后处理视频中讲的有点小问题，不是调整参数，就是宽高
        #   pred_offset     128*128, 2              特征点的xy轴偏移情况
        #-------------------------------------------------------------------------#
        # heat_map    = pred_hms[batch].permute(1, 2, 0).view([-1, c])
        # pred_wh     = pred_whs[batch].permute(1, 2, 0).view([-1, 2])
        # pred_offset = pred_offsets[batch].permute(1, 2, 0).view([-1, 2])
        
        heat_map    = pred_hms[batch].transpose(1, 2, 0).reshape([-1, c])
        pred_wh     = pred_whs[batch].transpose(1, 2, 0).reshape([-1, 2])
        pred_offset = pred_offsets[batch].transpose(1, 2, 0).reshape([-1, 2])

        # yv, xv      = torch.meshgrid(torch.arange(0, output_h), torch.arange(0, output_w))
        xv, yv       = np.meshgrid(np.arange(0, output_h), np.arange(0, output_w))
        #-------------------------------------------------------------------------#
        #   xv              128*128,    特征点的x轴坐标
        #   yv              128*128,    特征点的y轴坐标
        #-------------------------------------------------------------------------#
        # xv, yv      = xv.flatten().float(), yv.flatten().float()
        xv, yv      = xv.flatten().astype(np.float32), yv.flatten().astype(np.float32)
        # if cuda:
        #     xv      = xv.cuda()
        #     yv      = yv.cuda()

        #-------------------------------------------------------------------------#
        #   class_conf      128*128,    特征点的种类置信度
        #   class_pred      128*128,    特征点的种类
        #-------------------------------------------------------------------------#
        # class_conf, class_pred  = torch.max(heat_map, dim = -1)
        class_conf  = np.max(heat_map, axis = -1)
        class_pred = np.argmax(heat_map, axis = -1)
        mask                    = class_conf > confidence

        #-----------------------------------------#
        #   取出得分筛选后对应的结果
        #-----------------------------------------#
        pred_wh_mask        = pred_wh[mask]
        pred_offset_mask    = pred_offset[mask]
        if len(pred_wh_mask) == 0:
            detects.append([])
            continue     

        #----------------------------------------#
        #   计算调整后预测框的中心
        #----------------------------------------#
        # xv_mask = torch.unsqueeze(xv[mask] + pred_offset_mask[..., 0], -1)
        # yv_mask = torch.unsqueeze(yv[mask] + pred_offset_mask[..., 1], -1)
        xv_mask = np.expand_dims(xv[mask] + pred_offset_mask[..., 0], -1)
        yv_mask = np.expand_dims(yv[mask] + pred_offset_mask[..., 1], -1)
        #----------------------------------------#
        #   计算预测框的宽高
        #----------------------------------------#
        half_w, half_h = pred_wh_mask[..., 0:1] / 2, pred_wh_mask[..., 1:2] / 2
        #----------------------------------------#
        #   获得预测框的左上角和右下角
        #----------------------------------------#
        # bboxes = torch.cat([xv_mask - half_w, yv_mask - half_h, xv_mask + half_w, yv_mask + half_h], dim=1)
        bboxes = np.concatenate([xv_mask - half_w, yv_mask - half_h, xv_mask + half_w, yv_mask + half_h], axis=1)
        bboxes[:, [0, 2]] /= output_w
        bboxes[:, [1, 3]] /= output_h
        # detect = torch.cat([bboxes, torch.unsqueeze(class_conf[mask],-1), torch.unsqueeze(class_pred[mask],-1).float()], dim=-1)
        detect = np.concatenate([bboxes, np.expand_dims(class_conf[mask],-1), np.expand_dims(class_pred[mask],-1).astype(np.float32)], axis=-1)
        detects.append(detect)

    return detects



def merge_image(images):
    img_tl, img_tr, img_bl, img_br = images
    heigt = img_tl.shape[0]
    width = img_tl.shape[1]
    c = img_tl.shape[2]
    merge_img = np.ones([2*heigt, 2*width,c], img_tl.dtype)
    merge_img[:heigt, :width, :] = img_tl
    merge_img[:heigt, width: 2*width, :] = img_tr
    merge_img[heigt: 2*heigt, :width, :] = img_bl
    merge_img[heigt: 2*heigt, width:2*width, :] = img_br
    return merge_img
    

def zpmc_onnx2trt(onnxFile, trtFile_save_dir, trtFile_save_name, FPMode, images_dir, detect_save_dir, class_names):
    if not os.path.exists(detect_save_dir):
        os.makedirs(detect_save_dir)
    input_shape = [512, 512]
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


    print('inputTensor:', inputTensor.name, inputTensor.shape)
    print('outputTensor0:', outputTensor0.name, outputTensor0.shape)
    print('outputTensor1:', outputTensor1.name, outputTensor1.shape)
    print('outputTensor2:', outputTensor2.name, outputTensor2.shape)


    batch_size, nHeight, nWidth, _  = inputTensor.shape
    profile.set_shape(inputTensor.name, (1, nHeight, nWidth, 3), (1, nHeight, nWidth, 3), (1, nHeight, nWidth, 3)) # 最小batch，常见batch，最大batch
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
        print(counter, ':', image_name)
        image = cv2.imread(os.path.join(images_dir, image_name))
        image_shape = image.shape[0:2]
        starttime = datetime.datetime.now()  
        
        image_data = cv2.resize(image, (input_shape[1], input_shape[0]))
        image_data = np.expand_dims(image_data, 0)
        inputs[0].host = np.ascontiguousarray(image_data, dtype=np.float32)
        trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

        trt_outputs_shape = [(batch_size, 2, 128, 128), (batch_size, 2, 128, 128), (batch_size, 4, 128, 128)]
        offset = trt_outputs[1].reshape(trt_outputs_shape[0])
        p_w_h = trt_outputs[0].reshape(trt_outputs_shape[1])
        heat_map= trt_outputs[2].reshape(trt_outputs_shape[2])
        # print('heat_map: {}'.format(heat_map.shape))
        # print('p_w_h: {}'.format(p_w_h.shape))
        # print('offset: {}'.format(offset.shape))

        outputs_ = decode_bbox(heat_map, p_w_h, offset, 0.3, False)
        
        results = postprocess(outputs_, False, image_shape, input_shape, False, 0.4)
                
        #--------------------------------------#
        #   如果没有检测到物体，则返回原图
        #--------------------------------------#
        if results[0] is None:
            print('no target in this frame!')
            continue
        
        top_label   = np.array(results[0][:, 5], dtype = 'int32')
        top_conf    = results[0][:, 4]
        top_boxes   = results[0][:, :4]

        endtime = datetime.datetime.now()
        timediff = (endtime - starttime).total_seconds()
        print(1/timediff)
        src = image

        colors = [(0,0,255), (255,0,0), (0,0,0), (0, 255, 255)]     #红（人脸），蓝（四旋翼）， 黑(飞机)，黄（降落伞）    
        for i, c in list(enumerate(top_label)):
            predicted_class = class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top))
            left    = max(0, np.floor(left))
            bottom  = min(image.shape[0], np.floor(bottom))
            right   = min(image.shape[1], np.floor(right))
            cv2.rectangle(src, (int(left), int(top)), (int(right), int(bottom)), colors[int(c)], 2)
            cv2.imwrite(os.path.join(detect_save_dir, 'trt_'+image_name.split('/')[-1]), src)
            # print('/root/code/AiEngineering/01-ObjectDetection/CenterNet/code/img_out/trt_'+image_name.split('/')[-1])
    

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN')
    parser.add_argument('--onnxFile', default='/root/code/AiEngineering/01-ObjectDetection/CenterNet/code/model_data/detection_models.onnx', type=str, help='')
    parser.add_argument('--trtFile_save_dir', default='/root/code/AiEngineering/01-ObjectDetection/CenterNet/code/trt', type=str, help='')
    parser.add_argument('--trtFile_save_name', default='centernet_detection16.trt', type=str, help='')
    parser.add_argument('--FPMode', default='FP16', type=str, help='')
    parser.add_argument('--images_dir', default='/root/code/dataset/610/show/', type=str, help='')
    parser.add_argument('--detect_save_dir', default='/root/code/AiEngineering/01-ObjectDetection/CenterNet/code/img_out', type=str, help='')
    parser.add_argument('--class_file', default='/root/code/dataset/610/dataset_610_v1/dataset_610_v1/class.txt', type=str, help='')
    args = parser.parse_args()
    class_names = get_classes(args.class_file)
    zpmc_onnx2trt(onnxFile=args.onnxFile, trtFile_save_dir=args.trtFile_save_dir, trtFile_save_name=args.trtFile_save_name, FPMode=args.FPMode, 
                  images_dir=args.images_dir, detect_save_dir=args.detect_save_dir, class_names=class_names)