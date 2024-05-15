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

def centernet_correct_boxes(box_xy, input_shape, image_shape, letterbox_image):
    #-----------------------------------------------------------------#
    #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
    #-----------------------------------------------------------------#
    box_yx = box_xy[..., ::-1]
    # box_hw = box_wh[..., ::-1]
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
        # box_hw *= scale

    # box_mins    = box_yx - (box_hw / 2.)
    # box_maxes   = box_yx + (box_hw / 2.)
    # boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
    boxes  = np.concatenate([box_yx[..., 0:1], box_yx[..., 1:2]], axis=-1)
    # boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    boxes *= np.concatenate([image_shape], axis=-1)
    return boxes

def postprocess(prediction, image_shape, input_shape, letterbox_image, nms_thres=0.4):
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
            
            max_detections  = detections_class
            
            # output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))
            output[i] = max_detections if output[i] is None else np.concatenate((output[i], max_detections))

        if output[i] is not None:
            output[i]           = output[i]
            # box_xy, box_wh      = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
            box_xy = output[i][:, 0:2]
            output[i][:, :2]    = centernet_correct_boxes(box_xy, input_shape, image_shape, letterbox_image)
    return output

def max_pool2d(input_array, kernel_size, stride=None, padding=0):
    batch_size, channels, height, width = input_array.shape
    kernel_size = kernel_size
    stride = stride if stride is not None else kernel_size

    # 计算输出特征图的尺寸
    out_height = (height - kernel_size + 2 * padding) // stride + 1
    out_width = (width - kernel_size + 2 * padding) // stride + 1

    # 使用零填充（padding）扩展输入数组
    input_padded = np.pad(input_array, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    # 初始化输出数组
    output = np.zeros((batch_size, channels, out_height, out_width))

    # 对每个像素位置进行最大池化
    for i in range(0, out_height, stride):
        for j in range(0, out_width, stride):
            # 选取当前窗口内的最大值
            window = input_padded[:, :, i:i+kernel_size, j:j+kernel_size]
            max_val = np.max(window, axis=(2, 3))
            output[:, :, i//stride, j//stride] = max_val

    return output

def pool_nms(heat, kernel = 3):
    pad = (kernel - 1) // 2

    hmax = max_pool2d(heat, kernel_size=kernel, stride=1, padding=1)
    # keep = (hmax == heat).float()
    keep = (hmax == heat).astype(np.float32)
    
    return heat * keep

def decode_bbox(pred_hms, pred_offsets, confidence, cuda):
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
        heat_map    = pred_hms[batch].transpose(1, 2, 0).reshape([-1, c])
        # pred_wh     = pred_whs[batch].permute(1, 2, 0).view([-1, 2])
        # pred_offset = pred_offsets[batch].permute(1, 2, 0).view([-1, 2])
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
        # pred_wh_mask        = pred_wh[mask]
        pred_offset_mask    = pred_offset[mask]
        # if len(pred_wh_mask) == 0:
        #     detects.append([])
        #     continue     

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
        # half_w, half_h = pred_wh_mask[..., 0:1] / 2, pred_wh_mask[..., 1:2] / 2
        #----------------------------------------#
        #   获得预测框的左上角和右下角
        #----------------------------------------#
        # bboxes = torch.cat([xv_mask - half_w, yv_mask - half_h, xv_mask + half_w, yv_mask + half_h], dim=1)
        # bboxes = torch.cat([xv_mask, yv_mask], dim=1)
        bboxes = np.concatenate([xv_mask, yv_mask], axis=1)
        bboxes[:, [0]] /= output_w
        bboxes[:, [1]] /= output_h
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

    print('inputTensor:', inputTensor.name, inputTensor.shape)
    print('outputTensor0:', outputTensor0.name, outputTensor0.shape)
    print('outputTensor1:', outputTensor1.name, outputTensor1.shape)


    batch_size, nHeight, nWidth, _  = inputTensor.shape
    profile.set_shape(inputTensor.name, (12, nHeight, nWidth, 3), (12, nHeight, nWidth, 3), (12, nHeight, nWidth, 3)) # 最小batch，常见batch，最大batch
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
    
    
    images_list = os.listdir(images_dir)[0:12]
    
    input_list = []
    input_src = []
    for image_name in images_list:
        image = cv2.imread(os.path.join(images_dir, image_name))
        image_shape = image.shape[0:2]        
        image_data = cv2.resize(image, (input_shape[1], input_shape[0]))
        input_list.append(image_data)
        input_src.append(image)
    
    starttime = datetime.datetime.now()      
    image_data = np.array(input_list)
    inputs[0].host = np.ascontiguousarray(image_data, dtype=np.float32)
    trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    trt_outputs_shape = [(batch_size, 2, 128, 128), (batch_size, 2, 128, 128)]
    heat_map = trt_outputs[1].reshape(trt_outputs_shape[0])
    p_w_h = trt_outputs[0].reshape(trt_outputs_shape[1])
    print('heat_map: {}'.format(heat_map.shape))
    print('p_w_h: {}'.format(p_w_h.shape))

    outputs_ = decode_bbox(heat_map, p_w_h, 0.3, False)
    
    results = postprocess(outputs_, image_shape, input_shape, False, 0.3)
    endtime = datetime.datetime.now()
    timediff = (endtime - starttime).total_seconds()
    print(1/timediff)
            
    # #--------------------------------------#
    # #   如果没有检测到物体，则返回原图
    # #--------------------------------------#
    # if results[0] is None:
    #     return image

    for idx in range(len(input_list)):
        src = input_src[idx]
        save_name = images_list[idx]
        print(save_name)
        if results[idx] is None:
            cv2.imwrite(os.path.join(detect_save_dir, save_name), src)
            continue
        top_label   = np.array(results[idx][:, 3], dtype = 'int32')
        top_conf    = results[idx][:, 2]
        top_boxes   = results[idx][:, :2]

        
        for i, c in list(enumerate(top_label)):
            predicted_class = class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]
            y_c, x_c= box
            
            x_c     = max(0, np.floor(x_c).astype('int32'))
            y_c    = max(0, np.floor(y_c).astype('int32'))
            # y_c  = min(image.size[1], np.floor(y_c).astype('int32'))
            # x_c   = min(image.size[0], np.floor(x_c).astype('int32'))
            y_c  = min(src.shape[0], np.floor(y_c).astype('int32'))
            x_c   = min(src.shape[1], np.floor(x_c).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            print(label, x_c, y_c)
            cv2.circle(src, (x_c, y_c), 4, (0,0,255), -1)
        cv2.imwrite(os.path.join(detect_save_dir, save_name), src)
        
    
    
    # mx_Capture_0 = cv2.VideoCapture(os.path.join(video_dir, 'cam1_000.avi'))
    # mx_Capture_1 = cv2.VideoCapture(os.path.join(video_dir, 'cam2_000.avi'))
    # mx_Capture_2 = cv2.VideoCapture(os.path.join(video_dir, 'cam3_000.avi'))
    # mx_Capture_3 = cv2.VideoCapture(os.path.join(video_dir, 'cam4_000.avi'))
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # size = (int(mx_Capture_0.get(cv2.CAP_PROP_FRAME_WIDTH) * 2), int(mx_Capture_0.get(cv2.CAP_PROP_FRAME_HEIGHT) * 2))
    # if not os.path.exists(detect_save_dir):  # os模块判断并创建
    #     os.mkdir(detect_save_dir)
    # merge_out = cv2.VideoWriter(os.path.join(detect_save_dir, FPMode+'.avi'), fourcc, 22, size)

    # counter = 0
    # while 1:
    #     mx_ret_0, image_src_0 = mx_Capture_0.read()
    #     mx_ret_1, image_src_1 = mx_Capture_1.read()
    #     mx_ret_2, image_src_2 = mx_Capture_2.read()
    #     mx_ret_3, image_src_3 = mx_Capture_3.read()

    #     if mx_ret_0 and mx_ret_1 and mx_ret_2 and mx_ret_3:
    #         h, w, _ = image_src_0.shape
    #         images = [image_src_0, image_src_1, image_src_2, image_src_3]
    #         images_resize_list = zpmc_ImagePreprocess(images, img_size=(550, 550))

    #         starttime = datetime.datetime.now()    

    #         inputs[0].host = np.ascontiguousarray(images_resize_list, dtype=np.float32)
    #         # cfx.push()
    #         trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    #         # cfx.pop()
    #         batch_size = len(images_resize_list)
    #         trt_outputs_shape = [(batch_size, 138, 138, 32), (batch_size, 19248, 4), (batch_size, 19248, 32), (batch_size, 19248, 2), (19248, 4)]
            
    #         proto_data = trt_outputs[0].reshape(trt_outputs_shape[0])
    #         loc_data = trt_outputs[1].reshape(trt_outputs_shape[1])
    #         mask_data = trt_outputs[2].reshape(trt_outputs_shape[2])
    #         conf_data = trt_outputs[3].reshape(trt_outputs_shape[3])
    #         prior_data = trt_outputs[4].reshape(trt_outputs_shape[4])
            
    #         CLASSES = ['BG', 'LOCKHOLE']
    #         result = zpmc_PostProcess(CLASSES, proto_data, prior_data, mask_data, conf_data, loc_data, conf_thresh=0.05)
    #         classes, scores, boxes, masks = zpmc_display(result, images_resize_list, h, w, score_threshold=0.15)

    #         endtime = datetime.datetime.now()
    #         timediff = (endtime - starttime).total_seconds()
    #         counter+=1
    #         print(timediff, '    ', 1/timediff, '   ', counter)

    #         merge_images = []
    #         for nclass, score, box, mask, image in zip(classes, scores, boxes, masks, images):    
    #             mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    #             mask = np.where(mask > 80, 255, 0).astype(np.uint8)
    #             visualize_image = daw_mask(image, mask)
    #             for lable, sc, bb in zip(nclass, score, box):
    #                 caption = CLASSES[lable+1]
    #                 x0 = int(bb[0])
    #                 y0 = int(bb[1])
    #                 x1 = int(bb[2])
    #                 y1 = int(bb[3])
    #                 aa = [x0, y0, x1, y1]
    #                 visualize_image = draw_caption(visualize_image, aa, caption, sc)  
    #             merge_images.append(visualize_image)
    #         merge_img = merge_image(merge_images)
    #         merge_out.write(merge_img)
    #     else:
    #         mx_Capture_0.release()
    #         mx_Capture_1.release()
    #         mx_Capture_2.release()
    #         mx_Capture_3.release()
    #         merge_out.release()
    #         break
    #     #     cv2.imwrite('results/' + str(counter) + '.jpg', visualize_image)
    #     #     counter += 1

    

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN')
    parser.add_argument('--onnxFile', default='/root/code/AiEngineering/01-ObjectDetection/CenterNet/code/model_data/models.onnx', type=str, help='')
    parser.add_argument('--trtFile_save_dir', default='/root/code/AiEngineering/01-ObjectDetection/CenterNet/code/trt', type=str, help='')
    parser.add_argument('--trtFile_save_name', default='centernet_keypoints16.trt', type=str, help='')
    parser.add_argument('--FPMode', default='FP16', type=str, help='')
    parser.add_argument('--images_dir', default='/root/code/dataset/crop_cell_guide/images/val_v1/', type=str, help='')
    parser.add_argument('--detect_save_dir', default='/root/code/AiEngineering/01-ObjectDetection/CenterNet/code/img_out', type=str, help='')
    parser.add_argument('--class_file', default='/root/code/dataset/crop_cell_guide/annotation/cell_guide.txt', type=str, help='')
    args = parser.parse_args()
    class_names = get_classes(args.class_file)
    zpmc_onnx2trt(onnxFile=args.onnxFile, trtFile_save_dir=args.trtFile_save_dir, trtFile_save_name=args.trtFile_save_name, FPMode=args.FPMode, 
                  images_dir=args.images_dir, detect_save_dir=args.detect_save_dir, class_names=class_names)