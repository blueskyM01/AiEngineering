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


def zpmc_ImagePreprocess(images, img_size):
    # mean = np.array([103.94, 116.78, 123.68]).reshape(1,3,1,1)
    # std = np.array([57.38, 57.12, 58.4]).reshape(1,3,1,1)

    images_resize_list = []
    for image in images:
        image_resize = cv2.resize(image, img_size, interpolation=cv2.INTER_LINEAR)
        images_resize_list.append(image_resize)
    # images_resize_np = np.array(images_resize_list)

    # images_resize_np = np.transpose(images_resize_np, (0, 3, 1, 2))
    # images_resize_np = (images_resize_np - mean) / std

    # images_resize_np = images_resize_np[:, (2, 1, 0), :, :]
    return images_resize_list

def zpmc_decode(loc, priors):
    variances = [0.1, 0.2]
        
    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [n,A,4].
      box_b: (tensor) bounding boxes, Shape: [n,B,4].
    Return:
      (tensor) intersection area, Shape: [n,A,B].
    """
    n = box_a.shape[0]
    A = box_a.shape[1]
    B = box_b.shape[1]

    box_a_br_max = np.expand_dims(box_a[:, :, 2:], 2)#.expand(n, A, B, 2)
    box_b_br_max = np.expand_dims(box_b[:, :, 2:], 1)#.expand(n, A, B, 2)
    box_a_br_max = np.broadcast_to(box_a_br_max, (n, A, B, 2))
    box_b_br_max = np.broadcast_to(box_b_br_max, (n, A, B, 2))
    max_xy = np.minimum(box_a_br_max, box_b_br_max)

    box_a_br_min = np.expand_dims(box_a[:, :, :2], 2)#.expand(n, A, B, 2)
    box_b_br_min = np.expand_dims(box_b[:, :, :2], 1)#.expand(n, A, B, 2)
    box_a_br_min = np.broadcast_to(box_a_br_min, (n, A, B, 2))
    box_b_br_min = np.broadcast_to(box_b_br_min, (n, A, B, 2))
    min_xy = np.maximum(box_a_br_min, box_b_br_min)


    # inter = torch.clamp((max_xy - min_xy), min=0)
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=10e10)
    return inter[:, :, :, 0] * inter[:, :, :, 1]

def jaccard(box_a, box_b, iscrowd:bool=False):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    use_batch = True
    # if box_a.dim() == 2:
    #     use_batch = False
    #     box_a = box_a[None, ...]
    #     box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)
    inter_shape = inter.shape
    area_a_br = (box_a[:, :, 2]-box_a[:, :, 0]) * (box_a[:, :, 3]-box_a[:, :, 1])
    area_a_br = np.expand_dims(area_a_br, 2)
    area_a = np.broadcast_to(area_a_br, inter_shape)

    area_b_br = (box_b[:, :, 2]-box_b[:, :, 0]) * (box_b[:, :, 3]-box_b[:, :, 1])
    area_b_br = np.expand_dims(area_b_br, 1)
    area_b = np.broadcast_to(area_b_br, inter_shape)
    # area_a = ((box_a[:, :, 2]-box_a[:, :, 0]) * 
    #           (box_a[:, :, 3]-box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
    # area_b = ((box_b[:, :, 2]-box_b[:, :, 0]) *
    #           (box_b[:, :, 3]-box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter

    out = inter / area_a if iscrowd else inter / union
    # return out if use_batch else out.squeeze(0)
    return out

def fast_nms(boxes, masks, scores, iou_threshold:float=0.5, top_k:int=200):
    # scores, idx = scores.sort(1, descending=True)

    idx = np.argsort(scores, axis=1)
    idx = idx[:, ::-1]
    scores = scores[:, idx[0]]
    idx = idx[:, :top_k]
    scores = scores[:, :top_k]

    num_classes, num_dets = idx.shape

    boxes = boxes[idx.reshape(-1), :].reshape(num_classes, num_dets, 4)
    masks = masks[idx.reshape(-1), :].reshape(num_classes, num_dets, -1)

    iou = jaccard(boxes, boxes)
    iou=np.triu(iou, k=1)

    # iou_max_idx = np.argmax(iou, axis=1)
    iou_max =np.max(iou, axis=1)    

    # Now just filter out the ones higher than the threshold
    keep = (iou_max <= iou_threshold)

    classes = np.arange(num_classes)[:, None]
    classes = np.broadcast_to(classes, keep.shape)
    classes = classes[keep]

    boxes = boxes[keep]
    masks = masks[keep]
    scores = scores[keep]

    idx = np.argsort(scores, axis=0)
    idx = idx[::-1]
    scores = scores[idx]

    idx = idx[:100] # 最终的检测目标数不超过100
    scores = scores[idx]

    classes = classes[idx]
    boxes = boxes[idx]
    masks = masks[idx]

    return boxes, masks, classes, scores

def zpmc_detect(batch_idx, conf_preds, decoded_boxes, mask_data, conf_thresh):
    cur_scores = conf_preds[batch_idx, 1:, :]
    conf_scores = np.max(cur_scores, axis=0)

    keep = (conf_scores > conf_thresh)
    scores = cur_scores[:, keep]
    boxes = decoded_boxes[keep, :]
    masks = mask_data[batch_idx, keep, :]

    if scores.shape[1] == 0:
        return None

    boxes, masks, classes, scores = fast_nms(boxes, masks, scores, iou_threshold=0.5, top_k=200)

    return {'box': boxes, 'mask': masks, 'class': classes, 'score': scores}


def zpmc_PostProcess(classes, proto_data, prior_data, mask_data, conf_data, loc_data, conf_thresh):
    num_classes = len(classes)
    batch_size = loc_data.shape[0]
    num_priors = prior_data.shape[0]
    
    conf_preds = conf_data.reshape(batch_size, num_priors, num_classes)
    conf_preds = np.transpose(conf_preds, (0, 2, 1))
    out = []
    for batch_idx in range(batch_size):
        decoded_boxes = zpmc_decode(loc_data[batch_idx], prior_data)
        result = zpmc_detect(batch_idx, conf_preds, decoded_boxes, mask_data, conf_thresh)
        if result is not None:
            result['proto'] = proto_data[batch_idx]
        out.append(result)
    return out
def zpmc_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def zpmc_sanitize_coordinates(_x1, _x2, img_size:int, padding:int=0, cast:bool=True):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates and casts the results to long tensors.

    If cast is false, the result won't be cast to longs.
    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size
    if cast:
        _x1 = _x1.long()
        _x2 = _x2.long()
    x1 = np.minimum(_x1, _x2)
    x2 = np.maximum(_x1, _x2)
    x1 = np.clip(x1-padding, a_min=0, a_max=10e10)
    x2 = np.clip(x2+padding, a_min=-10e10, a_max=img_size)

    return x1, x2

def zpmc_crop(masks, boxes, padding:int=1):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """
    h, w, n = masks.shape
    x1, x2 = zpmc_sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding, cast=False)
    y1, y2 = zpmc_sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding, cast=False)

    # rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, -1, 1).expand(h, w, n)
    # cols = torch.arange(h, device=masks.device, dtype=x1.dtype).view(-1, 1, 1).expand(h, w, n)
    rows = np.arange(w, dtype=x1.dtype).reshape(1, -1, 1)
    rows = np.broadcast_to(rows, (h, w, n))
    cols = np.arange(h, dtype=x1.dtype).reshape(-1, 1, 1)
    cols = np.broadcast_to(cols, (h, w, n))
    
    # masks_left  = rows >= x1.view(1, 1, -1)
    # masks_right = rows <  x2.view(1, 1, -1)
    # masks_up    = cols >= y1.view(1, 1, -1)
    # masks_down  = cols <  y2.view(1, 1, -1)
    masks_left  = rows >= x1.reshape(1, 1, -1)
    masks_right = rows <  x2.reshape(1, 1, -1)
    masks_up    = cols >= y1.reshape(1, 1, -1)
    masks_down  = cols <  y2.reshape(1, 1, -1)
    
    crop_mask = masks_left * masks_right * masks_up * masks_down
    
    return masks * crop_mask.astype(np.float32)

def zpmc_display(dets_out, imgs, h, w, mask_alpha=0.45, score_threshold=0.15):
    pre_classes, pre_scores, pre_boxes, pre_masks = [], [], [], []
    for img, dets in zip(imgs, dets_out):
        # img_gpu = img / 255.0
        # h, w, _ = img.shape
        if dets != None:
            # for dets in dets_out:
            if dets is None:
                return None

            if score_threshold > 0:
                keep = dets['score'] > score_threshold

                for k in dets:
                    if k != 'proto':
                        dets[k] = dets[k][keep]

                if dets['score'].shape[0] == 0:
                    # return None
                    pre_classes.append(None)
                    pre_scores.append(None)
                    pre_boxes.append(None)
                    pre_masks.append(None)
                    continue

            # Actually extract everything from dets now
            classes = dets['class']
            boxes   = dets['box']
            scores  = dets['score']
            masks   = dets['mask']
            proto_data = dets['proto']

            masks = proto_data @ masks.T
            masks = zpmc_sigmoid(masks)
            masks = zpmc_crop(masks, boxes) * 255

            masks = np.sum(masks, axis=2)
            masks = np.clip(masks, a_min=0.0, a_max=255.0)

            # masks = cv2.resize(masks, (w, h), interpolation=cv2.INTER_LINEAR)
            # masks = np.where(masks > 80, 255, 0)
            # masks = cv2.cvtColor(masks.astype(np.uint8), cv2.COLOR_GRAY2BGR)

            boxes[:, 0], boxes[:, 2] = zpmc_sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, cast=False)
            boxes[:, 1], boxes[:, 3] = zpmc_sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, cast=False)



            pre_classes.append(classes)
            pre_scores.append(scores)
            pre_boxes.append(boxes)
            pre_masks.append(masks)
        else:
            pre_classes.append(None)
            pre_scores.append(None)
            pre_boxes.append(None)
            pre_masks.append(None)
    return pre_classes, pre_scores, pre_boxes, pre_masks

def daw_mask(src, mask):
    color_mask = np.zeros_like(src)
    color_mask[:, :, 2] = 255
    mask = np.expand_dims(mask, axis=-1)
    mask = np.repeat(mask, 3, axis=-1)
    tem_img = (0.4 * src.astype(np.float32) + 0.6 * color_mask.astype(np.float32)).astype(np.uint8)
    dst_img = np.where(mask == 0, src, tem_img)
    return dst_img

def draw_caption(img, b, caption, score):
    '''

    :param img:
    :param b: np.array([x_min, y_min, x_max, y_max])
    :param caption: str
    :param score: float
    :return:
    '''
    score = round(score, 4)
    num = 0
    for ii in caption:
        num += 1
    cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), ((0, 0, 255)), thickness=2)
    cv2.rectangle(img, (b[0], b[1]), (b[0] + 9 * (num + 5), b[1] + 10), (255, 0, 0), thickness=-1)
    cv2.putText(img, caption + ':' + str(score), (b[0], b[1] + 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(img, caption + ':' + str(score), (b[0], b[1] + 10), cv2.FONT_HERSHEY_PLAIN, 1,
                (255, 255, 255), 1)
    return img

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

def zpmc_onnx2trt(onnxFile, trtFile_save_dir, trtFile_save_name, FPMode, video_dir, detect_save_dir):
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
    print('inputTensor:', inputTensor.name, inputTensor.shape)
    print('outputTensor0:', outputTensor0.name, outputTensor0.shape)
    print('outputTensor1:', outputTensor1.name, outputTensor1.shape)
    print('outputTensor2:', outputTensor2.name, outputTensor2.shape)
    print('outputTensor3:', outputTensor3.name, outputTensor3.shape)
    print('outputTensor4:', outputTensor4.name, outputTensor4.shape)

    _, nHeight, nWidth, _ = inputTensor.shape
    profile.set_shape(inputTensor.name, (4, nHeight, nWidth, 3), (4, nHeight, nWidth, 3), (4, nHeight, nWidth, 3)) # 最小batch，常见batch，最大batch
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

    
    mx_Capture_0 = cv2.VideoCapture(os.path.join(video_dir, 'cam1_000.avi'))
    mx_Capture_1 = cv2.VideoCapture(os.path.join(video_dir, 'cam2_000.avi'))
    mx_Capture_2 = cv2.VideoCapture(os.path.join(video_dir, 'cam3_000.avi'))
    mx_Capture_3 = cv2.VideoCapture(os.path.join(video_dir, 'cam4_000.avi'))
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    size = (int(mx_Capture_0.get(cv2.CAP_PROP_FRAME_WIDTH) * 2), int(mx_Capture_0.get(cv2.CAP_PROP_FRAME_HEIGHT) * 2))
    if not os.path.exists(detect_save_dir):  # os模块判断并创建
        os.mkdir(detect_save_dir)
    merge_out = cv2.VideoWriter(os.path.join(detect_save_dir, FPMode+'.avi'), fourcc, 22, size)

    counter = 0
    while 1:
        mx_ret_0, image_src_0 = mx_Capture_0.read()
        mx_ret_1, image_src_1 = mx_Capture_1.read()
        mx_ret_2, image_src_2 = mx_Capture_2.read()
        mx_ret_3, image_src_3 = mx_Capture_3.read()

        if mx_ret_0 and mx_ret_1 and mx_ret_2 and mx_ret_3:
            h, w, _ = image_src_0.shape
            images = [image_src_0, image_src_1, image_src_2, image_src_3]
            images_resize_list = zpmc_ImagePreprocess(images, img_size=(550, 550))

            starttime = datetime.datetime.now()    

            inputs[0].host = np.ascontiguousarray(images_resize_list, dtype=np.float32)
            # cfx.push()
            trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            # cfx.pop()
            batch_size = len(images_resize_list)
            trt_outputs_shape = [(batch_size, 138, 138, 32), (batch_size, 19248, 4), (batch_size, 19248, 32), (batch_size, 19248, 2), (19248, 4)]
            
            proto_data = trt_outputs[0].reshape(trt_outputs_shape[0])
            loc_data = trt_outputs[1].reshape(trt_outputs_shape[1])
            mask_data = trt_outputs[2].reshape(trt_outputs_shape[2])
            conf_data = trt_outputs[3].reshape(trt_outputs_shape[3])
            prior_data = trt_outputs[4].reshape(trt_outputs_shape[4])
            
            CLASSES = ['BG', 'LOCKHOLE']
            result = zpmc_PostProcess(CLASSES, proto_data, prior_data, mask_data, conf_data, loc_data, conf_thresh=0.05)
            classes, scores, boxes, masks = zpmc_display(result, images_resize_list, h, w, score_threshold=0.15)

            endtime = datetime.datetime.now()
            timediff = (endtime - starttime).total_seconds()
            counter+=1
            print(timediff, '    ', 1/timediff, '   ', counter)

            merge_images = []
            for nclass, score, box, mask, image in zip(classes, scores, boxes, masks, images):    
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                mask = np.where(mask > 80, 255, 0).astype(np.uint8)
                visualize_image = daw_mask(image, mask)
                for lable, sc, bb in zip(nclass, score, box):
                    caption = CLASSES[lable+1]
                    x0 = int(bb[0])
                    y0 = int(bb[1])
                    x1 = int(bb[2])
                    y1 = int(bb[3])
                    aa = [x0, y0, x1, y1]
                    visualize_image = draw_caption(visualize_image, aa, caption, sc)  
                merge_images.append(visualize_image)
            merge_img = merge_image(merge_images)
            merge_out.write(merge_img)
        else:
            mx_Capture_0.release()
            mx_Capture_1.release()
            mx_Capture_2.release()
            mx_Capture_3.release()
            merge_out.release()
            break
        #     cv2.imwrite('results/' + str(counter) + '.jpg', visualize_image)
        #     counter += 1

    

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN')
    parser.add_argument('--onnxFile', default='onnx/yolact.onnx', type=str, help='')
    parser.add_argument('--trtFile_save_dir', default='trt', type=str, help='')
    parser.add_argument('--trtFile_save_name', default='yolact16.trt', type=str, help='')
    parser.add_argument('--FPMode', default='FP16', type=str, help='')
    parser.add_argument('--video_dir', default='/root/code/temp_file/firstlanding/', type=str, help='')
    parser.add_argument('--detect_save_dir', default='result', type=str, help='')
    args = parser.parse_args()

    zpmc_onnx2trt(onnxFile=args.onnxFile, trtFile_save_dir=args.trtFile_save_dir, trtFile_save_name=args.trtFile_save_name, FPMode=args.FPMode, 
                  video_dir=args.video_dir, detect_save_dir=args.detect_save_dir)
