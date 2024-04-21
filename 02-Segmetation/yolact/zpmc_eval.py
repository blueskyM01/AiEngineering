from data import COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
import pycocotools

from data import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image

import matplotlib.pyplot as plt
import cv2

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model',
                        default='weights/yolact_base_0.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--top_k', default=15, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                        help='Whether compute NMS cross-class or per-class.')
    parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display_bboxes', default=True, type=str2bool,
                        help='Whether or not to display bboxes around masks')
    parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])')
    parser.add_argument('--display_scores', default=True, type=str2bool,
                        help='Whether or not to display scores in addition to classes')
    parser.add_argument('--display', dest='display', action='store_true',
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
    parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                        help='In quantitative mode, the file to save detections before calculating mAP.')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='If display not set, this resumes mAP calculations from the ap_data_file.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
    parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                        help='The output file for coco bbox results if --coco_results is set.')
    parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                        help='The output file for coco mask results if --coco_results is set.')
    parser.add_argument('--config', default='yolact_base_config',
                        help='The config object to use.')
    parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
    parser.add_argument('--web_det_path', default='web/dets/', type=str,
                        help='If output_web_json is set, this is the path to dump detections into.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                        help='Equivalent to running display mode but without displaying an image.')
    parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                        help='Do not sort images by hashed image ID.')
    parser.add_argument('--seed', default=None, type=int,
                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                        help='Do not crop output masks with the predicted bounding box.')
    parser.add_argument('--image', default='/software/yangjianbing/temp/image_0000008691.jpeg:/software/yangjianbing/temp/image_0000008691-out.jpeg', type=str,
                        help='A path to an image to use for display.')
    parser.add_argument('--images', default=None, type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
    parser.add_argument('--video', default=None, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--video_multiframe', default=1, type=int,
                        help='The number of frames to evaluate in parallel to make videos play at higher fps.')
    parser.add_argument('--score_threshold', default=0.15, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--dataset', default='zpmc_cornerline_segmentation_dataset', type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')
    parser.add_argument('--display_fps', default=False, dest='display_fps', action='store_true',
                        help='When displaying / saving video, draw the FPS on the frame')
    parser.add_argument('--emulate_playback', default=False, dest='emulate_playback', action='store_true',
                        help='When saving a video, emulate the framerate that you\'d get running in real-time mode.')
    parser.add_argument('--detect_save_dir', default='results', type=str, help='')
    parser.add_argument('--class_name_path', default='/root/code/AI-Note-Demo/02-Segmetation/yolact/result_temp/coco.name', type=str, help='')
    parser.add_argument('--val_label_path', default='/root/code/AI-Note-Demo/02-Segmetation/yolact/result_temp/val_detection_parser.txt', type=str, help='')
    parser.add_argument('--val_img_dir', default='/root/code/dataset/coco/val2017', type=str, help='')

    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False, shuffle=False,
                        benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False, display_fps=False,
                        emulate_playback=False)

    global args
    args = parser.parse_args(argv)

    if args.output_web_json:
        args.output_coco_json = True
    
    if args.seed is not None:
        random.seed(args.seed)

iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {} # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})


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

def get_label(label_path):
    '''
    读取的".txt"文件的每行存储格式为： [num_gt, image_absolute_path, img_width, img_height, label_index, box_1, label_index, box_2, ..., label_index, box_n]
                              Box_x format: label_index x_min y_min x_max y_max. (The origin of coordinates is at the left top corner, left top => (xmin, ymin), right bottom => (xmax, ymax).)
                              num_gt：
                              label_index： is in range [0, class_num - 1].
                              For example:
                              2 xxx/xxx/a.jpg 1920 1080 0 453 369 473 391 1 588 245 608 268
                              2 xxx/xxx/b.jpg 1920 1080 1 466 403 485 422 2 793 300 809 320

    :param label_dir:
    :param label_name:
    :return: lines： 将".txt"文件的每行变成列表， 存储到lines这个大列表中
    '''
    lines = []
    with open(label_path, 'r') as f:
        line = f.readline()
        while line:
            lines.append(line.rstrip('\n').split(' '))
            line = f.readline()
    # random.shuffle(lines)
    return lines

def parse_line(line):
    '''
    功能： 获取每张图像上所有的标注矩形框和标注类别
    :param line: ".txt"文件的每行变成列表（每个元素都是字符串）
    :return: num_gt： 数据集中有多少个gt
             img_path： 图像的存储路径（绝对路径）
             annotations： [[xmin, ymin, xmax, ymax, label], [xmin, ymin, xmax, ymax, , label], ....]
             img_width： 图像的宽度
             img_height： 图像的高度
    '''
    num_gt = int(line[0])
    img_path = line[1]
    img_width = int(line[2])
    img_height = int(line[3])
    img_id = int(line[4])
    annotations = []

    s = line[5:]
    for i in range(len(s) // 5):
        label, xmin, ymin, xmax, ymax = float(s[i * 5]), float(s[i * 5 + 1]), float(s[i * 5 + 2]), float(
            s[i * 5 + 3]), float(s[i * 5 + 4])
        annotations.append([xmin, ymin, xmax, ymax, label])
    annotations = np.asarray(annotations, np.float32)
    return num_gt, img_path, annotations, img_width, img_height

def ZPMC_getClasses(zpmc_class_path_):
    '''
    description: read xxxx.txt to classes list
    :param zpmc_class_path_:
    :return:
    '''
    zpmc_classes_ = []
    with open(zpmc_class_path_, 'r') as f:
        zpmc_line_ = f.readline()
        while zpmc_line_:
            zpmc_classes_.append(zpmc_line_.rstrip('\n'))
            zpmc_line_ = f.readline()
    return zpmc_classes_

def generate_gt_files(output_dir, lines, classes):

    for line in lines:
        l_idx, img_name, annotations, img_width, img_height = parse_line(line)

        file_name = ''
        for e in img_name.split('/')[-1].split('.')[0:-1]:
            file_name += e
        file_name = os.path.join(output_dir, file_name) + '.txt'
        # print(file_name)
        f = open(file_name, 'w')

        for box in annotations:
            label = int(box[4])
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            line_str = classes[label+1] + ' ' + str(x0) + ' ' + str(y0) + ' ' + str(x1) + ' ' + str(y1) + '\n'
            f.write(line_str)
        f.close()
        print('Generate {}'.format(file_name))

def generate_detection_result_files(output_dir, annotations, scores, classes, img_name):
    file_name = ''
    for e in img_name.split('/')[-1].split('.')[0:-1]:
        file_name += e
    file_name = os.path.join(output_dir, file_name) + '.txt'
    # print(file_name)
    f = open(file_name, 'w')

    for box, score in zip(annotations, scores):
        label = int(box[4])
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        line_str = classes[label] + ' ' + str(score) + ' ' + str(x0) + ' ' + str(y0) + ' ' + str(x1) + ' ' + str(y1) + '\n'
        f.write(line_str)
    f.close()
    print('Generate {}'.format(file_name))

def evaluate(net:Yolact, dataset, train_mode=False):
    net.detect.use_fast_nms = args.fast_nms
    net.detect.use_cross_class_nms = args.cross_class_nms
    cfg.mask_proto_debug = args.mask_proto_debug

    CLASSES = ZPMC_getClasses(args.class_name_path)
    CLASSES.insert(0, 'BG')
    lines = get_label(args.val_label_path)

    gt_files_save_dir = os.path.join(args.detect_save_dir, 'ground-truth')
    if not os.path.exists(gt_files_save_dir):
        os.makedirs(gt_files_save_dir)
    else:
        os.system('rm -rf {}'.format(gt_files_save_dir+'/*'))
    detect_file_save_dir = os.path.join(args.detect_save_dir, 'detection-results')
    if not os.path.exists(detect_file_save_dir):
        os.makedirs(detect_file_save_dir)
    else:
        os.system('rm -rf {}'.format(detect_file_save_dir+'/*'))
    
    img_result_dir = os.path.join(args.detect_save_dir, 'visualizes')
    if not os.path.exists(img_result_dir):
        os.makedirs(img_result_dir)
    else:
        os.system('rm -rf {}'.format(img_result_dir+'/*'))

    generate_gt_files(gt_files_save_dir, lines, CLASSES)

    detect_counter = 0

    for line in lines:
        img_name = line[1]
        img_path = os.path.join(args.val_img_dir, img_name)
        # img_path, img_save_path = args.image.split(':')
        img_src = cv2.imread(img_path)
        h, w, _ = img_src.shape
        img_resize = cv2.resize(img_src, (550, 550), interpolation=cv2.INTER_LINEAR)
        frame = torch.from_numpy(img_resize).cuda().float()
        frame = frame.unsqueeze(0)
        preds = net(frame)

        zpmc_loc, zpmc_conf, zpmc_mask, zpmc_priors, zpmc_proto = preds
        zpmc_loc = zpmc_loc.cpu().numpy()
        zpmc_conf = zpmc_conf.cpu().numpy()
        zpmc_mask = zpmc_mask.cpu().numpy()
        zpmc_priors = zpmc_priors.cpu().numpy()
        zpmc_proto = zpmc_proto.cpu().numpy()
        images_resize_list = [img_resize]
        result = zpmc_PostProcess(CLASSES, zpmc_proto, zpmc_priors, zpmc_mask, zpmc_conf, zpmc_loc, conf_thresh=0.05)
        classes, scores, boxes, masks = zpmc_display(result, images_resize_list, h, w, score_threshold=0.15)

        if detect_counter < 10:
            for nclass, score, box, mask, image in zip(classes, scores, boxes, masks, [img_src]):
                
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
            cv2.imwrite(os.path.join(img_result_dir, img_name.split('/')[-1]), visualize_image)
            detect_counter += 1    
            print(detect_counter)

        for nclass, score, box, mask, image in zip(classes, scores, boxes, masks, [img_src]):
            if nclass is not None:
                nclass = nclass.reshape(-1, 1) + 1
                annotations = np.concatenate([box, nclass], axis=-1)
                generate_detection_result_files(detect_file_save_dir, annotations, score, CLASSES, img_name)
            else:
                annotations = []
                score = []
                generate_detection_result_files(detect_file_save_dir, annotations, score, CLASSES, img_name)
        

if __name__ == '__main__':
    parse_args()

    if args.config is not None:
        set_cfg(args.config)

    if args.dataset is not None:
        set_dataset(args.dataset)

    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')

        if args.cuda:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        dataset = None        

        print('Loading model {}...'.format(args.trained_model), end='')
        net = Yolact()
        net.load_weights(args.trained_model)
        net.eval()
        print(' Done.')

        if args.cuda:
            net = net.cuda()

        evaluate(net, dataset)


