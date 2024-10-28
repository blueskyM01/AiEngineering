import cv2, json, argparse
import numpy as np
import torch, os, sys
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from models import build_model
from mmcv import Config
import os.path as osp

font = cv2.FONT_HERSHEY_SIMPLEX


def fuse_conv_bn(conv, bn):
    """During inference, the functionary of batch norm layers is turned off but
    only the mean and var alone channels are used, which exposes the chance to
    fuse it with the preceding conv layers to save computations and simplify
    network structures."""
    conv_w = conv.weight
    conv_b = conv.bias if conv.bias is not None else torch.zeros_like(
        bn.running_mean)

    factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    conv.weight = nn.Parameter(conv_w *
                               factor.reshape([conv.out_channels, 1, 1, 1]))
    conv.bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)
    return conv

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

def fuse_module(m):
    last_conv = None
    last_conv_name = None

    for name, child in m.named_children():
        if isinstance(child, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            fused_conv = fuse_conv_bn(last_conv, child)
            m._modules[last_conv_name] = fused_conv
            # To reduce changes, set BN as Identity instead of deleting it.
            m._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, nn.Conv2d):
            last_conv = child
            last_conv_name = name
        else:
            fuse_module(child)
    return m

def main(args):
    gt_save_dir = os.path.join(args.result_save_dir, 'ground-truth')
    detection_save_dir = os.path.join(args.result_save_dir, 'detection-results')
    
    if not os.path.exists(args.result_save_dir):  # os模块判断并创建
        os.mkdir(args.result_save_dir)

    if not os.path.exists(gt_save_dir):  # os模块判断并创建
        os.mkdir(gt_save_dir)

    if not os.path.exists(detection_save_dir):  # os模块判断并创建
        os.mkdir(detection_save_dir)


    # config param
    cfg = Config.fromfile(args.config)
    for d in [cfg, cfg.data.test]:
        d.update(dict(
            report_speed=args.report_speed
        ))
    print(json.dumps(cfg._cfg_dict, indent=4))
    sys.stdout.flush() 

    # model
    model = build_model(cfg.model)
    model = model.cuda()

    # load param
    if os.path.isfile(args.checkpoint):
        print("Loading model and optimizer from checkpoint '{}'".format(args.checkpoint))
        sys.stdout.flush()

        checkpoint = torch.load(args.checkpoint)

        d = dict()
        for key, value in checkpoint['state_dict'].items():
            tmp = key[7:]
            d[tmp] = value
        model.load_state_dict(d)
    else:
        print("No checkpoint found at '{}'".format(args.resume))
        raise

    model = fuse_module(model)
    model.eval()

    with open(args.ann_path, 'r') as load_f:
        load_dict = json.load(load_f)
    load_f.close()

    img_names = list(load_dict.keys())
    
    for img_name in img_names:
        img_name_base = osp.basename(img_name)
        f = open(os.path.join(gt_save_dir, img_name_base + '.txt'), 'w')
        ocr_anns = load_dict[img_name]
        for ocr_ann in ocr_anns:
            text = ocr_ann[0]
            polygon = ocr_ann[1]
            contour = []
            for i in range(0, len(polygon), 2):
                point = [int(polygon[i]), int(polygon[i+1])]
                contour.append(point)
            contour = np.array(contour)
            x0 = contour[:, 0].min()
            y0 = contour[:, 1].min()
            x1 = contour[:, 0].max()
            y1 = contour[:, 1].max()
            line_str = 'num' + ' ' + str(x0) + ' ' + str(y0) + ' ' + str(x1) + ' ' + str(y1) + '\n'
            f.write(line_str)
        f.close()

    for img_name in img_names:
        img_name_base = osp.basename(img_name)
        f = open(os.path.join(detection_save_dir, img_name_base + '.txt'), 'w')
        # 读取图像
        image_name = os.path.join(args.img_dir, img_name)
        img_org = cv2.imread(image_name)
        img = img_org[:, :, [2, 1, 0]]

        img_meta = dict(
                org_img_size=np.array(img.shape[:2]).reshape(-1, 2)
            )
        img = scale_aligned_short(img, short_size=736)
        img_meta.update(dict(
                img_size=np.array(img.shape[:2]).reshape(-1, 2)
            ))

        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        img = img.unsqueeze(0)
        data = dict(
            imgs=img,
            img_metas=img_meta
        )

        # prepare input
        data['imgs'] = data['imgs'].cuda()
        data.update(dict(
            cfg=cfg
        ))

        with torch.no_grad():
            outputs = model(**data)

        
        bboxes = outputs['bboxes']
        scores = outputs['scores']
        contours = []
        scores_ = []
        for score, bbox in zip(scores, bboxes):
            contour = []
            for i in range(0, len(bbox), 2):
                point = [int(bbox[i]), int(bbox[i+1])]
                contour.append(point)

            # cv2.putText(img_org, str(score), (int(bbox[0]), int(bbox[1])-5), font, 2, (255, 255, 255), 4)
            
            contours.append(np.array(contour))
            scores_.append(score)
        # cv2.drawContours(img_org, contours, -1, color=(255,0,0), thickness=2)
        # cv2.imwrite('./1q.jpg', img_org)

        for score, cc in zip(scores_, contours):
            x0 = cc[:, 0].min()
            y0 = cc[:, 1].min()
            x1 = cc[:, 0].max()
            y1 = cc[:, 1].max()

            line_str = 'num' + ' ' + str(score) + ' ' + str(x0) + ' ' + str(y0) + ' ' + str(x1) + ' ' + str(y1) + '\n'
            f.write(line_str)
        f.close()
        print('generate {}'.format(os.path.join(detection_save_dir, img_name + '.txt')))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--config', help='', type=str, default='config/psenet/psenet_r50_ctw.py')
    parser.add_argument('--checkpoint', nargs='?', type=str, default='checkpoints/checkpoint_epoch_1.pth.tar')
    parser.add_argument('--report_speed', type=bool, default=False)
    parser.add_argument('--img_dir', help='', type=str, default='/root/code/dataset/containercode/images/val')
    parser.add_argument('--ann_path', help='', type=str, default='/root/code/AiEngineering/03-OCR/psenet/data/val.json')
    parser.add_argument('--result_save_dir', help='', type=str, default='/root/code/AiEngineering/03-OCR/psenet/data')
    args = parser.parse_args()

    main(args)