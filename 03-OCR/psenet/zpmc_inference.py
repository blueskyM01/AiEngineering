import torch
import numpy as np
import argparse
import os, cv2
import os.path as osp
import sys
import time
import json
from mmcv import Config

from dataset import build_data_loader
from models import build_model
from models.utils import fuse_module
from utils import ResultFormat, AverageMeter

font = cv2.FONT_HERSHEY_SIMPLEX

def report_speed(outputs, speed_meters):
    total_time = 0
    for key in outputs:
        if 'time' in key:
            total_time += outputs[key]
            speed_meters[key].update(outputs[key])
            print('%s: %.4f' % (key, speed_meters[key].avg))

    speed_meters['total_time'].update(total_time)
    print('FPS: %.1f' % (1.0 / speed_meters['total_time'].avg))


def test(test_loader, model, cfg):
    model.eval()

    rf = ResultFormat(cfg.data.test.type, cfg.test_cfg.result_path)

    if cfg.report_speed:
        speed_meters = dict(
            backbone_time=AverageMeter(500),
            neck_time=AverageMeter(500),
            det_head_time=AverageMeter(500),
            det_pse_time=AverageMeter(500),
            rec_time=AverageMeter(500),
            total_time=AverageMeter(500)
        )

    for idx, data in enumerate(test_loader):
        print('Testing %d/%d' % (idx, len(test_loader)))
        sys.stdout.flush()

        # prepare input
        data['imgs'] = data['imgs'].cuda()
        data.update(dict(
            cfg=cfg
        ))

        # forward
        with torch.no_grad():
            outputs = model(**data)

        if cfg.report_speed:
            report_speed(outputs, speed_meters)

        # save result
        image_name, _ = osp.splitext(osp.basename(test_loader.dataset.img_paths[idx]))
        image_path = test_loader.dataset.img_paths[idx]

        img = cv2.imread(image_path)
        bboxes = outputs['bboxes']
        scores = outputs['scores']
        contours = []
        for score, bbox in zip(scores, bboxes):
            contour = []
            for i in range(0, len(bbox), 2):
                point = [int(bbox[i]), int(bbox[i+1])]
                contour.append(point)

            cv2.putText(img, str(score), (int(bbox[0]), int(bbox[1])-5), font, 2, (255, 255, 255), 4)
            
            contours.append(np.array(contour))
        cv2.drawContours(img, contours, -1, color=(255,0,0), thickness=2)
        if not os.path.exists('outputs/show/'):
            os.makedirs('outputs/show/')
        cv2.imwrite('outputs/show/' + image_name+'.jpg', img)

def main(args):
    cfg = Config.fromfile(args.config)
    for d in [cfg, cfg.data.test]:
        d.update(dict(
            report_speed=args.report_speed
        ))
    print(json.dumps(cfg._cfg_dict, indent=4))
    sys.stdout.flush()

    # data loader
    data_loader = build_data_loader(cfg.data.test)
    test_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )
    # model
    model = build_model(cfg.model)
    model = model.cuda()

    if args.checkpoint is not None:
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

    # fuse conv and bn
    model = fuse_module(model)

    # test
    test(test_loader, model, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--config', help='', type=str, default='config/psenet/psenet_r50_ctw.py')
    parser.add_argument('--checkpoint', nargs='?', type=str, default='checkpoints/checkpoint_epoch_1.pth.tar')
    parser.add_argument('--img_dir', nargs='?', type=str, default='/root/code/dataset/containercode/images/val')
    parser.add_argument('--report_speed', type=bool, default=False)
    args = parser.parse_args()

    main(args)