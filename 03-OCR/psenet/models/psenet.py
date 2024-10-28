import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import time

from .backbone import build_backbone
from .neck import build_neck
from .head import build_head
from .utils import Conv_BN_ReLU


class PSENet(nn.Module):
    def __init__(self,
                 backbone,
                 neck,
                 detection_head):
        super(PSENet, self).__init__()
        self.backbone = build_backbone(backbone)
        self.fpn = build_neck(neck)

        self.det_head = build_head(detection_head)

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self,
                imgs,
                gt_texts=None,
                gt_kernels=None,
                training_masks=None,
                img_metas=None,
                cfg=None):
        outputs = dict()

        if gt_texts!=None and gt_kernels!=None and training_masks!=None:
            if not self.training and cfg.report_speed: #false
                torch.cuda.synchronize()
                start = time.time()

        # backbone
        # f: P2, P3, P4, P5
        f = self.backbone(imgs)
        
        if gt_texts!=None and gt_kernels!=None and training_masks!=None:
            if not self.training and cfg.report_speed: #false
                torch.cuda.synchronize()
                outputs.update(dict(
                    backbone_time=time.time() - start
                ))
                start = time.time()

        # FPN
        f1, f2, f3, f4, = self.fpn(f[0], f[1], f[2], f[3])

        f = torch.cat((f1, f2, f3, f4), 1)
        
        if gt_texts!=None and gt_kernels!=None and training_masks!=None:
            if not self.training and cfg.report_speed: #false
                torch.cuda.synchronize()
                outputs.update(dict(
                    neck_time=time.time() - start
                ))
                start = time.time()

        # detection
        # 7个kernel对应的output
        det_out = self.det_head(f)

        if gt_texts!=None and gt_kernels!=None and training_masks!=None:
            if not self.training and cfg.report_speed:  #false
                torch.cuda.synchronize()
                outputs.update(dict(
                    det_head_time=time.time() - start
                ))

        if self.training:
            # 放大到与输入图像同尺寸， [736, 736]
            det_out = self._upsample(det_out, imgs.size())
            det_loss = self.det_head.loss(det_out, gt_texts, gt_kernels, training_masks)
            outputs.update(det_loss)
            return outputs
        else:
            det_out = self._upsample(det_out, imgs.size(), 1)
            # return det_out
            det_res = self.det_head.get_results(det_out, img_metas, cfg)
            outputs.update(det_res)
            return outputs

        
