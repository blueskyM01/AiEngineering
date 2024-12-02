import numpy as np
from PIL import Image
from torch.utils import data
import cv2, os, json
import random
import torchvision.transforms as transforms
import torch
import pyclipper
import Polygon as plg
import math
import string
import scipy.io as scio
import mmcv
import random

# ctw_root_dir = '/software/dataset/ZPMC_Container_Number/'
ctw_train_data_dir = '/root/code/dataset/containercode/images/train'
ctw_train_gt_dir = '/root/code/AiEngineering/03-OCR/psenet/data/train.json'
ctw_test_data_dir = '/root/code/dataset/containercode/images/val'
ctw_test_gt_dir = '/root/code/AiEngineering/03-OCR/psenet/data/val.json'


def get_img(img_path, read_type='pil'):
    try:
        if read_type == 'cv2':
            img = cv2.imread(img_path)
            img = img[:, :, [2, 1, 0]]
        elif read_type == 'pil':
            img = np.array(Image.open(img_path))
    except Exception as e:
        print(img_path)
        raise
    return img


def get_ann(img, gt_path):
    h, w = img.shape[0:2]
    # lines = mmcv.list_from_file(gt_path)
    lines = []
    for ann in gt_path:
        polygon = ann[1]
        lines.append(polygon)

    bboxes = []
    words = []
    for line in lines:
        # line = line.replace('\xef\xbb\xbf', '')
        # gt = line.split(',')
        gt = line

        # x1 = np.int(gt[0])
        # y1 = np.int(gt[1])

        # bbox = [np.int(gt[i]) for i in range(4, 32)]
        # bbox = np.asarray(bbox) + ([x1 * 1.0, y1 * 1.0] * 14)
        # bbox = np.asarray(bbox) / ([w * 1.0, h * 1.0] * 14)

        # bboxes.append(bbox)
        # words.append('???')


        num_ele = len(gt)
        bbox = [int(gt[i]) for i in range(num_ele)]
        bbox = np.asarray(bbox) / ([w * 1.0, h * 1.0] * (num_ele // 2))
        bboxes.append(bbox)
        words.append('???')

    return bboxes, words


def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w), flags=cv2.INTER_NEAREST)
        imgs[i] = img_rotation
    return imgs


def scale_aligned(img, scale):
    h, w = img.shape[0:2]
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img


def random_scale(img, short_size=736):
    h, w = img.shape[0:2]

    random_scale = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    scale = (np.random.choice(random_scale) * short_size) / min(h, w)

    img = scale_aligned(img, scale)
    return img


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


def random_crop_padding(imgs, target_size):
    h, w = imgs[0].shape[0:2]
    t_w, t_h = target_size
    p_w, p_h = target_size
    if w == t_w and h == t_h:
        return imgs

    t_h = t_h if t_h < h else h
    t_w = t_w if t_w < w else w

    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        # make sure to crop the text region
        tl = np.min(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        br[br < 0] = 0
        br[0] = min(br[0], h - t_h)
        br[1] = min(br[1], w - t_w)

        i = random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
        j = random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
    else:
        i = random.randint(0, h - t_h) if h - t_h > 0 else 0
        j = random.randint(0, w - t_w) if w - t_w > 0 else 0

    n_imgs = []
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            s3_length = int(imgs[idx].shape[-1])
            img = imgs[idx][i:i + t_h, j:j + t_w, :]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT,
                                       value=tuple(0 for i in range(s3_length)))
        else:
            img = imgs[idx][i:i + t_h, j:j + t_w]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT, value=(0,))
        n_imgs.append(img_p)
    return n_imgs


def dist(a, b):
    return np.linalg.norm((a - b), ord=2, axis=0)


def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri

# pyclipper提供了对线段和多边形的裁剪(Clipping)以及偏置(offseting)的功能，详见：https://www.cnblogs.com/zhigu/p/11943118.html
def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        try:
            # 可以将多边形向内缩放n个距离后的图形
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

            # 计算缩放的距离
            offset = min(int(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)
            # shrinked_bbox 就是内缩后的路径。Execute()里面的参数就是缩放的距离
            shrinked_bbox = pco.Execute(-offset)
            if len(shrinked_bbox) == 0:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bbox = np.array(shrinked_bbox[0])
            if shrinked_bbox.shape[0] <= 2:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bboxes.append(shrinked_bbox)
        except Exception as e:
            print(type(shrinked_bbox), shrinked_bbox)
            print('area:', area, 'peri:', peri)
            shrinked_bboxes.append(bbox)

    return shrinked_bboxes


class PSENET_CTW(data.Dataset):
    def __init__(self,
                 split='train',
                 is_transform=False,
                 img_size=None,
                 short_size=736,
                 kernel_num=7,
                 min_scale=0.4,
                 read_type='pil',
                 report_speed=False):
        self.split = split
        self.is_transform = is_transform

        self.img_size = img_size if (img_size is None or isinstance(img_size, tuple)) else (img_size, img_size)
        self.kernel_num = kernel_num
        self.min_scale = min_scale
        self.short_size = short_size
        self.read_type = read_type

        if split == 'train':
            # data_dirs = [ctw_train_data_dir]
            # gt_dirs = [ctw_train_gt_dir]
            data_dirs = ctw_train_data_dir
            gt_dirs = ctw_train_gt_dir
        elif split == 'test':
            # data_dirs = [ctw_test_data_dir]
            # gt_dirs = [ctw_test_gt_dir]
            data_dirs = ctw_test_data_dir
            gt_dirs = ctw_test_gt_dir
        else:
            print('Error: split must be test or train!')
            raise

        self.img_paths = []
        self.gt_paths = []

        # for data_dir, gt_dir in zip(data_dirs, gt_dirs):
        #     img_names = [img_name for img_name in mmcv.utils.scandir(data_dir, '.jpg')]
        #     img_names.extend([img_name for img_name in mmcv.utils.scandir(data_dir, '.png')])

        #     img_paths = []
        #     gt_paths = []
        #     for idx, img_name in enumerate(img_names):
        #         img_path = data_dir + img_name
        #         img_paths.append(img_path)

        #         gt_name = img_name.split('.')[0] + '.txt'
        #         gt_path = gt_dir + gt_name
        #         gt_paths.append(gt_path)

        #     self.img_paths.extend(img_paths)
        #     self.gt_paths.extend(gt_paths)
        # img_names = os.listdir(data_dirs)

        with open(gt_dirs, 'r') as load_f:
            load_dict = json.load(load_f)
        load_f.close()

        img_names = list(load_dict.keys())
        random.shuffle(img_names)
        
        for img_name in img_names:
            img_path = os.path.join(data_dirs, img_name)
            self.img_paths.append(img_path)
            ocr_anns = load_dict[img_name]
            self.gt_paths.append(ocr_anns)

            # if '.jpg' in img_name:
            #     label_name = img_name.split('.jpg')[0] + '.txt'
            #     label_path = os.path.join(gt_dirs, label_name)
            #     img_path = os.path.join(data_dirs, img_name)
            #     self.img_paths.append(img_path)
            #     self.gt_paths.append(label_path)
            # if '.png' in img_name:
            #     label_name = img_name.split('.png')[0] + '.txt'
            #     label_path = os.path.join(gt_dirs, label_name)
            #     img_path = os.path.join(data_dirs, img_name)
            #     self.img_paths.append(img_path)
            #     self.gt_paths.append(label_path)

        if report_speed: # False
            target_size = 3000
            data_size = len(self.img_paths)
            extend_scale = (target_size + data_size - 1) // data_size
            self.img_paths = (self.img_paths * extend_scale)[:target_size]
            self.gt_paths = (self.gt_paths * extend_scale)[:target_size]

        self.max_word_num = 200
        

    def __len__(self):
        return len(self.img_paths)

    def prepare_train_data(self, index):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        # 读取图像，格式BGR
        img = get_img(img_path, self.read_type)
        # np数组，shape=[num_rows, 28], 28代表多边形，x,y一组，共14组，注：已进行归一化处理：0~1
        bboxes, words = get_ann(img, gt_path) 

        # 若标注的行数超过self.max_word_num，则取前self.max_word_num个，这里self.max_word_num=200
        if len(bboxes) > self.max_word_num:
            bboxes = bboxes[:self.max_word_num]

        # 图像预处理，缩放图像，增加模型；鲁棒性
        if self.is_transform:
            img = random_scale(img, self.short_size)

        # 
        gt_instance = np.zeros(img.shape[0:2], dtype='uint8') 
        # 
        training_mask = np.ones(img.shape[0:2], dtype='uint8')
        if len(bboxes) > 0:
            # 对每行的bbox，进行mask绘制，最终生成gt_instance
            for i in range(len(bboxes)): 
                # 由于bbox已进行了归一化处理，因此，只需乘以图像w,h，即可获取真实坐标
                bboxes[i] = np.reshape(bboxes[i] * ([img.shape[1], img.shape[0]] * (bboxes[i].shape[0] // 2)),
                                       (bboxes[i].shape[0] // 2, 2)).astype('int32')
            # mask绘制
            for i in range(len(bboxes)):
                cv2.drawContours(gt_instance, [bboxes[i]], -1, i + 1, -1)
                if words[i] == '###':
                    cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)

                
            # cv2.imwrite('/software/code/zpmc_psenet_ocr/outputs/show/1.jpg', gt_instance * 125)
            # cv2.imwrite('/software/code/zpmc_psenet_ocr/outputs/show/1-1.jpg', img)

        # PSENet需要进行多个kernel的融合。这里self.kernel_num=7
        # 生成多个gt_kernel，生成6个，第七个就是他本身
        # 注意：gt_kernel的w, h保持不变，只是图中的mask进行了缩放
        gt_kernels = []
        for i in range(1, self.kernel_num):
            rate = 1.0 - (1.0 - self.min_scale) / (self.kernel_num - 1) * i
            gt_kernel = np.zeros(img.shape[0:2], dtype='uint8')
            kernel_bboxes = shrink(bboxes, rate)
            for i in range(len(bboxes)):
                cv2.drawContours(gt_kernel, [kernel_bboxes[i].astype(int)], -1, 1, -1)
            gt_kernels.append(gt_kernel)

        if self.is_transform:
            imgs = [img, gt_instance, training_mask]
            imgs.extend(gt_kernels)

            imgs = random_horizontal_flip(imgs)
            imgs = random_rotate(imgs)
            imgs = random_crop_padding(imgs, self.img_size)
            img, gt_instance, training_mask, gt_kernels = imgs[0], imgs[1], imgs[2], imgs[3:]

        gt_text = gt_instance.copy()
        gt_text[gt_text > 0] = 1
        gt_kernels = np.array(gt_kernels)

        if self.is_transform:
            img = Image.fromarray(img)
            img = img.convert('RGB')
            img = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(img)
        else:
            img = Image.fromarray(img)
            img = img.convert('RGB')

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        # 原图上的mask，shape=[736, 736]
        gt_text = torch.from_numpy(gt_text).long()
        # 生成的6个kernel, shape=[6, 736, 736]
        gt_kernels = torch.from_numpy(gt_kernels).long()
        # 同原图上的mask，但都是1，没做任何处理, shape=[736, 736]
        training_mask = torch.from_numpy(training_mask).long()

        data = dict(
            imgs=img,
            gt_texts=gt_text,
            gt_kernels=gt_kernels,
            training_masks=training_mask,
        )

        return data
        # return img, gt_text, gt_kernels, training_mask

    def prepare_test_data(self, index):
        img_path = self.img_paths[index]

        img = get_img(img_path, self.read_type)
        img_meta = dict(
            org_img_size=np.array(img.shape[:2])
        )

        img = scale_aligned_short(img, self.short_size)
        img_meta.update(dict(
            img_size=np.array(img.shape[:2])
        ))

        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        data = dict(
            imgs=img,
            img_metas=img_meta
        )

        return data

    def __getitem__(self, index):
        if self.split == 'train':
            return self.prepare_train_data(index)
        elif self.split == 'test':
            return self.prepare_test_data(index)
