import numpy as np
import cv2, os, argparse, json
import random
from pycocotools import mask as maskUtils

class zpmc_ShowReslut:
    def __init__(self, dataset_dir, dataset_name, label_dir, label_name, class_dir, class_name, result_save_dir):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.label_dir = label_dir
        self.label_name = label_name
        self.class_dir = class_dir
        self.class_name = class_name
        self.result_save_dir = result_save_dir

        self.classes = self.get_classes(os.path.join(class_dir, class_name))
        self.lines = self.get_label(self.label_dir, self.label_name)

        print("The classes are:", self.classes)

    def show_result(self):

        for image_name in self.lines.keys():
            src_img = cv2.imread(os.path.join(self.dataset_dir, self.dataset_name, image_name))
            line = self.lines[image_name]
            ids, iscrowds, segs, img_width, img_height, img_id = self.parse_line(line)
            merge_mask = np.zeros([img_height, img_width], dtype=np.uint8)
            bboxes = []
            for id, seg in zip(ids, segs):
                src_mask = self.annToMask(seg, img_height, img_width)
                bbox = self.get_bbox(src_mask)
                bboxes.append(bbox)
                merge_mask = np.where(src_mask == 0, merge_mask, src_mask)

            dst_img = self.daw_mask(src_img, merge_mask)
            for id, bbox in zip(ids,bboxes):
                dst_img = self.draw_caption(dst_img, bbox, self.classes[int(id)], 1.0)
            cv2.imshow('image_name', dst_img)
            cv2.waitKey(2000)

    def save_result(self):
        for image_name in self.lines.keys():
            img_path = os.path.join(self.dataset_dir, self.dataset_name, image_name)
            print('startint parsing {}'.format(img_path))
            src_img = cv2.imread(img_path)
            line = self.lines[image_name]
            ids, iscrowds, segs, img_width, img_height, img_id = self.parse_line(line)
            merge_mask = np.zeros([img_height, img_width], dtype=np.uint8)
            bboxes = []
            for id, seg in zip(ids, segs):
                src_mask = self.annToMask(seg, img_height, img_width)
                bbox = self.get_bbox(src_mask)
                bboxes.append(bbox)
                merge_mask = np.where(src_mask == 0, merge_mask, src_mask)

            dst_img = self.daw_mask(src_img, merge_mask)
            for id, bbox in zip(ids, bboxes):
                dst_img = self.draw_caption(dst_img, bbox, self.classes[int(id)], 1.0)

            save_path = os.path.join(self.result_save_dir, image_name)
            cv2.imwrite(save_path, dst_img)
            print('" ' + save_path + ' is saved!"')
        print("num_class:", len(self.classes))

    def get_label(self, label_dir, label_name):
        label_path = os.path.join(self.label_dir, self.label_name)
        with open(label_path, 'r') as load_f:
            load_dict = json.load(load_f)
        load_f.close()
        image_names = list(load_dict.keys())
        print("Total image:", len(image_names))
        random.shuffle(image_names)

        new_dict = {}
        for key in image_names:
            new_dict[key] = load_dict.get(key)
        return new_dict


    def get_bbox(self, mask):
        h_idx, w_idx = np.where(mask == 1)
        x_min = w_idx.min()
        y_min = h_idx.min()
        x_max = w_idx.max()
        y_max = h_idx.max()
        bbox = np.array([x_min, y_min, x_max, y_max])
        return bbox

    def parse_line(self, line):

        img_width = int(line[0])
        img_height = int(line[1])
        img_id = int(line[2])
        seg_ids = line[3:]
        ids = []
        iscrowds = []
        segs = []
        for i in range(len(seg_ids) // 3):
            cat = seg_ids[i*3]
            iscrowd = seg_ids[i*3 + 1]
            seg = seg_ids[i*3 + 2]
            ids.append(cat)
            iscrowds.append(iscrowd)
            segs.append(seg)
        return ids, iscrowds, segs, img_width, img_height, img_id

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

    def get_classes(self, class_file):
        classes = []
        with open(class_file, 'r') as f:
            line = f.readline()
            while line:
                classes.append(line.rstrip('\n'))
                line = f.readline()
        return classes

    def draw_caption(self, img, b, caption, score):
        num = 0
        for ii in caption:
            num += 1
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), ((0, 0, 255)), thickness=2)
        cv2.rectangle(img, (b[0], b[1]), (b[0] + 9 * (num + 5), b[1] + 10), (255, 0, 0), thickness=-1)
        cv2.putText(img, caption + ':' + str(score), (b[0], b[1] + 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(img, caption + ':' + str(score), (b[0], b[1] + 10), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 255), 1)
        return img

    def daw_mask(self, src, mask):
        color_mask = np.zeros_like(src)
        color_mask[:, :, 2] = 255
        mask = np.expand_dims(mask, axis=-1)
        mask = np.repeat(mask, 3, axis=-1)
        tem_img = (0.4 * src.astype(np.float32) + 0.6 * color_mask.astype(np.float32)).astype(np.uint8)
        dst_img = np.where(mask == 0, src, tem_img)
        return dst_img


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--save_show", default='save', type=str, choices=['save', 'show'],
                        help="save or show")
    parser.add_argument("--dataset_dir", default='/zpmc_disk/COCO', type=str,
                        help="the dir of image dataset")
    parser.add_argument("--dataset_name", default='val2017', type=str, help="the name of image dataset")
    parser.add_argument("--label_dir", default='/zpmc_disk/COCO/annotations',
                        type=str, help="the dir of .txt label")
    parser.add_argument("--label_name", default='instance_val2017_seg.json', type=str, help="the name of .txt label")
    parser.add_argument("--class_dir", default='/zpmc_disk/COCO/annotations', type=str,
                        help="the path to save generate label")
    parser.add_argument("--class_name", default='coco.name', type=str,
                        help="the name of saving generate label")
    parser.add_argument("--result_save_dir", default='/zpmc_disk/COCO/val_result_image_save', type=str, help="the name to classes")
    cfg = parser.parse_args()

    show_uav = zpmc_ShowReslut(cfg.dataset_dir, cfg.dataset_name, cfg.label_dir,
                               cfg.label_name, cfg.class_dir, cfg.class_name, cfg.result_save_dir)

    if cfg.save_show == 'save':
        show_uav.save_result()
    else:
        show_uav.show_result()