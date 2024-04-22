import numpy as np
import cv2, os, argparse, json
from collections import defaultdict

'''
功能： Generate train.txt/val.txt/test.txt files One line for one image, in the format like：
      image_index, image_name, img_width, img_height, img_id cat1, box_1, cat2, box_2, ... cat_n, box_n.
      Box_x format: label_index x_min y_min x_max y_max.
                    (The origin of coordinates is at the left top corner, left top => (xmin, ymin), right bottom => (xmax, ymax).)
      image_index: is the line index which starts from zero.
      label_index: is in range [0, class_num - 1].
      For example:
      0 a.jpg 1920 1080 84765 0 453 369 473 391 1 588 245 608 268
      1 b.jpg 1920 1080 74532 1 466 403 485 422 2 793 300 809 320
'''



class zpmc_GenerateTrainLabel:
    def __init__(self, ann_dir, ann_name, label_save_dir, label_save_name):
        self.ann_dir = ann_dir
        self.ann_name = ann_name
        self.label_save_dir = label_save_dir
        self.label_save_name = label_save_name
        if not os.path.exists(self.label_save_dir):
            os.mkdir(self.label_save_dir)

    def get_ann(self):
        name_box_id = defaultdict(list)  # 创建一个字典，值的type是list

        with open(os.path.join(self.ann_dir, self.ann_name), 'r') as load_f:
            load_dict = json.load(load_f)
        load_f.close()

        for img_name in load_dict.keys():
            
            anns = load_dict[img_name]
            for ann in anns:
                box = ann['bbox']
                cat = ann['category_id']
                new_cat_id = cat -1
                name_box_id[img_name].append([box, -1, -1, new_cat_id, -1])
        return name_box_id

    def generate_train_label(self):
        name_box_id = self.get_ann()
        f = open(os.path.join(self.label_save_dir, self.label_save_name), 'w')
        counter = 0
        for key in name_box_id.keys():
            elem = []
            elem.append(counter)  # image_index
            counter += 1
            elem.append(key)  # image_name

            # img = cv2.imread(os.path.join(self.dataset_dir, self.dataset_name, key.split('/')[-1]))
            # width = img.shape[1]
            # height = img.shape[0]
            width = name_box_id[key][0][1]
            height = name_box_id[key][0][2]
            img_id = name_box_id[key][0][4]
            elem.append(width)
            elem.append(height)
            elem.append(img_id)

            boxes = []
            box_infos = name_box_id[key]
            for info in box_infos:
                x_min = info[0][0]
                y_min = info[0][1]
                x_max = x_min + info[0][2]
                y_max = y_min + info[0][3]
                boxes.append(info[3])
                boxes.append(x_min)
                boxes.append(y_min)
                boxes.append(x_max)
                boxes.append(y_max)

            elem = elem + boxes
            for index in range(len(elem)):
                if index == 1:
                    f.write(elem[index] + ' ')

                elif index == (len(elem) - 1):
                    f.write(str(round(elem[index], 2)) + '\n')

                else:
                    f.write(str(round(elem[index], 2)) + ' ')
            print('num:', counter)

        print('Genrate"', os.path.join(self.label_save_dir, self.label_save_name), '"')
        f.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_dir", default='/root/code/AiEngineering/02-Segmetation/yolact/result_temp',
                        type=str, help="the dir of josn label")
    parser.add_argument("--ann_name", default='ceLL_guide_val_v1_seg.json', type=str, help="the name of josn label")
    parser.add_argument("--label_save_dir", default='/root/code/AiEngineering/02-Segmetation/yolact/result_temp', type=str,
                        help="the path to save generate label")
    parser.add_argument("--label_save_name", default='cell_guide_eval.txt', type=str, help="the name of saving generate label")
    cfg = parser.parse_args()

    uav = zpmc_GenerateTrainLabel(cfg.ann_dir, cfg.ann_name, cfg.label_save_dir, cfg.label_save_name)
    uav.generate_train_label()