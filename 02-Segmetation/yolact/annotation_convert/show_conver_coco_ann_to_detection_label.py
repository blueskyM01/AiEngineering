import numpy as np
import cv2, os, argparse
import random

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

    def show_result(self):
        for line in self.lines:
            num_gt, img_path, annotations, img_width, img_height = self.parse_line(line)
            image = cv2.imread(os.path.join(self.dataset_dir, self.dataset_name, img_path))
            h, w = image.shape[0:2]
            for box in annotations:
                label = int(box[4])
                x0 = int(box[0])
                y0 = int(box[1])
                x1 = int(box[2])
                y1 = int(box[3])
                cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 8)
                cv2.putText(image, self.classes[label], (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 8)
            image = cv2.resize(image, (int(w // 8), int(h // 8)))
            cv2.imshow('image_name', image)
            cv2.waitKey(2000)
        print("num_class:", len(self.classes))

    def save_result(self):
        for line in self.lines:
            num_gt, img_path, annotations, img_width, img_height = self.parse_line(line)
            # img_path = img_path.split('/')[-1]
            image = cv2.imread(os.path.join(self.dataset_dir, self.dataset_name, img_path))

            for box in annotations:
                label = int(box[4])
                x0 = int(box[0])
                y0 = int(box[1])
                x1 = int(box[2])
                y1 = int(box[3])
                cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 4)
                cv2.putText(image, self.classes[label], (x0, y0+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # cv2.imshow('image_name', image)
            # cv2.waitKey(2000)

            if not os.path.exists(self.result_save_dir):
                os.makedirs(self.result_save_dir)

            image_path = os.path.join(self.result_save_dir, img_path)
            cv2.imwrite(image_path, image)
            print('" ' + image_path + ' is saved!"')
        print("num_class:", len(self.classes))


    def get_label(self, label_dir, label_name):
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
        label_path = os.path.join(self.label_dir, self.label_name)
        lines = []
        with open(label_path, 'r') as f:
            line = f.readline()
            while line:
                lines.append(line.rstrip('\n').split(' '))
                line = f.readline()
        random.shuffle(lines)
        return lines

    def parse_line(self, line):
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

    def get_classes(self, class_file):
        classes = []
        with open(class_file, 'r') as f:
            line = f.readline()
            while line:
                classes.append(line.rstrip('\n'))
                line = f.readline()
        return classes

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--save_show", default='save', type=str, choices=['save', 'show'],
                        help="save or show")
    parser.add_argument("--dataset_dir", default='/zpmc_disk/COCO', type=str,
                        help="the dir of image dataset")
    parser.add_argument("--dataset_name", default='val2017', type=str, help="the name of image dataset")
    parser.add_argument("--label_dir", default='/zpmc_disk/COCO/annotations',
                        type=str, help="the dir of .txt label")
    parser.add_argument("--label_name", default='instance_val2017.txt', type=str, help="the name of .txt label")
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