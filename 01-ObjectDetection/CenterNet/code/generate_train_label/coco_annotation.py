from pycocotools.coco import COCO
import numpy as np
import cv2, os, argparse
from collections import defaultdict

'''
功能： Generate train.txt/val.txt/test.txt files One line for one image, in the format like：
      image_index, image_absolute_path, img_width, img_height, cat1, box_1, cat2, box_2, ... catn, box_n.
      Box_x format: label_index x_min y_min x_max y_max.
                    (The origin of coordinates is at the left top corner, left top => (xmin, ymin), right bottom => (xmax, ymax).)
      image_index: is the line index which starts from zero.
      label_index: is in range [0, class_num - 1].
      For example:
      0 xxx/xxx/a.jpg 1920 1080 0 453 369 473 391 1 588 245 608 268
      1 xxx/xxx/b.jpg 1920 1080 1 466 403 485 422 2 793 300 809 320
'''



class zpmc_GenerateTrainLabel:
    def __init__(self, ann_dir, ann_name, label_save_dir, label_save_name, class_names, images_dir):
        self.ann_dir = ann_dir
        self.ann_name = ann_name
        self.label_save_dir = label_save_dir
        self.label_save_name = label_save_name
        self.class_names = class_names
        self.images_dir = images_dir

    def get_ann(self):
        name_box_id = defaultdict(list)  # 创建一个字典，值的type是list

        annFile = os.path.join(self.ann_dir, self.ann_name)

        # initialize COCO api for instance annotations
        coco = COCO(annFile)

        # display COCO categories and supercategories
        CatIds = sorted(coco.getCatIds())  # 获得满足给定过滤条件的category的id

        # 使用指定的id加载category
        cats = coco.loadCats(CatIds)


        source_cat_names = {}
        new_cat_names = {}
        old_map_new_cat = {}
        num_class = len(CatIds)
        new_class_id = [i for i in range(num_class)]

        # 将原ann中种类的id与name存成dict，id为key， name为value，如下：
        # {1: 'person', 2: 'car', 3: 'scarf', 4: 'schoolbag'}
        for cat in cats:
            source_cat_names[cat['id']] = cat['name']

        # old_map_new_cat: 将ann中种类的id按照[0, 1, 2, 3, ....]排序，以原id为key，新id为value，如下：
        # {1: 0, 2: 1, 3: 2, 4: 3}
        # new_cat_names: 新id对应ann中的种类name，如下：
        # {0: 'person', 1: 'car', 2: 'scarf', 3: 'schoolbag'}
        for new_id, source_cat in enumerate(sorted(source_cat_names.keys())):
            old_map_new_cat[source_cat] = new_id
            new_cat_names[new_id] = source_cat_names[source_cat]

        # save ".name" file
        f_name = open(os.path.join(self.label_save_dir, self.class_names), 'w')
        for new_key in sorted(new_cat_names.keys()):
            f_name.write(new_cat_names[new_key] + '\n')
        f_name.close()

        # 找出所有category_id的image_id, 参数没有给定的话，指的是数据集中所有图像id
        # image_id = [1, 2, 3, 4, 5,......]
        imgIds = []
        for CatId in CatIds:
            imgIds.extend(list(coco.getImgIds(catIds=[CatId])))
        imgIds = list(set(imgIds))

        # 使用给定imgIds加载image
        # imgs = [{}, {}, ....], {} = {'id': 1, 'width': 4000, 'height': 3000, 'file_name': 'MVIMG_20201022_095333.jpg',
        #                              'license': 0, 'flickr_url': '', 'coco_url': '', 'date_captured': 0}
        imgs = coco.loadImgs(imgIds)

        for img in imgs:
            # img_path = os.path.join(self.dataset_dir, self.dataset_name, 'image', img['file_name'].split('/')[-1])
            # img_src = cv2.imread(img_path)

            # coco中每个标注实例都对应一个id，如果一张图像中有多个实例，也就有多个ann
            # annIds = [id1, id2, ....]
            annIds = coco.getAnnIds(imgIds=img['id'])

            # 使用给定的annIds加载annotation
            # anns = [{}, {}], {} = {'id': 5, 'image_id': 5, 'category_id': 1, 'segmentation': [], 'area': 109903.41049999993,
            #                        'bbox': [2614.17, 188.47, 368.05, 298.61], 'iscrowd': 0, 'attributes': {'occluded': False}}
            anns = coco.loadAnns(annIds)
            for ann in anns:
                box = ann['bbox']
                cat = ann['category_id']
                img_id = ann['image_id']
                new_cat_id = old_map_new_cat[cat]
                name_box_id[img['file_name'].split('/')[-1]].append([box, img['width'], img['height'], new_cat_id, img_id])
        return name_box_id

    def generate_train_label(self):
        name_box_id = self.get_ann()
        f = open(os.path.join(self.label_save_dir, self.label_save_name), 'w')
        counter = 0
        for key in name_box_id.keys():
            elem = []
            counter += 1
            elem.append(os.path.join(self.images_dir, key))  # image_name

            boxes = []
            box_infos = name_box_id[key]
            for info in box_infos:
                x_min = int(info[0][0])
                y_min = int(info[0][1])
                x_max = int(x_min + info[0][2])
                y_max = int(y_min + info[0][3])
                boxes.append(x_min)
                boxes.append(y_min)
                boxes.append(x_max)
                boxes.append(y_max)
                boxes.append(info[3])

            elem = elem + boxes
            for index in range(len(elem)):
                if index == 0:
                    f.write(elem[index] + ' ')
                elif index % 5 == 0:
                    if index == (len(elem) - 1):
                        f.write(str(elem[index]) + '\n')
                    else:
                        f.write(str(elem[index]) + ' ')
                else:
                    f.write(str(elem[index]) + ',')
            print('num:', counter)

        print('Genrate"', os.path.join(self.label_save_dir, self.label_save_name), '" and "', \
              os.path.join(self.label_save_dir, self.class_names), '"')
        f.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset_dir", default='/zpmc_disk', type=str, help="the dir of image dataset")
    # parser.add_argument("--dataset_name", default='COCO', type=str, help="the name of image dataset")
    parser.add_argument("--ann_dir", default='/root/code/dataset/annotations',
                        type=str, help="the dir of josn label")
    parser.add_argument("--ann_name", default='instances_val2017.json', type=str, help="the name of josn label")
    parser.add_argument("--label_save_dir", default='/root/code/dataset/annotations', type=str,
                        help="the path to save generate label")
    parser.add_argument("--label_save_name", default='instances_val2017.txt', type=str, help="the name of saving generate label")
    parser.add_argument("--class_names", default='coco_classes.txt', type=str, help="the name to classes")
    parser.add_argument("--images_dir", default='/root/code/dataset/val2017', type=str, help="dataset images dir")
    cfg = parser.parse_args()

    uav = zpmc_GenerateTrainLabel(cfg.ann_dir, cfg.ann_name, cfg.label_save_dir, cfg.label_save_name, cfg.class_names, cfg.images_dir)
    uav.generate_train_label()