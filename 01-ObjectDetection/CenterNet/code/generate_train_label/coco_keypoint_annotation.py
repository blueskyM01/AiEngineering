from pycocotools.coco import COCO
import numpy as np
import cv2, os, argparse, json
from collections import defaultdict

'''
1、训练标签
    dict={'image1_path': [[x1,y1,cls,index], [x2,y2,cls,index], ...], 'image2_path': [[x1,y1,cls,index], [x2,y2,cls,index], ...]}

    Description：
    key: image_path;
    value: list，where, the list container several sub lists. The elements of sub list are x , y, cls and index. Note that 1) cls start from 0，if there are 10 classes in dataset，the cls is 0, 1, 2, 3, 4, 5, 6, 7, 8, 9; 2) index only avaliable in container corener point detection. Or use -1 to respresent the index

    For example (train.json): 
    {"train01/image_0000000432.jpeg": [["4", "2", 0, "0"]], 
     "train01/image_0000000523.jpeg": [["4", "2", 0, "1"], ["2", "7", 0, "0"], ["1", "9", 0, "0"], ["3", "0", 0, "0"], ["2", "4", 0, "0"], ["2", "2", 0, "0"], ["2", "7", 0, "0"], ["5", "1", 0, "0"], ["5", "5", 0, "0"], ["5", "3", 0, "0"], ["4", "3", 0, "0"], ["4", "4", 0, "0"], ["4", "8", 0, "0"], ["4", "8", 0, "0"], ["3", "9", 0, "0"], ["2", "7", 0, "0"], ["2", "1", 0, "0"]], 
     "train01/image_0000000524.jpeg": [["4", "2", 0, "1"]], 
     "train01/image_0000000525.jpeg": [["4", "2", 0, "1"]], 
     "train01/image_0000000526.jpeg": [["4", "2", 0, "1"]], 
     "train01/image_0000000527.jpeg": [["4", "2", 0, "1"]], 
     "train01/image_0000000528.jpeg": [["4", "2", 0, "1"]], 
     "train01/image_0000000529.jpeg": [["4", "2", 0, "1"]], 
     "train01/image_0000000530.jpeg": [["4", "2", 0, "1"]], 
     "train01/image_0000000531.jpeg": [["4", "2", 0, "1"]]}

2、类别标签 (`.txt` file)，如果是10个类
cls0
cls1
cls2
cls3
cls4
cls5
cls6
cls7
cls8
cls9
cls10
cls11
cls12
cls13
cls14
cls15
cls16

For example (class.txt):
nose
left_eye
right_eye
left_ear
right_ear
left_shoulder
right_shoulder
left_elbow
right_elbow
left_wrist
right_wrist
left_hip
right_hip
left_knee
right_knee
left_ankle
right_ankle
'''



class zpmc_GenerateTrainLabel:
    def __init__(self, dataset_dir, ann_dir, ann_name, label_save_dir, label_save_name, class_names):
        self.ann_dir = ann_dir
        self.ann_name = ann_name
        self.label_save_dir = label_save_dir
        self.label_save_name = label_save_name
        self.class_names = class_names
        self.dataset_dir = dataset_dir

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
        ccls = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]

        for new_key in ccls:
            f_name.write(new_key + '\n')
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
                keypoints = ann['keypoints']
                cat = ann['category_id']
                img_id = ann['image_id']
                new_cat_id = old_map_new_cat[cat]
                name_box_id[img['file_name'].split('/')[-1]].append([keypoints, img['width'], img['height'], new_cat_id, img_id])
        return name_box_id

    def generate_train_label(self):
        name_box_id = self.get_ann()
        key_point_anns = defaultdict(list)  # 创建一个字典，值的type是list
        # f = open(os.path.join(self.label_save_dir, self.label_save_name), 'w')
        counter = 0
        for key in name_box_id.keys():
            # elem = []
            counter += 1
            # elem.append(os.path.join(self.images_dir, key))  # image_name

            keypoints_list = []
            box_infos = name_box_id[key]
            for info in box_infos:
                # x_min = int(info[0][0])
                # y_min = int(info[0][1])
                # x_max = int(x_min + info[0][2])
                # y_max = int(y_min + info[0][3])
                keypoints = info[0]
                for i in range(len(keypoints)//3):
                    x = keypoints[i*3+0]
                    y = keypoints[i*3+1]
                    v = keypoints[i*3+2]
                    if v == 2:
                        key_point_anns[os.path.join(self.dataset_dir, key)].append([int(x), int(y), i, -1])

            print('num:', counter)

        with open(os.path.join(self.label_save_dir, self.label_save_name), "w") as f:
            json.dump(key_point_anns, f)   
        
        print('Genrate"', os.path.join(self.label_save_dir, self.label_save_name), '" and "', \
              os.path.join(self.label_save_dir, self.class_names), '"')
        # f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default='/root/code/dataset/val2017', type=str, help="the dir of image dataset")
    # parser.add_argument("--dataset_name", default='COCO', type=str, help="the name of image dataset")
    parser.add_argument("--ann_dir", default='/root/code/dataset/annotations',
                        type=str, help="the dir of josn label")
    parser.add_argument("--ann_name", default='person_keypoints_val2017.json', type=str, help="the name of josn label")
    parser.add_argument("--label_save_dir", default='/root/code/AI-Note-Demo/01-ObjectDetection/CenterNet/code/img_out', type=str,
                        help="the path to save generate label")
    parser.add_argument("--label_save_name", default='person_keypoints_val2017.json', type=str, help="the name of saving generate label")
    parser.add_argument("--class_names", default='coco_keypoint_classes.txt', type=str, help="the name to classes")
    cfg = parser.parse_args()

    uav = zpmc_GenerateTrainLabel(cfg.dataset_dir, cfg.ann_dir, cfg.ann_name, cfg.label_save_dir, cfg.label_save_name, cfg.class_names)
    uav.generate_train_label()