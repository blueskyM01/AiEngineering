from pycocotools.coco import COCO
import numpy as np
import cv2, os, argparse, json
from collections import defaultdict



class zpmc_GenerateTrainLabel:
    def __init__(self, ann_dir, ann_name, label_save_dir, label_save_name, class_names):
        self.ann_dir = ann_dir
        self.ann_name = ann_name
        self.label_save_dir = label_save_dir
        self.label_save_name = label_save_name
        self.class_names = class_names
        
        if not os.path.exists(self.label_save_dir):  # os模块判断并创建
            os.mkdir(self.label_save_dir)

    def get_ann(self):
        name_seg_id = defaultdict(list)
        annFile = os.path.join(self.ann_dir, self.ann_name)
        coco = COCO(annFile)
        CatIds = sorted(coco.getCatIds()) 
        cats = coco.loadCats(CatIds)
        source_cat_names = {}
        new_cat_names = {}
        old_map_new_cat = {}
        num_class = len(CatIds)
        new_class_id = [i for i in range(num_class)]

        for cat in cats:
            source_cat_names[cat['id']] = cat['name']

        for new_id, source_cat in enumerate(sorted(source_cat_names.keys())):
            old_map_new_cat[source_cat] = new_id
            new_cat_names[new_id] = source_cat_names[source_cat]

        f_name = open(os.path.join(self.label_save_dir, self.class_names), 'w')
        for new_key in sorted(new_cat_names.keys()):
            f_name.write(new_cat_names[new_key] + '\n')
        f_name.close()

        imgIds = []
        for CatId in CatIds:
            imgIds.extend(list(coco.getImgIds(catIds=[CatId])))
        imgIds = list(set(imgIds))

        imgs = coco.loadImgs(imgIds)

        for idx, img in enumerate(imgs):
            annIds = coco.getAnnIds(imgIds=img['id'])
            anns = coco.loadAnns(annIds)
            for i, ann in enumerate(anns):
                image_name = img['file_name'].split('/')[-1]
                seg = ann['segmentation'][0]
                text = ann['attributes']['character']
                name_seg_id[image_name].append([text, seg])
            print('Process the ', idx+1, 'th image')

        return name_seg_id

    def generate_train_label(self):
        name_seg_id = self.get_ann()
        with open(os.path.join(self.label_save_dir, self.label_save_name), "w") as f:
            json.dump(name_seg_id, f)

        print('Genrate"', os.path.join(self.label_save_dir, self.label_save_name), '" and "',os.path.join(self.label_save_dir, self.class_names), '"')
        f.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_dir", default='/root/code/yangjianbing/dataset/01-changqiao/annotations/',
                        type=str, help="the dir of josn label")
    parser.add_argument("--ann_name", default='instances_Validation.json', type=str, help="the name of josn label")
    parser.add_argument("--label_save_dir", default='/root/code/yangjianbing/dataset/01-changqiao/annotations/', type=str,
                        help="the path to save generate label")
    parser.add_argument("--label_save_name", default='val.json', type=str, help="the name of saving generate label")
    parser.add_argument("--class_names", default='container_code.name', type=str, help="the name to classes")
    cfg = parser.parse_args()

    uav = zpmc_GenerateTrainLabel(cfg.ann_dir, cfg.ann_name, cfg.label_save_dir, cfg.label_save_name, cfg.class_names)
    uav.generate_train_label()