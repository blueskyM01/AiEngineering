from pycocotools.coco import COCO
import numpy as np
import cv2, os, argparse, json
from collections import defaultdict

class zpmc_GenerateTrainLabel:
    def __init__(self, ann_dir, ann_name, label_save_dir, class_names, train_val):
        self.ann_dir = ann_dir
        self.ann_name = ann_name
        self.label_save_dir = label_save_dir
        self.class_names = class_names
        self.train_val = train_val

    def get_ann(self):
        name_seg_id = defaultdict(list)  # 创建一个字典，值的type是list

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

        for idx, img in enumerate(imgs):
            # coco中每个标注实例都对应一个id，如果一张图像中有多个实例，也就有多个ann
            # annIds = [id1, id2, ....]
            annIds = coco.getAnnIds(imgIds=img['id'])

            # 使用给定的annIds加载annotation
            # anns = [{}, {}], {} = {'id': 5, 'image_id': 5, 'category_id': 1, 'segmentation': [], 'area': 109903.41049999993,
            #                        'bbox': [2614.17, 188.47, 368.05, 298.61], 'iscrowd': 0, 'attributes': {'occluded': False}}
            anns = coco.loadAnns(annIds)
            for i, ann in enumerate(anns):
                img_id = ann['image_id']
                seg = ann['segmentation']
                iscowd = ann['iscrowd']
                cat = ann['category_id']
                new_cat_id = old_map_new_cat[cat]
                name_seg_id[img['file_name'].split('/')[-1]].extend([img['width'], img['height'], new_cat_id, seg])
                # if i == 0:
                #     name_seg_id[img['file_name'].split('/')[-1]].extend([img['width'], img['height'], img_id, new_cat_id, iscowd, seg])
                # else:
                #     name_seg_id[img['file_name'].split('/')[-1]].extend([new_cat_id, iscowd, seg])
            print('Process the ', idx+1, 'th image')
        return name_seg_id

    def generate_train_label(self):
        if not os.path.exists(os.path.join(self.label_save_dir, self.train_val)):
            os.makedirs(os.path.join(self.label_save_dir, self.train_val))
        name_seg_id = self.get_ann()
        for img_name in name_seg_id.keys():
            segs = name_seg_id[img_name]
            label_name = img_name.split('.')[0]+'.txt'
            f = open(os.path.join(self.label_save_dir, self.train_val, label_name), 'w')
            for i in range(len(segs) // 4):
                img_w = segs[i*4+0]
                img_h = segs[i*4+1]
                cls = segs[i*4+2]
                seg = segs[i*4+3]
                
                line_ele = ''
                for idx in range(len(seg[-1]) // 2):
                    ele_x = seg[-1][idx*2+0] / img_w
                    ele_y = seg[-1][idx*2+1] / img_h
                    ele_x_s = '{:.3f}'.format(ele_x)
                    ele_y_s = '{:.3f}'.format(ele_y)
                    line_ele += str(cls) + " " + ele_x_s +  " " + ele_y_s
                line_ele += '\n'
                f.write(line_ele)
                # print("img_w:%d, img_h:%d, cls:%d" % (img_w, img_h, cls))
                # print(seg)
                
                
        
        # with open(os.path.join(self.label_save_dir, self.label_save_name), "w") as f:
        #     json.dump(name_seg_id, f)

        # print('Genrate"', os.path.join(self.label_save_dir, self.label_save_name), '" and "', \
        #       os.path.join(self.label_save_dir, self.class_names), '"')
        # f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_dir", default='/zpmc_disk/COCO/annotations',
                        type=str, help="the dir of josn label")
    parser.add_argument("--ann_name", default='instances_val2017.json', type=str, help="the name of josn label")
    parser.add_argument("--label_save_dir", default='/zpmc_disk/COCO/annotations', type=str,
                        help="the path to save generate label")
    parser.add_argument("--class_names", default='coco.name', type=str, help="the name to classes")
    parser.add_argument("--train_val", default='val', type=str, help="train or val")
    cfg = parser.parse_args()

    uav = zpmc_GenerateTrainLabel(cfg.ann_dir, cfg.ann_name, cfg.label_save_dir, cfg.class_names, cfg.train_val)
    uav.generate_train_label()