from pycocotools.coco import COCO
import numpy as np
import cv2, os, argparse, json, random, shutil
from collections import defaultdict

class crop_images:
    def __init__(self, images_dir, ann_dir, ann_name, label_save_dir, label_save_name, images_save_name, class_names, crop_num, crop_w, crop_h):
        self.images_dir = images_dir
        self.ann_dir = ann_dir
        self.ann_name = ann_name
        self.label_save_dir = label_save_dir
        self.label_save_name = label_save_name
        self.class_names = class_names
        self.crop_num = crop_num
        self.crop_w = crop_w
        self.crop_h = crop_h
        self.crop_images_dir = os.path.join(self.label_save_dir, images_save_name)
        
        if not os.path.exists(self.label_save_dir):
            os.mkdir(self.label_save_dir)
        if not os.path.exists(self.crop_images_dir):
            os.mkdir(self.crop_images_dir)

    def get_ann(self):
        name_seg_id = defaultdict(list)  # 创建一个字典，值的type是list

        annFile = os.path.join(self.ann_dir, self.ann_name)

        # initialize COCO api for instance annotations
        coco = COCO(annFile)

        # display COCO categories and supercategories
        CatIds = sorted(coco.getCatIds(["cell_guide"]))  # 获得满足给定过滤条件的category的id
        # CatIds = sorted(coco.getCatIds())  # 获得满足给定过滤条件的category的id, 不指定的话，则给出所有category的id
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

        for idx, img in enumerate(imgs):
            # coco中每个标注实例都对应一个id，如果一张图像中有多个实例，也就有多个ann
            # annIds = [id1, id2, ....]
            annIds = coco.getAnnIds(imgIds=img['id'], catIds=CatIds)
            w = img['width']
            h = img['height']
            img_name = img['file_name'].split('/')[-1]
            src_img = cv2.imread(os.path.join(self.images_dir, img_name))

            # 使用给定的annIds加载annotation
            # anns = [{}, {}], {} = {'id': 5, 'image_id': 5, 'category_id': 1, 'segmentation': [], 'area': 109903.41049999993,
            #                        'bbox': [2614.17, 188.47, 368.05, 298.61], 'iscrowd': 0, 'attributes': {'occluded': False}}
            anns = coco.loadAnns(annIds)
            for i, ann in enumerate(anns):
                seg = ann['segmentation']
                iscowd = ann['iscrowd']
                bbbox = ann['bbox']
                cat = ann['category_id']
                new_cat_id = old_map_new_cat[cat] + 1
                
                x = int(bbbox[0]) + int(bbbox[2]) // 2
                y = int(bbbox[1]) + int(bbbox[3]) // 2
                obj_w = int(bbbox[2])
                obj_h = int(bbbox[3])
                
                if isinstance(seg,list):
                    for iidx in range(self.crop_num):
                        rx = random.randint(obj_w, self.crop_w - obj_w)
                        ry = random.randint(obj_h, self.crop_h - obj_h)
                        x0 = x - rx
                        y0 = y - ry
                        x1 = x0 + self.crop_w
                        y1 = y0 + self.crop_h
                        
                        # boundaru check
                        if x0 < 0:
                            x0 = 0
                            x1 = x0 + self.crop_w
                        if y0 < 0:
                            y0 = 0
                            y1 = y0 + self.crop_h
                        if x1 > w -1:
                            x1 = w
                            x0 = w - self.crop_w
                        if y1 > h -1:
                            y1 = h
                            y0 = h -self.crop_h
                            
                        clip_image = src_img[y0:y1, x0:x1, :]
                        new_seg = [(np.array(sub_seg).reshape(-1,2)-np.array([x0, y0])).reshape(-1).tolist() for sub_seg in seg]
                        new_bbbox = [int(bbbox[0])-x0, int(bbbox[1])-y0, int(bbbox[2]), int(bbbox[3])]
                        sub_dict = {'category_id': new_cat_id, 'bbox': new_bbbox, 'segmentation': new_seg, 'iscrowd': iscowd}
                        crop_name = str(iidx) + '_' + img_name
                        name_seg_id[crop_name].append(sub_dict)
                        cv2.imwrite(os.path.join(self.crop_images_dir, crop_name), clip_image)
                        print('save {}!'.format(os.path.join(self.crop_images_dir, crop_name)))


            print('Process the ', idx+1, 'th image')
        return name_seg_id

    def generate_train_label(self):
        name_seg_id = self.get_ann()
        with open(os.path.join(self.label_save_dir, self.label_save_name), "w") as f:
            json.dump(name_seg_id, f)

        print('Genrate"', os.path.join(self.label_save_dir, self.label_save_name), '" and "', \
              os.path.join(self.label_save_dir, self.class_names), '"')
        f.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", default='/root/code/dataset/cell_guide/images/train_v1',
                        type=str, help="")
    parser.add_argument("--ann_dir", default='/root/code/dataset/cell_guide/annotations',
                        type=str, help="the dir of josn label")
    parser.add_argument("--ann_name", default='train_v1.json', type=str, help="the name of josn label")
    parser.add_argument("--label_save_dir", default='/root/code/AiEngineering/02-Segmetation/yolact/result_temp', type=str,
                        help="the path to save generate label")
    parser.add_argument("--label_save_name", default='ceLL_guide_train_v1_seg.json', type=str, help="the name of saving generate label")
    parser.add_argument("--images_save_name", default='train_crop_images', type=str, help="")
    parser.add_argument("--class_names", default='cell_guide.name', type=str, help="the name to classes")
    parser.add_argument("--crop_num", default=10, type=int, help=" ")
    parser.add_argument("--crop_w", default=256, type=int, help=" ")
    parser.add_argument("--crop_h", default=256, type=int, help=" ")
    cfg = parser.parse_args()

    uav = crop_images(cfg.images_dir, cfg.ann_dir, cfg.ann_name, cfg.label_save_dir, cfg.label_save_name, cfg.images_save_name, cfg.class_names, cfg.crop_num, cfg.crop_w, cfg.crop_h)
    uav.generate_train_label()
    