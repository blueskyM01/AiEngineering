import cv2, os, argparse, json
import numpy as np
import random, copy, shutil
from collections import defaultdict


class crop_image:
    def __init__(self, label_path, class_name_path, crop_dataset_save_dir, train_val, crop_label_name, crop_class_name, crop_num):
        self.label_path = label_path
        self.class_name_path = class_name_path
        self.crop_dataset_save_dir = crop_dataset_save_dir
        self.train_val = train_val
        self.crop_label_name = crop_label_name
        self.crop_class_name = crop_class_name
        self.crop_num = crop_num
        self.crop_w = 256
        self.crop_h = 256
        self.images_save_dir = os.path.join(self.crop_dataset_save_dir, 'images', self.train_val)
        self.annotation_save_dir = os.path.join(self.crop_dataset_save_dir, 'annotation')
        
        if not os.path.exists(self.crop_dataset_save_dir):
            os.makedirs(self.crop_dataset_save_dir)
        if not os.path.exists(self.images_save_dir):
            os.makedirs(self.images_save_dir)
        if not os.path.exists(self.annotation_save_dir):
            os.makedirs(self.annotation_save_dir)
            
        self.classes = self.get_classes()
        self.generate_clip()
        
    def get_classes(self):
        classes = []
        shutil.copyfile(self.class_name_path, os.path.join(self.annotation_save_dir, self.crop_class_name))
        with open(self.class_name_path, 'r') as f:
            line = f.readline()
            while line:
                classes.append(line.rstrip('\n'))
                line = f.readline()
        return classes
    
    def get_label(self):
            with open(self.label_path, 'r') as load_f:
                load_dict = json.load(load_f)
            load_f.close()
            # image_names = list(load_dict.keys())
            # random.shuffle(lines)
            return load_dict  
          
    def generate_clip(self):
        load_dict = self.get_label()
        images_path = list(load_dict.keys())
        random.shuffle(images_path)
        counter = 0
        
        new_anns = defaultdict(list)  # 创建一个字典，值的type是list
        
        for image_path in images_path:
            img = cv2.imread(image_path)
            img_name = image_path.split('/')[-1].split('.')[0]
            h, w, c = img.shape
            anns = load_dict[image_path]
            counter += 1
            for ik, ann in enumerate(anns):
                x = int(ann[0])
                y = int(ann[1])
                cls = int(ann[2])
                index = int(ann[3])
                
                for idx in range(self.crop_num):
                    rx = random.randint(50, self.crop_w - 50)
                    ry = random.randint(50, self.crop_h - 50)
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
                        
                    clip_image = img[y0:y1, x0:x1, :]
                    new_x = x - x0
                    new_y = y - y0
                    new_cls = cls
                    new_index = index
                    new_img_name = img_name + '_cls_' + str(cls) + '_'+ str(idx) + '.jpg'
                    new_anns[os.path.join(self.images_save_dir, new_img_name)].append([new_x, new_y, new_cls, new_index])
                    copy_anns = copy.deepcopy(anns) 
                    copy_anns.pop(ik)
                    for elel in copy_anns:
                        x_t = int(elel[0])
                        y_t = int(elel[1])
                        cls_t = int(elel[2])
                        index_t = int(elel[3])
                        if x_t > x0 and y_t > y0 and x_t < x1 and y_t < y1:
                            new_anns[os.path.join(self.images_save_dir, new_img_name)].append([x_t - x0, y_t - y0, cls_t, index_t])
                            # print(os.path.join(self.crop_images_save_dir, new_img_name))
                    
                    cv2.imwrite(os.path.join(self.images_save_dir, new_img_name), clip_image)
                    print('Saving {}'.format(os.path.join(self.images_save_dir, new_img_name)))
            
            
        with open(os.path.join(self.annotation_save_dir, self.crop_label_name), "w") as f:
            json.dump(new_anns, f)    

        print('num_orignal images: {}'.format(len(images_path)))
        print('generate {} crop images:'.format(len(new_anns.keys())))
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_path", default='/root/code/AI-Note-Demo/01-ObjectDetection/CenterNet/code/img_out/cell_guide_contanier_corner/CellGuideContainerCorner_train01.json', type=str, help=" ")
    parser.add_argument("--class_name_path", default='/root/code/AI-Note-Demo/01-ObjectDetection/CenterNet/code/img_out/cell_guide_contanier_corner/CellGuideContainerCorner.txt', type=str, help=" ")
    parser.add_argument("--crop_dataset_save_dir", default='/root/code/AI-Note-Demo/01-ObjectDetection/CenterNet/code/img_out/Crop_CellGuideContainerCorner', type=str, help=" ")
    parser.add_argument("--train_val", default='train_v1', type=str, help=" ")
    parser.add_argument("--crop_label_name", default='Crop_CellGuideContainerCorner_train01.json', type=str, help=".json file")
    parser.add_argument("--crop_class_name", default='crop_classes.txt', type=str, help=" ")
    parser.add_argument("--crop_num", default=10, type=int, help=" ")
    
    cfg = parser.parse_args()  
    crop_image(cfg.label_path, cfg.class_name_path, cfg.crop_dataset_save_dir, cfg.train_val, cfg.crop_label_name, cfg.crop_class_name, cfg.crop_num)         
    

