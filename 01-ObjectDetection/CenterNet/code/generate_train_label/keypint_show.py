import numpy as np
import cv2, os, argparse, json
import random
from collections import defaultdict
class ann_show:
    def __init__(self, label_path, class_name_path, result_save_dir):
        self.label_path = label_path
        self.class_name_path = class_name_path
        self.result_save_dir = result_save_dir
        self.classes = self.get_classes()
        self.decode()
        
    def get_classes(self):
        classes = []
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
          
    def decode(self):
        load_dict = self.get_label()
        images_path = load_dict.keys()
        counter = 0
        
        if not os.path.exists(self.result_save_dir):
            os.makedirs(self.result_save_dir)
        
        for image_path in images_path:
            image_name = image_path.split('/')[-1]
            img = cv2.imread(image_path)
            anns = load_dict[image_path]
            counter += 1
            for ann in anns:
                x = int(ann[0])
                y = int(ann[1])
                cls = int(ann[2])
                index = int(ann[3])
                cv2.circle(img, (x, y), 8, (0,0,255), -1)
                cv2.putText(img, self.classes[cls], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            cv2.imwrite(os.path.join(self.result_save_dir, str(counter)+image_name), img)
            print('Saving {}'.format(os.path.join(self.result_save_dir, str(counter)+image_name)))
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_path", default='/root/code/AI-Note-Demo/01-ObjectDetection/CenterNet/code/img_out/train01.json', type=str, help=" ")
    parser.add_argument("--class_name_path", default='/root/code/AI-Note-Demo/01-ObjectDetection/CenterNet/code/img_out/cell_guide_classes.txt', type=str, help=" ")
    parser.add_argument("--result_save_dir", default='/root/code/AI-Note-Demo/01-ObjectDetection/CenterNet/code/img_out/results_images', type=str, help=" ")
    
    cfg = parser.parse_args()  
    ann_show(cfg.label_path, cfg.class_name_path, cfg.result_save_dir)         
    
# lines = get_label('/root/code/AI-Note-Demo/01-ObjectDetection/CenterNet/code/generate_train_label', 'person_keypoints_val2017.txt')

# image_keypoints = decode(lines)
# point_size = 2
# thickness = -1
# counter = 0
# for key in image_keypoints.keys():
#     counter += 1
#     img = cv2.imread(key)
#     keypoints_set = image_keypoints[key]
#     for keypoints in keypoints_set:
#         color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
#         for keypoint in keypoints:
#             x = keypoint[0]
#             y = keypoint[1]
#             v = keypoint[2]
#             if(v == 2):
#                 show_img = cv2.circle(img, (x, y), point_size, color, thickness)
#     # if counter % 100 == 0:
#     #     cv2.imwrite('/root/code/AI-Note-Demo/01-ObjectDetection/CenterNet/code/img_out/' + str(counter) +'.png', show_img)

