import numpy as np
import cv2, os, argparse, json

class ann_show:
    def __init__(self, label_path, class_name_path, result_save_dir, image_dir):
        self.label_path = label_path
        self.class_name_path = class_name_path
        self.result_save_dir = result_save_dir
        self.image_dir = image_dir
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
        images_name = load_dict.keys()
        counter = 0
        
        if not os.path.exists(self.result_save_dir):
            os.makedirs(self.result_save_dir)
        
        for image_name in images_name:
            image_path = os.path.join(self.image_dir, image_name)
            img = cv2.imread(image_path)
            
            # imgray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # ret,thresh=cv2.threshold(imgray,127,255,0)
            # contours,hierarchy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            # print(type(contours))
            # print(type(contours[0]))
            anns = load_dict[image_name]
            counter += 1
            for ann in anns:
                bbox = ann["bbox"]
                x = int(bbox[0])
                y = int(bbox[1])
                w = int(bbox[2])
                h = int(bbox[3])
                segmentation = ann["segmentation"]
                contours = [np.array(seg,dtype=np.int32).reshape(-1, 1, 2) for seg in segmentation]
                img=cv2.drawContours(img,contours,-1,(0,255,0),2)  # img为三通道才能显示轮廓
                img=cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 1)
                
            #     x = int(ann[0])
            #     y = int(ann[1])
            #     cls = int(ann[2])
            #     index = int(ann[3])
            #     cv2.circle(img, (x, y), 8, (0,0,255), -1)
            #     cv2.putText(img, self.classes[cls], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            cv2.imwrite(os.path.join(self.result_save_dir, str(counter)+image_name), img)
            print('Saving {}'.format(os.path.join(self.result_save_dir, str(counter)+image_name)))
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_path", default='/root/code/AiEngineering/02-Segmetation/yolact/result_temp/ceLL_guide_val_v1_seg.json', type=str, help=" ")
    parser.add_argument("--class_name_path", default='/root/code/AiEngineering/02-Segmetation/yolact/result_temp/cell_guide.name', type=str, help=" ")
    parser.add_argument("--result_save_dir", default='/root/code/AiEngineering/02-Segmetation/yolact/result_temp/show', type=str, help=" ")
    parser.add_argument("--image_dir", default='/root/code/AiEngineering/02-Segmetation/yolact/result_temp/val_crop_images', type=str, help=" ")
    
    cfg = parser.parse_args()  
    ann_show(cfg.label_path, cfg.class_name_path, cfg.result_save_dir, cfg.image_dir)  