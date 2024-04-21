import os, json, argparse
import xml.dom.minidom
from collections import defaultdict
import math
import numpy as np
 
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

For example (class.txt):
aerop
bicyc
bird
boat
bottl
bus
car
cat
chair
cow
'''
 
class zpmc_GenerateTrainLabel:
    def __init__(self, dataset_dir, ann_dir, ann_name, label_save_dir, label_save_name, class_names):
        self.dataset_dir = dataset_dir
        self.ann_dir = ann_dir
        self.ann_name = ann_name
        self.label_save_dir = label_save_dir
        self.label_save_name = label_save_name
        self.class_names = class_names
        if not os.path.exists(self.label_save_dir):
            os.makedirs(self.label_save_dir)
        #获取 xml 文档对象
        self.domTree = xml.dom.minidom.parse(os.path.join(self.ann_dir, self.ann_name))
        #获得根节点
        self.rootNode = self.domTree.documentElement
        self.name_cat = self.generate_classes_label()
        self.generate_ann_label()
        
        # print('显示xml文档内容')
        # print(rootNode.toxml())
        
    def generate_classes_label(self):
        classes = self.rootNode.getElementsByTagName('meta')[0].getElementsByTagName('task')[0].getElementsByTagName('labels')[0].getElementsByTagName('label')
        num_classes = len(classes)
        name_cat = {}
        for i in range(num_classes):
            class_name = classes[i].getElementsByTagName('name')[0].childNodes[0].data
            name_cat[class_name] = i


        # save ".name" file
        f_name = open(os.path.join(self.label_save_dir, self.class_names), 'w')
        for key in name_cat.keys():
            f_name.write(key + '\n')
        f_name.close()
        return name_cat
    
    def generate_ann_label(self):
        images = self.rootNode.getElementsByTagName('image')
        num_images = len(images)
        
        anns = defaultdict(list)  # 创建一个字典，值的type是list

        for i in range(num_images):
            sub_noe = images[i]
            image_name = sub_noe.getAttribute('name').split('/')[-1]
            image_path = os.path.join(self.dataset_dir, image_name)
            points = sub_noe.getElementsByTagName('points')
            polygons = sub_noe.getElementsByTagName('polygon')
            
            num_points = len(points)
            num_polygons = len(polygons)
            
            if(num_points == 0) and (num_polygons == 0):
                print("********************** '{}' No label! **********************".format(image_path))
                
            else:
                print('image_path: {}'.format(image_path))
                elem = [] # [x, y, cls, index]
                index = -1
                if num_points != 0:
                    for j in range(num_points):
                        index = int(points[j].getElementsByTagName('attribute')[0].childNodes[0].data)
                        label = points[j].getAttribute('label')
                        points_xy = points[j].getAttribute('points')
                        points_xy = points_xy.split(';')
                        for xy in points_xy:
                            xy = xy.split(',')
                            x = int(float(xy[0]))
                            y = int(float(xy[1]))
                            anns[image_path].append([x, y, self.name_cat[label], int(index)])
                    
                if num_polygons != 0:
                    for k in range(num_polygons):
                        label = polygons[k].getAttribute('label')
                        points_xy = polygons[k].getAttribute('points').split(';')
                        contour = []
                        for xy in points_xy:
                            xy = xy.split(',')
                            x = int(float(xy[0]))
                            y = int(float(xy[1]))
                            contour.append([x,y])
                        
                        contour_np = np.array(contour)
                        contour_x = contour_np[:, 0]
                        contour_y = contour_np[:, 1]
                        
                        x_min = np.min(contour_x)
                        y_min = np.min(contour_y)
                        x_max = np.max(contour_x)
                        y_max = np.max(contour_y)
                        
                        if(index == 0 or index ==3):
                            xy_1 = [x_max, y_min]
                            xy_2 = [x_min, y_max]
                            corner = self.generate_marker_corner_point(contour, index, xy_1, xy_2)
                        elif(index == 1 or index ==2):
                            xy_1 = [x_min, y_min]
                            xy_2 = [x_max, y_max]
                            corner = self.generate_marker_corner_point(contour, index, xy_1, xy_2)
                        anns[image_path].append([corner[0], corner[1], self.name_cat[label], int(index)])
                        
                        print('label:', label)
                        print("polygons: ", points_xy)
            
        with open(os.path.join(self.label_save_dir, self.label_save_name), "w") as f:
            json.dump(anns, f)    
        print('There are %d images!' % num_images)
        
    def generate_marker_corner_point(self, contour, L_Type, xy_1, xy_2):
        x1 = xy_1[0]
        y1 = xy_1[1]
        x2 = xy_2[0]
        y2 = xy_2[1]
        A = y2 - y1
        B = x1 - x2
        C = x1*(-A) + y1*(-B)
        distance_set = []
        filter_set = []
        num_points = len(contour)
        for i in range(num_points):
            X1 = float(contour[i][0])
            Y1 = float(contour[i][1])
            fenzi = A * X1 + B * Y1 + C
            fenmu = math.sqrt(math.pow(A, 2) + math.pow(B, 2))
            distance = math.fabs(fenzi) / fenmu
            if(L_Type == 0 or L_Type == 2):
                if fenzi < 0:
                    distance_set.append(distance)
                    filter_set.append(contour[i])
            elif(L_Type == 1 or L_Type == 3):
                if fenzi > 0:
                    distance_set.append(distance)
                    filter_set.append(contour[i])

        maxPosition = distance_set.index(max(distance_set))# 最大值的索引
        corner = filter_set[maxPosition]
        return corner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default='/root/code/dataset/cell_guide/train01', type=str, help="the dir of image dataset")
    parser.add_argument("--ann_dir", default='/root/code/dataset/cell_guide/annotation',
                        type=str, help="the dir of josn label")
    parser.add_argument("--ann_name", default='train01.xml', type=str, help="the name of josn label")
    parser.add_argument("--label_save_dir", default='/root/code/AI-Note-Demo/01-ObjectDetection/CenterNet/code/img_out', type=str,
                        help="the path to save generate label")
    parser.add_argument("--label_save_name", default='train01.json', type=str, help="the name of saving generate label")
    parser.add_argument("--class_names", default='cell_guide_classes.txt', type=str, help="the name to classes")
    cfg = parser.parse_args()    
    zpmc_GenerateTrainLabel(dataset_dir=cfg.dataset_dir, ann_dir=cfg.ann_dir, ann_name=cfg.ann_name, label_save_dir=cfg.label_save_dir, label_save_name=cfg.label_save_name, class_names=cfg.class_names)