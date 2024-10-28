import cv2, os
import numpy as np
from shutil import copyfile



label_names = os.listdir('/software/dataset/container_number/label_horizon')

# for label_name in label_names:
#     # if '.jpg' in img_name:
#     #     img = cv2.imread('/software/dataset/container_number/vertical/'+ img_name)
#     #     # cv2.imwrite('/software/dataset/ZPMC_Container_Number/images-1/'+ img_name, img)
#     #     print(img.shape)
#     img_name = label_name.split('.txt')[0] + '.jpg'
#     img = cv2.imread('/software/dataset/container_number/horizon/'+ img_name)
#     # cv2.imwrite('/software/dataset/ZPMC_Container_Number/detect/vertical/'+ img_name, img)
#     print(img_name)


    
for label in label_names:
    img_name = label.split('.txt')[0] + '.jpg'
    img_path = os.path.join('/software/dataset/container_number/horizon', img_name)
    new_path = os.path.join('/software/dataset/ZPMC_Container_Number/detect/horizon', img_name)
    
    copyfile(img_path, new_path)
    print(new_path)


print('total_num:', len(label_names))

# gt_instance = np.zeros([400, 400], dtype='uint8') 

# cv2.imwrite('/software/code/zpmc_psenet_ocr/outputs/show/1.jpg', gt_instance)
# cv2.imwrite('/software/code/zpmc_psenet_ocr/outputs/show/1-1.jpg', img)