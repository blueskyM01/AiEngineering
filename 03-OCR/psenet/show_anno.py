import cv2, argparse, os, json
import numpy as np
from shutil import copyfile

font = cv2.FONT_HERSHEY_SIMPLEX
def visualize_anno(images_dir, anno_label_path, output_dir):
    with open(anno_label_path, 'r') as load_f:
        load_dict = json.load(load_f)
    load_f.close()

    img_names = list(load_dict.keys())

    for img_name in img_names:
        print(img_name)
        img = cv2.imread(os.path.join(images_dir, img_name))
        ocr_anns = load_dict[img_name]
        contours = []
        for ocr_ann in ocr_anns:
            text = ocr_ann[0]
            polygon = ocr_ann[1]
            contour = []
            for i in range(0, len(polygon), 2):
                point = [int(polygon[i]), int(polygon[i+1])]
                contour.append(point)
            cv2.putText(img, text, (int(polygon[0]), int(polygon[1])-5), font, 2, (255, 255, 255), 4)
            contours.append(np.array(contour))
        print(len(contours))
        # cv2.polylines(img, contours, isClosed=True, color=(255,0,0), thickness=1)
        cv2.drawContours(img, contours, -1, color=(255,0,0), thickness=2)
        cv2.imwrite(os.path.join(output_dir, img_name), img)
    print(len(img_names))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--images_dir', type=str, default='/software/dataset/ZPMC_Container_Number/detect/images/val')
    parser.add_argument('--anno_label_path', type=str, default='/software/dataset/ZPMC_Container_Number/detect/annotations/val.json', help='')
    parser.add_argument('--output_dir', type=str, default='./outputs/show/', help='')
    args = parser.parse_args()
    visualize_anno(args.images_dir, args.anno_label_path, args.output_dir)