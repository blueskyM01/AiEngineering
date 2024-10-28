import cv2, argparse, os
import numpy as np



def visualize_result(images_dir, result_label_dir):
    label_names = os.listdir(result_label_dir)
    for label_name in label_names:
        img_name = label_name.split('.txt')[0] + '.png'
        print(os.path.join(images_dir, img_name))
        img = cv2.imread(os.path.join(images_dir, img_name))
        
        label_path = os.path.join(result_label_dir, label_name)
        contours = []
        with open(label_path, 'r') as f:
            line = f.readline()
            while line:
                xys = line.rstrip('\n').split(',')
                contour = []
                for i in range(0, len(xys), 2):
                    point = [int(xys[i+1]), int(xys[i])]
                    contour.append(point)

                # cv2.polylines(img, [polygon], isClosed=True, color=(255,0,0), thickness=1)
                line = f.readline()
                contours.append(np.array(contour))
        print(len(contours))
        # cv2.polylines(img, contours, isClosed=True, color=(255,0,0), thickness=1)
        cv2.drawContours(img, contours, -1, color=(255,0,0), thickness=2)

        cv2.imwrite('./outputs/show/'+img_name, img)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--images_dir', nargs='?', type=str, default='/software/dataset/ZPMC_Container_Number/test-images')
    parser.add_argument('--result_label_dir', default='outputs/submit_ctw', help='config file path')
    args = parser.parse_args()
    visualize_result(args.images_dir, args.result_label_dir)