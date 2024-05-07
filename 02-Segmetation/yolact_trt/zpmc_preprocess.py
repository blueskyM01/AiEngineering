import cv2
import numpy as np

def zpmc_ImagePreprocess(images, img_size):
    mean = np.array([103.94, 116.78, 123.68]).reshape(1,3,1,1)
    std = np.array([57.38, 57.12, 58.4]).reshape(1,3,1,1)

    images_resize_list = []
    for image in images:
        image_resize = cv2.resize(image, img_size, interpolation=cv2.INTER_LINEAR)
        images_resize_list.append(image_resize)
    images_resize_np = np.array(images_resize_list)

    images_resize_np = np.transpose(images_resize_np, (0, 3, 1, 2))
    images_resize_np = (images_resize_np - mean) / std

    images_resize_np = images_resize_np[:, (2, 1, 0), :, :]
    return images_resize_np



if __name__ == '__main__':
    image1 = cv2.imread('/software/temp/image_0000008691.jpeg')
    # image = cv2.resize(image, (nWidth, nHeight))
    # image = np.transpose(image, (2, 0, 1))
    

    images = [image1]


    image_process = zpmc_ImagePreprocess(images, img_size=(550, 550))
    print(image_process.shape)