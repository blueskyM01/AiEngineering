from ultralytics import YOLO
import cv2

def draw_str_on_image(img, text, text_position):
    # 获取文本大小
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    x_min, y_min = text_position
    # 设置文本背景位置
    # 计算背景矩形的坐标
    text_background_tl = (x_min, y_min - text_height - 5)  # 左上角 (top-left)
    text_background_br = (x_min + text_width, y_min)  # 右下角 (bottom-right)
    
    # 绘制背景矩形 (这里使用黑色背景)
    cv2.rectangle(img, text_background_tl, text_background_br, (0, 0, 0), -1)
    
    # 在图像上写置信度
    cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    

if __name__ == "__main__":
    # Load a model
    model = YOLO("/root/code/AiEngineering/02-Segmetation/Yolo-v8/ultralytics/runs/segment/train/weights/epoch0.pt")  # load an official model

    # Predict with the model
    src_img0 = cv2.imread('/root/code/dataset/cornerline-ningbobeier/cornerline-ningbobeier/images/val/image_0000013357.jpeg')
    src_img1 = cv2.imread('/root/code/dataset/cornerline-ningbobeier/cornerline-ningbobeier/images/val/image_0000013375.jpeg')
    src_img2 = cv2.imread('/root/code/dataset/cornerline-ningbobeier/cornerline-ningbobeier/images/val/image_0000013589.jpeg')
    src_img3 = cv2.imread('/root/code/dataset/cornerline-ningbobeier/cornerline-ningbobeier/images/val/image_0000013592.jpeg')
    
    src_list = [src_img0, src_img1, src_img2, src_img3]
    results = model(src_list, imgsz=640, conf=0.3)  # predict on an image

    # Process results list
    for idx, result in enumerate(results):
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        
        clses = boxes.cls.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        bboxes_xywh = boxes.xywh.cpu().numpy()    
        masks_xy = masks.xy
        num_target = clses.shape[0]
        
        # 绘制轮廓，置信度
        for n_idx in range(num_target):
            n_cls = int(clses[n_idx])
            n_conf = confs[n_idx]
            n_bbox_xywh = bboxes_xywh[n_idx]
            
            x_c = n_bbox_xywh[0]
            y_c = n_bbox_xywh[1]
            w = n_bbox_xywh[2]
            h = n_bbox_xywh[3]
            x_min = int(x_c - w / 2.0)
            y_min = int(y_c - h / 2.0)
            x_max = int(x_c + w / 2.0)
            y_max = int(y_c + h / 2.0)
            cv2.rectangle(src_list[idx], (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
            
            # 绘制置信度文本
            # 设置置信度文本，保留两位小数
            confidence_text = "class:" + str(n_cls) + " score:" + f'{n_conf:.3f}'
            draw_str_on_image(src_list[idx], confidence_text, (x_min, y_min))
            
        masks_contours = [mask_xy.reshape(-1, 1, 2).astype(int) for mask_xy in masks_xy]
        cv2.drawContours(src_list[idx], masks_contours, -1, (0, 255, 0), 2)

        cv2.imwrite('/root/code/AiEngineering/02-Segmetation/Yolo-v8/ultralytics/results/'+str(idx)+"result.jpg", src_list[idx])

    
