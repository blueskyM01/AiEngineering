from cuda import cudart
import cv2
from datetime import datetime as dt
from glob import glob
import numpy as np
import os, sys
import tensorrt as trt
sys.path.append('/root/TensorRT-8.4.3.1/samples/python') 
import common
import datetime
import argparse
import math, random



def get_classes(file_path):
    classes = []
    with open(file_path, 'r') as f:
        line = f.readline()
        while line:
            classes.append(line.rstrip('\n'))
            line = f.readline()
    return classes

def prepare_input(image, input_width, input_height):
    input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize input image
    input_img = cv2.resize(input_img, (input_width, input_height))

    # Scale input pixel values to 0 to 1
    input_img = input_img / 255.0
    input_img = input_img.transpose(2, 0, 1)
    input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

    return input_tensor

def rescale_boxes(boxes, input_shape, image_shape):
    # Rescale boxes to original image dimensions
    input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])

    return boxes

def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y
    
def extract_boxes(box_predictions, input_height, input_width, img_height, img_width):
    # Extract boxes from predictions
    boxes = box_predictions[:, :4]

    # Scale boxes to original image dimensions
    boxes = rescale_boxes(boxes,
                                (input_height, input_width),
                                (img_height, img_width))

    # Convert boxes to xyxy format
    boxes = xywh2xyxy(boxes)

    # Check the boxes are within the image
    boxes[:, 0] = np.clip(boxes[:, 0], 0, img_width)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, img_height)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, img_width)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, img_height)

    return boxes

def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou

def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes
    
def process_box_output(box_output, num_masks, conf_threshold, iou_threshold,
                       input_height, input_width, img_height, img_width):
    '''
    (batch_size, num_classes + 4 + num_masks, num_boxes)
    '''
    predictions = np.squeeze(box_output).T
    num_classes = box_output.shape[1] - num_masks - 4

    # Filter out object confidence scores below threshold
    scores = np.max(predictions[:, 4:4+num_classes], axis=1)
    predictions = predictions[scores > conf_threshold, :]
    scores = scores[scores > conf_threshold]

    if len(scores) == 0:
        return [], [], [], np.array([])

    box_predictions = predictions[..., :num_classes+4]
    mask_predictions = predictions[..., num_classes+4:]

    # Get the class with the highest confidence
    class_ids = np.argmax(box_predictions[:, 4:], axis=1)

    # Get bounding boxes for each object
    boxes = extract_boxes(box_predictions, input_height, input_width, img_height, img_width)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    indices = nms(boxes, scores, iou_threshold)

    return boxes[indices], scores[indices], class_ids[indices], mask_predictions[indices]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def process_mask_output(mask_predictions, mask_output, boxes, img_height, img_width):

    if mask_predictions.shape[0] == 0:
        return []

    mask_output = np.squeeze(mask_output)

    # Calculate the mask maps for each box
    num_mask, mask_height, mask_width = mask_output.shape  # CHW
    masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
    masks = masks.reshape((-1, mask_height, mask_width))

    # Downscale the boxes to match the mask size
    scale_boxes = rescale_boxes(boxes,
                                (img_height, img_width),
                                (mask_height, mask_width))

    # For every box/mask pair, get the mask map
    mask_maps = np.zeros((len(scale_boxes), img_height, img_width))
    blur_size = (int(img_width / mask_width), int(img_height / mask_height))
    for i in range(len(scale_boxes)):

        scale_x1 = int(math.floor(scale_boxes[i][0]))
        scale_y1 = int(math.floor(scale_boxes[i][1]))
        scale_x2 = int(math.ceil(scale_boxes[i][2]))
        scale_y2 = int(math.ceil(scale_boxes[i][3]))

        x1 = int(math.floor(boxes[i][0]))
        y1 = int(math.floor(boxes[i][1]))
        x2 = int(math.ceil(boxes[i][2]))
        y2 = int(math.ceil(boxes[i][3]))

        scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
        crop_mask = cv2.resize(scale_crop_mask,
                            (x2 - x1, y2 - y1),
                            interpolation=cv2.INTER_CUBIC)

        crop_mask = cv2.blur(crop_mask, blur_size)

        crop_mask = (crop_mask > 0.5).astype(np.uint8)
        mask_maps[i, y1:y2, x1:x2] = crop_mask

    return mask_maps

def draw_masks(image, boxes, class_ids, colors, mask_alpha=0.3, mask_maps=None,):
    mask_img = image.copy()

    # Draw bounding boxes and labels of detections
    for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # Draw fill mask image
        if mask_maps is None:
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
        else:
            crop_mask = mask_maps[i][y1:y2, x1:x2, np.newaxis]
            crop_mask_img = mask_img[y1:y2, x1:x2]
            crop_mask_img = crop_mask_img * (1 - crop_mask) + crop_mask * color
            mask_img[y1:y2, x1:x2] = crop_mask_img

    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)

def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3, mask_maps=None, class_names=['person']):
    rng = np.random.default_rng(3)
    colors = rng.uniform(0, 255, size=(len(class_names), 3))
    img_height, img_width = image.shape[:2]
    size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    mask_img = draw_masks(image, boxes, class_ids, colors, mask_alpha, mask_maps)

    # Draw bounding boxes and labels of detections
    for box, score, class_id in zip(boxes, scores, class_ids):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # Draw rectangle
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, 2)

        label = class_names[class_id]
        caption = f'{label} {int(score * 100)}%'
        (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=size, thickness=text_thickness)
        th = int(th * 1.2)

        cv2.rectangle(mask_img, (x1, y1),
                      (x1 + tw, y1 - th), color, -1)

        cv2.putText(mask_img, caption, (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

    return mask_img

def draw_results(image, boxes, scores, class_ids, mask_maps, draw_scores=True, mask_alpha=0.5, class_names=['person']):
    return draw_detections(image, boxes, scores,
                            class_ids, mask_alpha, mask_maps=mask_maps, class_names=class_names)

def zpmc_onnx2trt(onnxFile, trtFile_save_dir, trtFile_save_name, FPMode, images_dir, detect_save_dir, class_names):
    if not os.path.exists(detect_save_dir):
        os.makedirs(detect_save_dir)
    input_shape = [640, 640]
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 << 30)

    if FPMode == 'FP32':
        print('Set FPMode = FP32!')
    elif FPMode == 'FP16':
        config.set_flag(trt.BuilderFlag.FP16)
        print('Set FPMode = FP16!')
    else:
        print('Please set FPMode = FP32 or FP16')
        exit()

    parser = trt.OnnxParser(network, logger)

    if not os.path.exists(onnxFile):
        print("Failed finding {}!".format(onnxFile))
        exit()
    print("Succeeded finding {}!".format(onnxFile))

    with open(onnxFile, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing {}!".format(onnxFile))
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing {}!".format(onnxFile))

    inputTensor = network.get_input(0)
    outputTensor0 = network.get_output(0)
    outputTensor1 = network.get_output(1)

    print('inputTensor:', inputTensor.name, inputTensor.shape)
    print('outputTensor0:', outputTensor0.name, outputTensor0.shape) # dbox
    print('outputTensor1:', outputTensor1.name, outputTensor1.shape) # cls

    # batch_size, _, nHeight, nWidth = inputTensor.shape
    batch_size, _, nHeight, nWidth = (1, 3, 640, 640)
    profile.set_shape(inputTensor.name, (batch_size, _, nHeight, nWidth), (batch_size, _, nHeight, nWidth), (batch_size, _, nHeight, nWidth)) # 最小batch，常见batch，最大batch
    config.add_optimization_profile(profile)

    trtFile = os.path.join(trtFile_save_dir, trtFile_save_name)
    if not os.path.exists(trtFile):
        engineString = builder.build_serialized_network(network, config)
        if engineString == None:
            print("Failed building engine!")
            exit()
        print("Succeeded building engine!")

        if not os.path.exists(trtFile_save_dir):  # os模块判断并创建
            os.mkdir(trtFile_save_dir)
        with open(trtFile, "wb") as f:
            f.write(engineString)
        print("Generate {}!".format(trtFile))
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
        print('Loaded {}!'.format(trtFile))
    else:
         with open(trtFile, "rb") as f, trt.Runtime(logger) as runtime:
            engine =  runtime.deserialize_cuda_engine(f.read())
         print('Loaded {}!'.format(trtFile))


    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    
    
    images_list = os.listdir(images_dir)
    print(len(images_list))
    counter = 0
    for image_name in images_list:
        counter += 1
        # print(counter, ':', image_name)
        src = cv2.imread(os.path.join(images_dir, image_name))
        image_shape = src.shape[0:2]
        starttime = datetime.datetime.now()  
        image_data = prepare_input(src, input_shape[0], input_shape[1])
        
        inputs[0].host = np.ascontiguousarray(image_data, dtype=np.float32)
        trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        
        print('trt_outputs[0]:', trt_outputs[0].shape)
        print('trt_outputs[1]:', trt_outputs[1].shape)

        trt_outputs_shape = [(batch_size, 32, 160, 160), 
                             (batch_size, 37, 8400)]
        
        boxes_r = trt_outputs[1].reshape(trt_outputs_shape[1]) 
        masks = trt_outputs[0].reshape(trt_outputs_shape[0])
        boxes, scores, class_ids, mask_pred = process_box_output(box_output=boxes_r, num_masks=32, conf_threshold=0.3, iou_threshold=0.5,
                                                                 input_height=input_shape[1], input_width=input_shape[0], 
                                                                 img_height=image_shape[0], img_width=image_shape[1])
        mask_maps = process_mask_output(mask_predictions=mask_pred, mask_output=masks, boxes=boxes, 
                                        img_height=image_shape[0], img_width=image_shape[1])
        combined_img=draw_results(src, boxes=boxes, scores=scores, class_ids=class_ids, mask_maps=mask_maps, 
                                draw_scores=True, mask_alpha=0.5, class_names=class_names)
        cv2.imwrite(os.path.join(detect_save_dir, str(counter)+"result.jpg"), combined_img)
        
    

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='YoloV8')
    parser.add_argument('--onnxFile', default='/root/code/AiEngineering/02-Segmetation/Yolo-v8/ultralytics/runs/segment/train/weights/epoch9.onnx', type=str, help='')
    parser.add_argument('--trtFile_save_dir', default='/root/code/AiEngineering/02-Segmetation/Yolo-v8/ultralytics/trt', type=str, help='')
    parser.add_argument('--trtFile_save_name', default='yolov8_seg16.trt', type=str, help='')
    parser.add_argument('--FPMode', default='FP16', type=str, help='')
    parser.add_argument('--images_dir', default='/root/code/dataset/cornerline-ningbobeier/cornerline-ningbobeier/images/val', type=str, help='')
    parser.add_argument('--detect_save_dir', default='/root/code/AiEngineering/02-Segmetation/Yolo-v8/ultralytics/img_out', type=str, help='')
    parser.add_argument('--class_file', default='/root/code/AiEngineering/02-Segmetation/Yolo-v8/datas/cornerline.name', type=str, help='')
    args = parser.parse_args()
    class_names = get_classes(args.class_file)
    zpmc_onnx2trt(onnxFile=args.onnxFile, trtFile_save_dir=args.trtFile_save_dir, trtFile_save_name=args.trtFile_save_name, FPMode=args.FPMode, 
                  images_dir=args.images_dir, detect_save_dir=args.detect_save_dir, class_names=class_names)
