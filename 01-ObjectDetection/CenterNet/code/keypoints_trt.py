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
import cv2
import argparse

def merge_image(images):
    img_tl, img_tr, img_bl, img_br = images
    heigt = img_tl.shape[0]
    width = img_tl.shape[1]
    c = img_tl.shape[2]
    merge_img = np.ones([2*heigt, 2*width,c], img_tl.dtype)
    merge_img[:heigt, :width, :] = img_tl
    merge_img[:heigt, width: 2*width, :] = img_tr
    merge_img[heigt: 2*heigt, :width, :] = img_bl
    merge_img[heigt: 2*heigt, width:2*width, :] = img_br
    return merge_img

def zpmc_onnx2trt(onnxFile, trtFile_save_dir, trtFile_save_name, FPMode, video_dir, detect_save_dir):
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
    outputTensor2 = network.get_output(2)
    outputTensor3 = network.get_output(3)
    outputTensor4 = network.get_output(4)
    print('inputTensor:', inputTensor.name, inputTensor.shape)
    print('outputTensor0:', outputTensor0.name, outputTensor0.shape)
    print('outputTensor1:', outputTensor1.name, outputTensor1.shape)
    print('outputTensor2:', outputTensor2.name, outputTensor2.shape)
    print('outputTensor3:', outputTensor3.name, outputTensor3.shape)
    print('outputTensor4:', outputTensor4.name, outputTensor4.shape)

    _, nHeight, nWidth, _ = inputTensor.shape
    profile.set_shape(inputTensor.name, (4, nHeight, nWidth, 3), (4, nHeight, nWidth, 3), (4, nHeight, nWidth, 3)) # 最小batch，常见batch，最大batch
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

    
    mx_Capture_0 = cv2.VideoCapture(os.path.join(video_dir, 'cam1_000.avi'))
    mx_Capture_1 = cv2.VideoCapture(os.path.join(video_dir, 'cam2_000.avi'))
    mx_Capture_2 = cv2.VideoCapture(os.path.join(video_dir, 'cam3_000.avi'))
    mx_Capture_3 = cv2.VideoCapture(os.path.join(video_dir, 'cam4_000.avi'))
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    size = (int(mx_Capture_0.get(cv2.CAP_PROP_FRAME_WIDTH) * 2), int(mx_Capture_0.get(cv2.CAP_PROP_FRAME_HEIGHT) * 2))
    if not os.path.exists(detect_save_dir):  # os模块判断并创建
        os.mkdir(detect_save_dir)
    merge_out = cv2.VideoWriter(os.path.join(detect_save_dir, FPMode+'.avi'), fourcc, 22, size)

    counter = 0
    while 1:
        mx_ret_0, image_src_0 = mx_Capture_0.read()
        mx_ret_1, image_src_1 = mx_Capture_1.read()
        mx_ret_2, image_src_2 = mx_Capture_2.read()
        mx_ret_3, image_src_3 = mx_Capture_3.read()

        if mx_ret_0 and mx_ret_1 and mx_ret_2 and mx_ret_3:
            h, w, _ = image_src_0.shape
            images = [image_src_0, image_src_1, image_src_2, image_src_3]
            images_resize_list = zpmc_ImagePreprocess(images, img_size=(550, 550))

            starttime = datetime.datetime.now()    

            inputs[0].host = np.ascontiguousarray(images_resize_list, dtype=np.float32)
            # cfx.push()
            trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            # cfx.pop()
            batch_size = len(images_resize_list)
            trt_outputs_shape = [(batch_size, 138, 138, 32), (batch_size, 19248, 4), (batch_size, 19248, 32), (batch_size, 19248, 2), (19248, 4)]
            
            proto_data = trt_outputs[0].reshape(trt_outputs_shape[0])
            loc_data = trt_outputs[1].reshape(trt_outputs_shape[1])
            mask_data = trt_outputs[2].reshape(trt_outputs_shape[2])
            conf_data = trt_outputs[3].reshape(trt_outputs_shape[3])
            prior_data = trt_outputs[4].reshape(trt_outputs_shape[4])
            
            CLASSES = ['BG', 'LOCKHOLE']
            result = zpmc_PostProcess(CLASSES, proto_data, prior_data, mask_data, conf_data, loc_data, conf_thresh=0.05)
            classes, scores, boxes, masks = zpmc_display(result, images_resize_list, h, w, score_threshold=0.15)

            endtime = datetime.datetime.now()
            timediff = (endtime - starttime).total_seconds()
            counter+=1
            print(timediff, '    ', 1/timediff, '   ', counter)

            merge_images = []
            for nclass, score, box, mask, image in zip(classes, scores, boxes, masks, images):    
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                mask = np.where(mask > 80, 255, 0).astype(np.uint8)
                visualize_image = daw_mask(image, mask)
                for lable, sc, bb in zip(nclass, score, box):
                    caption = CLASSES[lable+1]
                    x0 = int(bb[0])
                    y0 = int(bb[1])
                    x1 = int(bb[2])
                    y1 = int(bb[3])
                    aa = [x0, y0, x1, y1]
                    visualize_image = draw_caption(visualize_image, aa, caption, sc)  
                merge_images.append(visualize_image)
            merge_img = merge_image(merge_images)
            merge_out.write(merge_img)
        else:
            mx_Capture_0.release()
            mx_Capture_1.release()
            mx_Capture_2.release()
            mx_Capture_3.release()
            merge_out.release()
            break
        #     cv2.imwrite('results/' + str(counter) + '.jpg', visualize_image)
        #     counter += 1

    

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN')
    parser.add_argument('--onnxFile', default='onnx/yolact.onnx', type=str, help='')
    parser.add_argument('--trtFile_save_dir', default='trt', type=str, help='')
    parser.add_argument('--trtFile_save_name', default='yolact16.trt', type=str, help='')
    parser.add_argument('--FPMode', default='FP16', type=str, help='')
    parser.add_argument('--video_dir', default='/root/code/temp_file/firstlanding/', type=str, help='')
    parser.add_argument('--detect_save_dir', default='result', type=str, help='')
    args = parser.parse_args()

    zpmc_onnx2trt(onnxFile=args.onnxFile, trtFile_save_dir=args.trtFile_save_dir, trtFile_save_name=args.trtFile_save_name, FPMode=args.FPMode, 
                  video_dir=args.video_dir, detect_save_dir=args.detect_save_dir)