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
import cv2, json
import argparse

def create_result_image(original_image, ocr_result, output_dir, img_name, font_scale=0.7, font_thickness=2):
    if original_image is None:
        print("无法读取图像文件。请检查路径是否正确。")
        return

    # 设置文本参数
    font = cv2.FONT_HERSHEY_SIMPLEX
    padding = 10
    text_color = (0, 0, 0)  # 黑色
    background_color = (255, 255, 255)  # 白色

    # 将OCR结果拆分为多行文本
    ocr_lines = ocr_result.split("\n")

    # 计算每行文本的大小
    text_sizes = [cv2.getTextSize(line, font, font_scale, font_thickness)[0] for line in ocr_lines]
    max_text_width = max([size[0] for size in text_sizes])
    total_text_height = sum([size[1] for size in text_sizes]) + (len(ocr_lines) - 1) * padding

    # 使用原图的宽度来设置文本背景图像的宽度
    text_image_width = original_image.shape[1]
    text_image_height = total_text_height + 2 * padding

    # 创建文本背景图像并居中绘制OCR结果
    text_image = np.full((text_image_height, text_image_width, 3), background_color, dtype=np.uint8)
    y_offset = padding
    for i, line in enumerate(ocr_lines):
        text_size = text_sizes[i]
        x = (text_image_width - text_size[0]) // 2  # 使文本居中
        y = y_offset + text_size[1]
        cv2.putText(text_image, line, (x, y), font, font_scale, text_color, font_thickness)
        y_offset += text_size[1] + padding

    # 将原图像与文本图像垂直拼接
    combined_image = np.vstack((original_image, text_image))

    # 保存结果图像
    output_path = os.path.join(output_dir, img_name)
    cv2.imwrite(output_path, combined_image)
    print(f"结果图片已保存为 {output_path}")


def read_vocab(path):
    """
    加载词典
    """
    with open(path) as f:
        vocab = json.load(f)
    return vocab

def do_norm(x):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    x = x/255.0
    x[0, :, :] -= mean[0]
    x[1, :, :] -= mean[1]
    x[2, :, :] -= mean[2]
    x[0, :, :] /= std[0]
    x[1, :, :] /= std[1]
    x[2, :, :] /= std[2]
    return x

def decode_text(tokens, vocab, vocab_inp):
    ##decode trocr
    s_start = vocab.get('<s>')
    s_end = vocab.get('</s>')
    unk = vocab.get('<unk>')
    pad = vocab.get('<pad>')
    text = ''
    for tk in tokens:

        if tk == s_end:
            break
        if tk not in [s_end, s_start, pad, unk]:
            text += vocab_inp[tk]

    return text


def onnx_parser_set(FPMode, onnxFile):
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
    
    return logger, builder, network, profile, config

def trt_compile(logger, builder, network, profile, config, trtFile_save_dir, trtFile):
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
    return engine



def zpmc_onnx2trt(encoder_onnxFile, decoder_onnxFile, trtFile_save_dir, trtFile_encoder_save_name, trtFile_decoder_save_name, FPMode, images_dir, detect_save_dir):
    if not os.path.exists(detect_save_dir):  # os模块判断并创建
            os.mkdir(detect_save_dir)
    # --------------------------------------------------------  encoder  -----------------------------------------------------------
    encoder_logger, encoder_builder, encoder_network, encoder_profile, encoder_config = onnx_parser_set(FPMode, encoder_onnxFile)
    encoder_inputTensor0 = encoder_network.get_input(0)
    encoder_outputTensor0 = encoder_network.get_output(0)
    print('encoder_inputTensor0:', encoder_inputTensor0.name, encoder_inputTensor0.shape)
    print('encoder_outputTensor0:', encoder_outputTensor0.name, encoder_outputTensor0.shape)
    batch_size, channel, nHeight, nWidth = [1, 3, 384, 384]
    encoder_profile.set_shape(encoder_inputTensor0.name, 
                      (batch_size, channel, nHeight, nWidth), 
                      (batch_size, channel, nHeight, nWidth), 
                      (batch_size, channel, nHeight, nWidth)) # 最小batch，常见batch，最大batch
    encoder_config.add_optimization_profile(encoder_profile)
    encoder_trtFile = os.path.join(trtFile_save_dir, trtFile_encoder_save_name)
    encoder_engine = trt_compile(encoder_logger, encoder_builder, encoder_network, encoder_profile, encoder_config, trtFile_save_dir, encoder_trtFile)
    encoder_context = encoder_engine.create_execution_context()
    encoder_inputs, encoder_outputs, encoder_bindings, encoder_stream = common.allocate_buffers(encoder_engine)
    # --------------------------------------------------------  encoder  -----------------------------------------------------------

    # --------------------------------------------------------  decoder  -----------------------------------------------------------
    decoder_logger, decoder_builder, decoder_network, decoder_profile, decoder_config = onnx_parser_set(FPMode, decoder_onnxFile)
    decoder_inputTensor0 = decoder_network.get_input(0)
    decoder_inputTensor1 = decoder_network.get_input(1)
    decoder_outputTensor0 = decoder_network.get_output(0)
    print('decoder_inputTensor0:', decoder_inputTensor0.name, decoder_inputTensor0.shape)
    print('decoder_inputTensor1:', decoder_inputTensor1.name, decoder_inputTensor1.shape)
    print('decoder_outputTensor0:', decoder_outputTensor0.name, decoder_outputTensor0.shape)
    
    n_ids = 100
    decoder_profile.set_shape(decoder_inputTensor0.name, 
                              (batch_size, n_ids), 
                              (batch_size, n_ids), 
                              (batch_size, n_ids)) # 最小batch，常见batch，最大batch
    
    decoder_profile.set_shape(decoder_inputTensor1.name, 
                              (batch_size, encoder_outputTensor0.shape[1], encoder_outputTensor0.shape[2]), 
                              (batch_size, encoder_outputTensor0.shape[1], encoder_outputTensor0.shape[2]), 
                              (batch_size, encoder_outputTensor0.shape[1], encoder_outputTensor0.shape[2])) # 最小batch，常见batch，最大batch

    decoder_config.add_optimization_profile(decoder_profile)
    decoder_trtFile = os.path.join(trtFile_save_dir, trtFile_decoder_save_name)
    decoder_engine = trt_compile(decoder_logger, decoder_builder, decoder_network, decoder_profile, decoder_config, trtFile_save_dir, decoder_trtFile)
    decoder_context = decoder_engine.create_execution_context()
    decoder_inputs, decoder_outputs, decoder_bindings, decoder_stream = common.allocate_buffers(decoder_engine)
    # --------------------------------------------------------  decoder  -----------------------------------------------------------
    vocab = read_vocab('/root/code/AiEngineering/03-OCR/trocr/onnx/vocab.json')
    vocab_inp = {vocab[key]: key for key in vocab}

    image_names = os.listdir(images_dir)
    for image_name in image_names:
        # 读取图像
        image_path = os.path.join(images_dir, image_name)
        image_ori = cv2.imread(image_path)
        image = image_ori[..., ::-1] ##BRG to RGB
        image = cv2.resize(image, (384, 384))
        pixel_values = cv2.split(np.array(image))
        pixel_values = do_norm(np.array(pixel_values))
        pixel_values = np.array([pixel_values])
        encoder_inputs[0].host = np.ascontiguousarray(pixel_values, dtype=np.float32)
        encode_trt_outputs = common.do_inference_v2(encoder_context, bindings=encoder_bindings, inputs=encoder_inputs, outputs=encoder_outputs, stream=encoder_stream)
        last_hidden_state = encode_trt_outputs[0].reshape(encoder_outputTensor0.shape)
        
        
        ids = [vocab["<s>"], ]
        vocab_size = len(vocab.keys())
        for i in range(100):
            input_ids = np.array([ids + [vocab["<pad>"]] * (n_ids - len(ids))])
            decoder_inputs[0].host = np.ascontiguousarray(input_ids, dtype=np.int32)
            decoder_inputs[1].host = np.ascontiguousarray(last_hidden_state, dtype=np.float32)
            decoder_trt_outputs = common.do_inference_v2(decoder_context, bindings=decoder_bindings, inputs=decoder_inputs, outputs=decoder_outputs, stream=decoder_stream)
            logits = decoder_trt_outputs[0].reshape([-1, n_ids, vocab_size])
        
            pred = logits[0]
            pred = pred.argmax(axis=1)
            if pred[i] == vocab["</s>"]:
                break
            ids.append(pred[i])
        
        text = decode_text(ids, vocab, vocab_inp)
        # print("logits shape:", logits.shape)
        # print("last_hidden_state shape:", last_hidden_state.shape)
        print(text)
        create_result_image(image_ori, text, detect_save_dir, image_name)
        
        
        
        



    

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='convert to trt and inference')
    parser.add_argument('--encoder_onnxFile', default='onnx/encoder_model.onnx', type=str, help='')
    parser.add_argument('--decoder_onnxFile', default='onnx/decoder_model.onnx', type=str, help='')
    parser.add_argument('--trtFile_save_dir', default='trt', type=str, help='')
    parser.add_argument('--trtFile_encoder_save_name', default='encoder_model32.trt', type=str, help='')
    parser.add_argument('--trtFile_decoder_save_name', default='decoder_model32.trt', type=str, help='')
    parser.add_argument('--FPMode', default='FP32', type=str, help='')
    parser.add_argument('--images_dir', default='/root/code/dataset/containercode/text_rec/val', type=str, help='')
    parser.add_argument('--detect_save_dir', default='result', type=str, help='')
    args = parser.parse_args()

    zpmc_onnx2trt(encoder_onnxFile=args.encoder_onnxFile, 
                  decoder_onnxFile=args.decoder_onnxFile, 
                  trtFile_save_dir=args.trtFile_save_dir, 
                  trtFile_encoder_save_name=args.trtFile_encoder_save_name, 
                  trtFile_decoder_save_name=args.trtFile_decoder_save_name,
                  FPMode=args.FPMode, 
                  images_dir=args.images_dir, detect_save_dir=args.detect_save_dir)