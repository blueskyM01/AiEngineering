import cv2, argparse, os, json, shutil
from collections import defaultdict
import numpy as np

def generate_text_dataset(input_dataset_dir, input_ann_path, output_dataset_dir):
    Define_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
    faith_counter = 0
    fault_dict = {}

    with open(input_ann_path, 'r') as load_f:
        load_dict = json.load(load_f)
    load_f.close()
    
    img_names = load_dict.keys()
    for idx, image_name in enumerate(img_names):
        text = load_dict[image_name]
        if ' ' in text:
            text = text.replace(" ", "")
        
        if text != "" and set(text).issubset(set(Define_str)):
            folder_name = '%09d' % (idx)
            folder_path = os.path.join(output_dataset_dir, folder_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            source_path = os.path.join(input_dataset_dir, image_name)
            destination_path = os.path.join(folder_path, image_name) 
            shutil.copyfile(source_path, destination_path)
            print("copy image from %s to %s " % (source_path, destination_path))
            
            
            txt_name = image_name.split(".")[0] + ".txt"
            txt_path = os.path.join(folder_path, txt_name)
            
            f = open(txt_path, 'w')
            f.write(text)
            if text == "":
                print("empty gt", txt_path)
                break
        else:
            print("text中的某些字符不在Define_str中, image name: %s, text: %s" % (image_name, text))
            faith_counter += 1
            fault_dict[image_name]=text
            continue
        

        
        print("generating %s, text: %s" % (txt_path, text))
        
    print("There are %d fault annotations! As shown in below:" % (faith_counter))
    print(fault_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--input_dataset_dir', type=str, default='/root/code/dataset/containercode/text_rec/train', help='')
    parser.add_argument('--input_ann_path', type=str, default='/root/code/dataset/containercode/text_rec/ann/train.json', help='')
    parser.add_argument('--output_dataset_dir', type=str, default='/root/code/AiEngineering/03-OCR/trocr/dataset/cust-data/train', help='')
    args = parser.parse_args()
    generate_text_dataset(args.input_dataset_dir, args.input_ann_path, args.output_dataset_dir)

