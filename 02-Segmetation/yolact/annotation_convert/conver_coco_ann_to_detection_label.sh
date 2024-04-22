python conver_coco_ann_to_detection_label.py \
--ann_dir='/root/code/dataset/coco/annotations' \
--ann_name='instances_val2017.json'  \
--label_save_dir='/root/code/AiEngineering/02-Segmetation/yolact/result_temp'  \
--label_save_name='coco_val_detection_parser.txt' \
--class_names='coco.name'