#!/bin/bash

# 设置检查点目录和评估脚本
CHECKPOINT_DIR="checkpoints/"
LOG_FILE="psenet_evaluation_results.log"
IMAGE_DIR="/root/code/dataset/containercode/images/val"
ANN_PATH="/root/code/AiEngineering/03-OCR/psenet/data/val.json"
RESULT_SAVE_DIR="ocr_detect_result"
# 清空日志文件
echo "" > $LOG_FILE

# 遍历 checkpoint/ 目录下的所有子文件夹
for DIR in $(ls $CHECKPOINT_DIR/ | sort -V); do
    
    MODEL_PATH=$CHECKPOINT_DIR$DIR
    
    echo "evaluating $MODEL_PATH: " | tee -a $LOG_FILE
    python zpmc_eval.py --checkpoint $MODEL_PATH --img_dir $IMAGE_DIR --ann_path $ANN_PATH --result_save_dir $RESULT_SAVE_DIR >> $LOG_FILE 2>&1
    rm -rf mAP/input/*
    mv $RESULT_SAVE_DIR/detection-results $RESULT_SAVE_DIR/ground-truth mAP/input
    cd mAP
    python main.py >> ../$LOG_FILE 2>&1
    cd ../
    # 记录完成状态
    echo "Finished evaluating $MODEL_PATH" | tee -a $LOG_FILE
    echo "------------------------------" | tee -a $LOG_FILE


done

