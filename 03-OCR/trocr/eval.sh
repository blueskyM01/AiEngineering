#!/bin/bash

# 设置检查点目录和评估脚本
CHECKPOINT_DIR="checkpoint/"
EVAL_SCRIPT="evaluate.py"
LOG_FILE="evaluation_results.log"
WEIGHTS="cust-data/weights"

# 清空日志文件
echo "" > $LOG_FILE

# 遍历 checkpoint/ 目录下的所有子文件夹
for DIR in $(ls -d $CHECKPOINT_DIR*/ | sort -V); do
    MODEL_PATH="$DIR/pytorch_model.bin"
    echo "MODEL_PATH: $MODEL_PATH"
    echo "Copying $MODEL_PATH to $WEIGHTS" | tee -a $LOG_FILE
    cp $MODEL_PATH $WEIGHTS 
    python eval.py --dataset_path "dataset/cust-data/val/*/*.jpg" --cust_data_init_weights_path $WEIGHTS >> $LOG_FILE 2>&1
    # 记录完成状态
    echo "Finished evaluating $MODEL_PATH" | tee -a $LOG_FILE
    echo "------------------------------" | tee -a $LOG_FILE
done

echo "Evaluation completed. Results saved to $LOG_FILE."
