from ultralytics import YOLO

model = YOLO("/root/code/AiEngineering/02-Segmetation/Yolo-v8/pre_trained_weights/yolov8m-seg.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/root/code/AiEngineering/02-Segmetation/Yolo-v8/dataset_preprocess/dataset.yaml", 
                      epochs=100, 
                      imgsz=640,
                      save_period=1,
                      optimizer='Adam')
