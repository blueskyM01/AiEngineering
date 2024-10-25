from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n-seg.pt")  # load an official model
model = YOLO("/root/code/AiEngineering/02-Segmetation/Yolo-v8/ultralytics/runs/segment/train/weights/epoch9.pt")  # load a custom trained model

# Export the model
model.export(format="onnx", dynamic=True) # dynamic=True 允许导出动态输入大小的 ONNX 模型。