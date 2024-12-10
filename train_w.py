from ultralytics import YOLOWorld

# Load a pretrained YOLOv8s-worldv2 model
model = YOLOWorld("yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="D:/Download/yolov8/pest/datasets/data.yaml", epochs=100, imgsz=640)

# Run inference with the YOLOv8n model on the 'bus.jpg' image
#results = model("ultralytics/assets/bus.jpg")



