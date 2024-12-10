import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('D:/Download/yolov8/weights/triplet.pt') # select your model.pt path
    model.predict(source='D:/Download/yolov8/test_images',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                )