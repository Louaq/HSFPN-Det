import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('D:/path/to/weight/') # select your model.pt path
    model.predict(source='D:/path/to/image',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                )