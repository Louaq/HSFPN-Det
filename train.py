import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8.yaml')

    model.train(data=r'D:/path/to/datasets/data.yaml',
                
                cache=False,
                imgsz=640,
                epochs=150,
                single_cls=False,  
                batch=4,
                close_mosaic=10,
                workers=0,
                device='0',
                optimizer='SGD', # using SGD
                # resume='runs/train/exp21/weights/last.pt', 
                amp=False,  
                project='runs/train',
                name='exp',
                )