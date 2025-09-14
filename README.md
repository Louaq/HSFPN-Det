
<div align="center">
  English | <a href="./README.zh-CN.md">中文</a> </a>
</div>

## 🔥 Updates
- (2025.05.09) submission
- (2025.01.02) upload code

## 👉 Dataset

Google Driver: [link](https://drive.google.com/file/d/1UJcbH2cKLstZdyEJPGv_Hb3GP06MYJff/view?usp=sharinghu)

Hugging Face: [link](https://huggingface.co/datasets/loupk/pest_diseases)

## 👉 Train

Find the train.py file in the root directory, the comments have been written clearly as below:

```py
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8.yaml')

    model.train(data=r'/path/to/datasets/',
 
                cache=False,
                imgsz=640,
                epochs=500,
                single_cls=False,  
                batch=4,
                close_mosaic=10,
                workers=0,
                device='0',
                optimizer='SGD', 
                amp=True,  
                project='runs/train',
                name='exp',
                )
```


There's also a script for detecting, Detect.py: 

```python
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('best.pt') # select your model.pt path
    model.predict(source='assets',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                )
```

Similarly, replace best.pt with your own trained model, enter the path of the detected image
inside source, run the script to start the detection, and save the results in the runs/detect directory.







