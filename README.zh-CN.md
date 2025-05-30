<div align="center">
  中文 | <a href="./README.md">English</a> </a>
</div>

# 数据集

Google Driver: https://drive.google.com/file/d/1UJcbH2cKLstZdyEJPGv_Hb3GP06MYJff/view?usp=sharinghu

Hugging Face: https://huggingface.co/datasets/loupk/pest_diseases

# 如何训练

找到根目录的train.py文件，注释已经写的很清楚，如下图：

```py
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8-HSFPN.yaml')

    # model.load('yolov8n.pt') 

    model.train(data=r'/path/to/datasets/',
               
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
                amp=True, 
                project='runs/train',
                name='exp',
                )
```

还有一个检测的脚本，Detect.py:

```python
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/path/to/ptht') # select your model.pt path
    model.predict(source='assets',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                )
```

同理，把best.pt换成你自己训练好的模型，source里面输入检测图片的路径，运行该脚本就可以开始检测，结果保存在runs/detect目录。
