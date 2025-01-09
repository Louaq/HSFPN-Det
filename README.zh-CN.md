# 数据集

Google Driver: https://drive.google.com/file/d/1UJcbH2cKLstZdyEJPGv_Hb3GP06MYJff/view?usp=sharinghu

Hugging Face: https://huggingface.co/datasets/yassAQ/pest_diseases

# 如何训练

找到根目录的train.py文件，注释已经写的很清楚，如下图：

```py
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8-HSFPN.yaml')

    # model.load('yolov8n.pt') # 是否加载预训练权重,科研不建议大家加载否则很难提升精度

    model.train(data=r'D:/Downloads/YOLOv8/datasets/data.yaml',
                # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                cache=False,
                imgsz=640,
                epochs=150,
                single_cls=False,  # 是否是单类别检测
                batch=4,
                close_mosaic=10,
                workers=0,
                device='0',
                optimizer='SGD', # using SGD
                # resume='runs/train/exp21/weights/last.pt', # 如过想续训就设置last.pt的地址
                amp=True,  # 如果出现训练损失为Nan可以关闭amp
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
    model = YOLO('D:/Downloads/YOLOv8/result/result_8_HSFPN/train/exp/weights/best.pt') # select your model.pt path
    model.predict(source='D:/Downloads/YOLOv8/ultralytics/assets',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                )
```

同理，把best.pt换成你自己训练好的模型，source里面输入检测图片的路径，运行该脚本就可以开始检测，结果保存在runs/detect目录。





# 经验之谈

**（1）以下为两个重要库的版本，必须对应下载，否则会报错**



> python == 3.9.7
> pytorch == 1.12.1 
> timm == 0.9.12  # 此安装包必须要
> mmcv-full == 1.6.2  # 不安装此包部分关于dyhead的代码运行不了以及Gold-YOLO





**（2）mmcv-full会安装失败是因为自身系统的编译工具有问题，也有可能是环境之间安装的有冲突**

推荐大家离线安装的形式,下面的地址中大家可以找找自己的版本,下载到本地进行安装。

https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
https://download.openmmlab.com/mmcv/dist/index.html

**（3）basicsr安装失败原因,通过pip install basicsr 下载如果失败,大家可以去百度搜一下如何换下载镜像源就可以修复**



## 针对一些报错的解决办法在这里说一下

**(1)训练过程中loss出现Nan值.**
   可以尝试关闭AMP混合精度训练.

**(2)多卡训练问题,修改模型以后不能支持多卡训练可以尝试下面的两行命令行操作，两个是不同的操作，是代表不同的版本现尝试第一个不行用第二个**

    python -m torch.distributed.run --nproc_per_node 2 train.py
    python -m torch.distributed.launch --nproc_per_node 2 train.py

**(3) 针对运行过程中的一些报错解决**
1.如果训练的过程中验证报错了(主要是一些形状不匹配的错误这是因为验证集的一些特殊图片导致)找到ultralytics/models/yolo/detect/train.py的DetectionTrainer class中的build_dataset函数中的rect=mode == 'val'改为rect=False

2.推理的时候运行detect.py文件报了形状不匹配的错误
找到ultralytics/engine/predictor.py找到函数def pre_transform(self, im),在LetterBox中的auto改为False

3.训练的过程中报错类型不匹配的问题
找到'ultralytics/engine/validator.py'文件找到 'class BaseValidator:' 然后在其'__call__'中
self.args.half = self.device.type != 'cpu'  # force FP16 val during training的一行代码下面加上self.args.half = False

**(4) 针对yaml文件中的nc修改**
    不用修改，模型会自动根据你数据集的配置文件获取。
    这也是模型打印两次的区别，第一次打印出来的就是你选择模型的yaml文件结构，第二次打印的就是替换了你数据集的yaml文件，模型使用的是第二种。

**(5) 针对环境的问题**
    环境的问题每个人遇见的都不一样，可自行上网查找。





