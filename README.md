# Dataset

Google Driver: https://drive.google.com/file/d/1UJcbH2cKLstZdyEJPGv_Hb3GP06MYJff/view?usp=sharinghu

Hugging Face: https://huggingface.co/datasets/yassAQ/pest_diseases

# How to train

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



# rule of thumb

**（1）The following are the versions of two important libraries, which must be downloaded 
accordingly, otherwise errors will be reported**

> python == 3.9.7
> pytorch == 1.12.1 
> timm == 0.9.12 
> mmcv-full == 1.6.2 

**（2）The reason why mmcv-full fails to install is because of a problem with the compilation tool on 
your own system, or because of a conflict between the environments in which it is installed.**

We recommend that you install offline in the form of the following address you can find their own version, 
download to the local installation.
    https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
    https://download.openmmlab.com/mmcv/dist/index.html



**（3）basicsr installation failure reasons, through pip install basicsr download if it fails, you can
go to Baidu search how to change the download mirror source can be repaired!**



# The solution to some of the reported errors is here

**(1)Nan values appear for loss during training.**
   Try turning off AMP mixed precision training.

**(2) Multi-card training problems, after modifying the model can not support multi-card
training you can try the following two lines of command line operations, the two are different
operations, is on behalf of different versions now try the first one does not work with the
second one**

    python -m torch.distributed.run --nproc_per_node 2 train.py
    python -m torch.distributed.launch --nproc_per_node 2 train.py

**(3) For the runtime of some of the error resolution**
    1. If the validation of the training process reported an error (mainly some shape mismatch error this 
is due to the validation of the set of some of the special pictures lead to)
Find rect=mode in the build_dataset function in the DetectionTrainer class of 
ultralytics/models/yolo/detect/train.py

```py
2. Running the detect.py file while reasoning reported a shape mismatch error
Find ultralytics/engine/predictor.py and find the function def pre_transform(self, im), 
change auto to False in LetterBox.

3. The problem of mismatched types of errors reported during the training process
Find the file 'ultralytics/engine/validator.py' and find 'class
BaseValidator:' and then in its 'call ' self.args.half = self.device.type ! = 'cpu' # force FP16 val during training with
self.args.half = False below the line of code
```

**(4) For the nc changes in the yaml file**
No need to modify it, the model will automatically get it based on the profile of your dataset.
This is also the difference between printing the model twice, the first printout is the structure of
the yaml file you chose for the model, and the second printout is the yaml file that replaces your
dataset, the model uses the second one.

**(5) Environment-specific issues**
The environment is different for everyone who meets it, so you can find out for yourself online.





