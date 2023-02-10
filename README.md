# yolov5-seg-labels-conversion
Converting Instance segmentation labels in COCO format to YOLOv5-seg


### Currently I found, that only Roboflow allows us to convert instance Segmentation labels to YOLOv5-seg format. But for that, we have to make our dataset public, otherwise we have to go for **Upgrade**, and that costs too much.

So, here I have written this small utility, where we can pass our Instance-Segmentation labels in COCO 1.0 format, and it will convert the annotations to YOLOv5-seg format, with our desired Train, Val, and Test Splits.

## Usage Example:

```
from labels_conversion import *
 convert_to_yolo_seg_labels('COCO-format/test-coco-1/annotations/instances_default.json','YOLO-format/test-1',images_src_pth='COCO-format/test-coco-1/images',train_percent=0.4,val_percent=0.3,increment_by_one=True)
```

 ********
 
  Keyword Arguments:  
  ********
  coco_labels_pth: type= str | pathlib.Path; required.  the labels json file of COCO format instance segmentation labels
  
  yolo_labels_root_pth: type= str | pathlib.Path; required.  The Directory, where you want to store YOLO-Seg format labels. It can/can't exist
  
  images_src_pth: Optional , if not passed, it means Src images are stored in "data" folder, at same level where labels json file was present.
  Anyway, it is good to pass the path of images, to avoid confusion
  
  train_percent: type: float; required Percentage of data to put in Train Split
  
  val_percent: type: float; required Percentage of data to put in Val Split. Based on Train, and Val, Test split will be calculated
  
  increment_by_one type: bool; If COCO-labels Categories id start with 1, then it should be True, as we need to subtract 1 from CAT-Ids for YOLO format
  As they should start from 0
