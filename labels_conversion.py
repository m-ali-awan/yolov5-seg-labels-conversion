
import pandas as pd
import json 
from pathlib import Path
import shutil
import numpy as np


def read_coco_labels(json_pth):
  with open(str(json_pth)) as f:
    coco_annotations = json.load(f)
  return coco_annotations

def return_ann_list(image_id,coco_ann_df):
  temp_df = coco_ann_df.loc[coco_ann_df['image_id']==image_id].copy()
  l=[]
  for i,row in temp_df.iterrows():
    temp_dct = {}
    temp_dct['category_id']=row['category_id']
    temp_dct['bbox']=row['bbox']
    temp_dct['segmentation']=row['segmentation']
    l.append(temp_dct)
  return l


def saving_for_src_images(src_df,annotations_df,
                          src_root,none_or_not,yolo_root,split,increment_by_one=False):

  valid_splits={'train','valid','test'}
  if split not in valid_splits:
    raise ValueError(f"Split must be in {valid_splits}")
  cnt=0
  for filename in src_df['file_name'].to_list():
    if none_or_not==None:
      true_f_pth = src_root/'data'/filename
    else:
      true_f_pth = src_root/filename
    img_id = src_df.loc[src_df['file_name']==filename,'id'].iloc[0]
    height = src_df.loc[src_df['file_name']==filename,'height'].iloc[0]
    width = src_df.loc[src_df['file_name']==filename,'width'].iloc[0]

    ann_l = return_ann_list(img_id,annotations_df)
    # Checking if exists:
    to_check = yolo_root/split/'labels'
    to_check.mkdir(exist_ok=True, parents=True)
    try:
      dest_name = f"{yolo_root}/{split}/labels/{filename.split('.jpg')[0]}.txt"
    except:
      dest_name = f"{yolo_root}/{split}/labels/{filename.split('.jpeg')[0]}.txt"
    file_obj = open(str(dest_name),'a')
    for ann in ann_l:
      
      if increment_by_one:
        curr_cat = ann['category_id'] - 1
      else:
        curr_cat = ann['category_id']
      file_obj.write(f"{curr_cat} ")
      for cnt,one_seg in enumerate(ann['segmentation'][0]):

        if cnt%2==0:
          one_val = format(one_seg/width,'.6f')
        else:
          one_val = format(one_seg/height,'.6f')
        file_obj.write(f"{one_val} ")
      file_obj.write('\n')
    file_obj.close()
    # For copying images

  # Checking if exists:
    to_check = yolo_root/split/'images'
    to_check.mkdir(exist_ok=True, parents=True)
    try:
        shutil.copy(true_f_pth,str(f"{yolo_root}/{split}/images"))
    except Exception as e:
        print(f'image canot be copied due to:::{e}')
        pass


  print('---------------SUCCESS-------------')
  #print(yolo_root)
    




def convert_to_yolo_seg_labels(coco_labels_pth,
                                yolo_labels_root_pth,
                                images_src_pth=None,
                                train_percent=0.7,
                                val_percent=0.20,
                                increment_by_one=False):

  '''
  
  Takes COCO format Instance segmentation labels, and images, and save YOLO format labels, in Train, Val, and Test splits, at desired folder

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
  
  '''
  if images_src_pth==None:
    images_root= coco_labels_pth.parent
  else:
    images_root=Path(images_src_pth)


  yolo_labels_root_pth=Path(yolo_labels_root_pth)
  yolo_labels_root_pth.mkdir(exist_ok=True,parents=True)
  

  yolo_train_pth = yolo_labels_root_pth/'train'
  yolo_train_pth.mkdir(exist_ok=True)
  yolo_valid_pth = yolo_labels_root_pth/'valid'
  yolo_valid_pth.mkdir(exist_ok=True)
  yolo_test_pth = yolo_labels_root_pth/'test'
  yolo_test_pth.mkdir(exist_ok=True) 


  coco_annotations = read_coco_labels(coco_labels_pth)
  coco_ann_imgs_df = pd.DataFrame(coco_annotations['images'])
  coco_train_df,coco_valid_df,coco_test_df = np.split(coco_ann_imgs_df,
                                                      [int(train_percent * len(coco_ann_imgs_df)),
                                            int((train_percent + val_percent) * len(coco_ann_imgs_df)),
                                            ])
  coco_ann_df = pd.DataFrame(coco_annotations['annotations'])

  
  #for split in ['train','valid','test']:
  saving_for_src_images(coco_train_df,coco_ann_df,
                        images_root,images_src_pth,
                        yolo_labels_root_pth,
                        'train',increment_by_one)
  print('Data and labels for Train Saved')
  saving_for_src_images(coco_valid_df,coco_ann_df,
                        images_root,images_src_pth,
                        yolo_labels_root_pth,
                        'valid',increment_by_one)
  print('Data and labels for Validation Saved')
  saving_for_src_images(coco_test_df,coco_ann_df,
                        images_root,images_src_pth,
                        yolo_labels_root_pth,
                        'test',increment_by_one)
  print('Data and labels for Test Saved')

  # Now saving data.yaml
  selected_labels=[i['name'] for i in coco_annotations['categories']]

  yaml_obj = open(str(yolo_labels_root_pth/'data.yaml'),'a')
  yaml_obj.write('names:')
  yaml_obj.write('\n')


  for label in selected_labels:
    yaml_obj.write(f'- {label}')
    yaml_obj.write('\n')
  yaml_obj.write(f'nc: {len(selected_labels)} \n')
  yaml_obj.write(f'train: {yolo_labels_root_pth}/train/images \n')
  yaml_obj.write(f'val: {yolo_labels_root_pth}/valid/images \n')
  yaml_obj.write(f'test: {yolo_labels_root_pth}/test/images \n')
 

  yaml_obj.close()








