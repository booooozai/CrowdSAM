import torch
import json
import os
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
from .coco_names import coco_classes
data_meta = {'crowdhuman':{1:'person'},
             'occhuman': {1:'person'},
             'coco_occ':coco_classes,
             'coco':  coco_classes, 
             'cityscape': {1:"person", 2:"rider", 3:"car", 4:"truck", 5:"bus", 6:"train", 7:"motorcycle", 8:"bicycle"}
             }

def load_img_and_annotation(dataset_path, annots, dataset, id=0):
  
    #load image
    image_cv = cv2.imread(img_path)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    bboxes = np.array([ annot['bbox'] for annot in annots['annotations'] if annot['image_id'] ==img_meta['id']])
    bboxes[...,2:] += bboxes[...,:2]
    img_id = img_meta['id']    
    return image_cv, bboxes, img_id 

class COCODataset(torch.utils.data.Dataset):
    def __init__(self,image_directory, annotation_file, transform=None):
        """
        Initialize COCO dataset
        
        Args:
            annotation_file (str): Path to COCO annotation file (JSON)
            image_directory (str): Path to directory containing images
            transform (callable, optional): Optional transform to be applied to the image
        """
        self.coco = COCO(annotation_file)
        self.image_directory = image_directory
        self.transform = transform
        self.image_ids = self.coco.getImgIds()
        self.mapping = self.create_category_mapping()
    def __len__(self):
        """
        Return the total number of samples in the dataset
        """
        return len(self.image_ids)
    def create_category_mapping(self):
        """
        Create a mapping from COCO category IDs to a continuous range from 0 to 80
        
        Returns:
            dict: Mapping from COCO category IDs to a continuous range from 0 to 80
        """
        coco_categories = self.coco.loadCats(self.coco.getCatIds())
        coco_category_ids = sorted([category['id'] for category in coco_categories])
        category_mapping = {coco_id: idx for idx, coco_id in enumerate(coco_category_ids)}
        return category_mapping
    def __getitem__(self, idx):
        """
        Get the sample at the specified index
        
        Args:
            idx (int): Index of the sample to retrieve
        
        Returns:
            dict: Dictionary containing the image and its associated masks and categories
        """
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = f"{self.image_directory}/{image_info['file_name']}"
        image = Image.open(image_path)

        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)
        if len(annotations)  == 0:
            sample = {
            'image_id': image_id,
            'image': image,
            'masks': None,
            'boxes' : np.zeros((0,4)), #xmin,ymin,xmax,ymax
            'categories': []
            }
            return sample


        if 'segmentation' in annotations[0]:
            masks = [self.coco.annToMask(annotation) for annotation in annotations]
        else:
            masks = None
        boxes = [annotation['bbox'] for annotation in annotations]
        boxes = np.array(boxes)
        boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]
        category_ids = [self.mapping[annotation['category_id']] for annotation in annotations]

        sample = {
            'image_id': image_id,
            'image': image,
            'masks': masks,
            'boxes' : boxes, #xmin,ymin,xmax,ymax
            'categories': category_ids
        }


        return sample


def collate_fn_crowdhuman(data):
    images = [d['image'] for d in  data]
    boxes = [d['boxes'] for d in data]
    # images,boxes, boxe_full= zip(*data.values())
    return images, boxes

def collate_fn_coco(data):
    images = [d['image'] for d in  data]
    masks = [d['masks'] for d in data]
    categories = [d['categories'] for d in data]
    # images,boxes, boxe_full= zip(*data.values())
    return images, masks, categories

