import os
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class PennFudanDataset(object):
    def __init__(self,root,transforms):
        self.root = root
        self.transforms = transforms
        # download all these image files and sort them to ensure alignment
        self.imgs =list(sorted(os.listdir(os.path.join(root,"PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root,"PedMasks"))))

    def __getitem__(self,idx):
        # load images and masks
        img_path = os.path.join(self.root,"PNGImages",self.imgs[idx])
        mask_path = os.path.join(self.root,"PedMasks",self.masks[idx])
        img = Image.open(img_path).convert("RGB")

        mask = Image.open(mask_path)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        # the first id 0 is background
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:,None,None]

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(mask[i])
            xmin = np.min(pos[1])
            ymin = np.min(pos[0])
            xmax = np.max(pos[1])
            ymax = np.max(pos[0])
            boxes.append([xmin,ymin,xmax,ymax])
        
        boxes = torch.as_tensor(boxes,dtype = torch.float32)
        labels = torch.ones((num_objs,),dtype = torch.int64)
        masks = torch.as_tensor(masks,dtype = torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:,3] - boxes[:,1]) * (boxes[:,2] - boxes[:,0])

        # assume all instances are not crowds
        iscrowd = torch.zeros((num_objs,),dtype = torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            img,target = self.transforms(img,target)
        
        return img,target
    
    def __len__(self):
        return len(self.imgs)

def get_model_instance_segmentation(num_classes):
    # load the instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained = True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pretrained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer  = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,hidden_layer,num_classes)
    return model
