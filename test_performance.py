import argparse
import os
from os.path import isfile, join
import random
import tempfile
import time
import copy
import multiprocessing
import subprocess
import shutil
import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import torchvision
from models import Darknet
from utils.datasets import ImageLabelDataset
from utils.nms import nms
from utils.utils import xywh2xyxy, calculate_padding

import warnings
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from pathlib import Path, PureWindowsPath
import glob
from timeit import default_timer as timer
import numpy as np
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

warnings.filterwarnings("ignore")


def extract_gt(gt_path, w, h):
    """
    Extract the ground truth  data and prapare a list of lists for for the calcolation.
    Prapare a list of gt_box as [xmin, ymin, xmax, ymax] and add it to the gt_boxes list.
    Args:
        gt_path : path to image ground truth txt file.
        w: width of the image
        h: hight of the image

    Returns:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
    """
    gt_boxes = []
    gt_file = open(gt_path)
    # opening the text file 
    with open(gt_path,'r') as gt_file:     
        for line in gt_file:
            data = line.split() 
            # YOLOv3 each row format is: class, x_center, y_center, width, height
            # Box coordinates must be in normalized xywh format (from 0 - 1). 
            # to retrive correct pixel: multiply x_center and width by image width
            # and y_center and height by image height.
            x_center = float(data[1]) * w
            y_center = float(data[2]) * h
            bb_width = float(data[3]) * w
            bb_height = float(data[4]) * h

            xmin = x_center - bb_width/2
            xmax = x_center + bb_width/2 
            ymin = y_center - bb_height/2
            ymax = y_center + bb_height/2     
            gt_box = [xmin, ymin, xmax, ymax]
            gt_boxes.append(gt_box)
                
    return gt_boxes

def calc_iou( gt_bbox, pred_bbox):
    '''
    Calculate the IoU for bounding boxes with Pascal VOC format:
    (x-top left, y-top left,x-bottom right, y-bottom right)
    This function takes the predicted bounding box and ground truth bounding box and 
    return the IoU ratio
    '''
    x_topleft_gt, y_topleft_gt, x_bottomright_gt, y_bottomright_gt = gt_bbox
    x_topleft_p, y_topleft_p, x_bottomright_p, y_bottomright_p = pred_bbox
    
    if (x_topleft_gt > x_bottomright_gt) or (y_topleft_gt> y_bottomright_gt):
        raise AssertionError("Ground Truth Bounding Box is not correct")
    if (x_topleft_p > x_bottomright_p) or (y_topleft_p> y_bottomright_p):
        raise AssertionError("Predicted Bounding Box is not correct",x_topleft_p, x_bottomright_p,y_topleft_p,y_bottomright_gt)
        
         
    #if the GT bbox and predcited BBox do not overlap then iou=0
    if(x_bottomright_gt< x_topleft_p):
        # If bottom right of x-coordinate  GT  bbox is less than or above the top left of x coordinate of  the predicted BBox
        
        return 0.0
    if(y_bottomright_gt< y_topleft_p):  # If bottom right of y-coordinate  GT  bbox is less than or above the top left of y coordinate of  the predicted BBox
        
        return 0.0
    if(x_topleft_gt> x_bottomright_p): # If bottom right of x-coordinate  GT  bbox is greater than or below the bottom right  of x coordinate of  the predcited BBox
        
        return 0.0
    if(y_topleft_gt> y_bottomright_p): # If bottom right of y-coordinate  GT  bbox is greater than or below the bottom right  of y coordinate of  the predcited BBox
        
        return 0.0
    
    
    GT_bbox_area = (x_bottomright_gt -  x_topleft_gt + 1) * (  y_bottomright_gt -y_topleft_gt + 1)
    Pred_bbox_area =(x_bottomright_p - x_topleft_p + 1 ) * ( y_bottomright_p -y_topleft_p + 1)
    
    x_top_left =np.max([x_topleft_gt, x_topleft_p])
    y_top_left = np.max([y_topleft_gt, y_topleft_p])
    x_bottom_right = np.min([x_bottomright_gt, x_bottomright_p])
    y_bottom_right = np.min([y_bottomright_gt, y_bottomright_p])
    
    intersection_area = (x_bottom_right- x_top_left + 1) * (y_bottom_right-y_top_left  + 1)
    
    union_area = (GT_bbox_area + Pred_bbox_area - intersection_area)
   
    return intersection_area/union_area

def get_single_image_results(gt_boxes, pred_boxes, iou_thr=0.5):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
       tp: true positives (int)
       fp: false positives (int)
       fn: false negatives (int)
    """
    all_pred_indices= range(len(pred_boxes))
    all_gt_indices=range(len(gt_boxes))
    if len(all_pred_indices)==0:
        tp=0
        fp=0
        fn=len(gt_boxes)
        # return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
        return tp, fp, fn
    if len(all_gt_indices)==0:
        tp=0
        fp=len(pred_boxes)
        fn=0
        # return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
        return tp, fp, fn
 
    gt_idx_thr=[]
    pred_idx_thr=[]
    ious=[]
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou= calc_iou(gt_box, pred_box)
 
            if iou >iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)
    # Getting the sorted indexes by an descending order
    iou_sort = np.argsort(ious)[::-1]
    if len(iou_sort)==0:
        tp=0
        fp=len(pred_boxes)
        fn=len(gt_boxes)
        # return {'true_positive':tp, 'false_positive':fp, 'false_negative':fn}
        return tp, fp, fn
    else:
        gt_match_idx=[]
        pred_match_idx=[]
        for idx in iou_sort:
            gt_idx=gt_idx_thr[idx]
            pr_idx= pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if(gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)

    # return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}
    return tp, fp, fn

def cones_detection(target_path,output_path,model,device,conf_thres,nms_thres):

    img = Image.open(target_path).convert('RGB')
    w, h = img.size
    new_width, new_height = model.img_size()
    pad_h, pad_w, ratio = calculate_padding(h, w, new_height, new_width)
    img = torchvision.transforms.functional.pad(img, padding=(pad_w, pad_h, pad_w, pad_h), fill=(127, 127, 127), padding_mode="constant")
    img = torchvision.transforms.functional.resize(img, (new_height, new_width))

    bw = model.get_bw()
    if bw:
        img = torchvision.transforms.functional.to_grayscale(img, num_output_channels=1)

    img = torchvision.transforms.functional.to_tensor(img)
    img = img.unsqueeze(0)
    
    with torch.no_grad():
        model.eval()
        img = img.to(device, non_blocking=True)
        # output,first_layer,second_layer,third_layer = model(img)
        output = model(img)

        for detections in output:
            detections = detections[detections[:, 4] > conf_thres]
            box_corner = torch.zeros((detections.shape[0], 4), device=detections.device)
            xy = detections[:, 0:2]
            wh = detections[:, 2:4] / 2
            box_corner[:, 0:2] = xy - wh
            box_corner[:, 2:4] = xy + wh
            probabilities = detections[:, 4]
            nms_indices = nms(box_corner, probabilities, nms_thres)
            main_box_corner = box_corner[nms_indices]
            if nms_indices.shape[0] == 0:  
                continue

        pred_boxes = []
        for i in range(len(main_box_corner)):
            x0 = main_box_corner[i, 0].to('cpu').item() / ratio - pad_w
            y0 = main_box_corner[i, 1].to('cpu').item() / ratio - pad_h
            x1 = main_box_corner[i, 2].to('cpu').item() / ratio - pad_w
            y1 = main_box_corner[i, 3].to('cpu').item() / ratio - pad_h 
            box = [x0, y0, x1, y1]
            pred_boxes.append(box)

        return pred_boxes
 
def main(target_path,output_path,weights_path,model_cfg,conf_thres,nms_thres,xy_loss,wh_loss,no_object_loss,object_loss,vanilla_anchor):
    """
        Testing the performance of the network model in following aspects:
        detection duration, precision (positive predictive value) and recall (sensitivity).
    Args:
        target_path: testset file location with images and ground truth txt files.
        output_path: output file location where each image result will be saved.
        weights_path: the path to the tested model weight file.
        model_cfg: the path to the tested model cfg file.
    Returns:
        Saves each test image in the output file location with the bb detection, precision and recall results.
        Prints each test image detetion duration. 
    """
    # Initializing the model
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    random.seed(0)
    torch.manual_seed(0)
    if cuda:
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    model = Darknet(config_path=model_cfg,xy_loss=xy_loss,wh_loss=wh_loss,no_object_loss=no_object_loss,object_loss=object_loss,vanilla_anchor=vanilla_anchor)

    # Load weights
    model.load_weights(weights_path, model.get_start_weight_dim())
    model.to(device, non_blocking=True)


    # Get the images from the folder
    images = glob.glob(f'{target_path}/*.png')
    precisions = []
    recalls = [] 
    # Looping over all images in the testset
    for idx, fname in enumerate(images):

        img_name = Path(fname).stem
        gt_path = Path(f"{target_path}/{img_name}.txt")

        # Prepare output image with BB with precision and recall
        img_with_boxes = Image.open(fname)
        draw = ImageDraw.Draw(img_with_boxes)  # get a drawing context
        w, h = img_with_boxes.size

        gt_boxes = extract_gt(gt_path, w, h)
        t1 = timer()
        pred_boxes = cones_detection(target_path=fname,output_path=output_path,model=model,device=device,conf_thres=conf_thres,nms_thres=nms_thres)
        t2 = timer()
        print(f"Image {idx+1} detection took {round((t2-t1)*1000)} [ms]")
        tp, fp, fn = get_single_image_results(gt_boxes, pred_boxes)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        precisions.append(precision)
        recalls.append(recall)

        # Draw predicted BB on the image
        for box in pred_boxes:
            x0 = box[0]
            y0 = box[1]
            x1 = box[2]
            y1 = box[3]
            draw.rectangle((x0, y0, x1, y1), outline="red")
        # # Draw ground truth BB on the image (for debugging)
        # for box in gt_boxes:
        #     x0 = box[0]
        #     y0 = box[1]
        #     x1 = box[2]
        #     y1 = box[3]
        #     draw.rectangle((x0, y0, x1, y1), outline="green")

        # draw text, full opacity
        fnt = ImageFont.truetype(font="arial.ttf", size=40) # get a font
        draw.text((10,10), f"Precision: {precision:.3f}, Recall: {recall:.3f}", fill=(0,255,0,255), font=fnt)

        # img_with_boxes.save(os.path.join(output_path,target_path.split('/')[-1]))
        img_with_boxes.save(os.path.join(output_path,os.path.basename(fname)))

    precision_score = np.mean(precisions)
    recall_score = np.mean(recalls)
    print(f"Model precision score: {precision_score}, recall score: {recall_score}")


#  run in terminal:
# test_visualization.py --model_cfg='model_cfg/yolo_baseline.cfg' --target_path='dataset/YOLO_Testset' --weights_path='weights/YOLOv3_1.weights' --output_path='outputs/visualization/out_test_1/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    def add_bool_arg(name, default, help):
        arg_group = parser.add_mutually_exclusive_group(required=False)
        arg_group.add_argument('--' + name, dest=name, action='store_true', help=help)
        arg_group.add_argument('--no_' + name, dest=name, action='store_false', help=("Do not " + help))
        parser.set_defaults(**{name:default})
    parser.add_argument('--model_cfg', type=str, default='model_cfg/yolo_baseline.cfg')
    parser.add_argument('--target_path', type=str, default='dataset/YOLO_Testset', help='path to target image/video')
    parser.add_argument('--output_path', type=str, default="outputs/visualization/out_test_1/")
    parser.add_argument('--weights_path', type=str, default='weights/YOLOv3_1.weights',help='path to weights file')
    parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
    parser.add_argument('--nms_thres', type=float, default=0.25, help='IoU threshold for non-maximum suppression')

    add_bool_arg('vanilla_anchor', default=False, help="whether to use vanilla anchor boxes for training")
    ##### Loss Constants #####
    parser.add_argument('--xy_loss', type=float, default=2, help='confidence loss for x and y')
    parser.add_argument('--wh_loss', type=float, default=1.6, help='confidence loss for width and height')
    parser.add_argument('--no_object_loss', type=float, default=25, help='confidence loss for background')
    parser.add_argument('--object_loss', type=float, default=0.1, help='confidence loss for foreground')

    opt = parser.parse_args()

    main(target_path=opt.target_path,
         output_path=opt.output_path,
         weights_path=opt.weights_path,
         model_cfg=opt.model_cfg,
         conf_thres=opt.conf_thres,
         nms_thres=opt.nms_thres,
         xy_loss=opt.xy_loss,
         wh_loss=opt.wh_loss,
         no_object_loss=opt.no_object_loss,
         object_loss=opt.object_loss,
         vanilla_anchor=opt.vanilla_anchor)
