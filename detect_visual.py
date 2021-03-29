import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os
import numpy as np
import matplotlib.pyplot as plt


def detect(opt, save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))
    output = ""
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name,
                    exist_ok=opt.exist_ok))  # increment run
    if (save_img or view_img or save_txt):
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                              exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # print(model)

    for name, param in model.named_parameters():

        # for layer in model.model:
        if 'conv.weight' in name:  # conv.weight
            weights = param
            print(name)  # , filters.shape)
            # print(weights)

            # normalize filter values between  0 and 1 for visualization
            f_min, f_max = weights.min(), weights.max()
            filters = (weights - f_min) / (f_max - f_min)
            print(filters.shape)
            filter_cnt = 1

            # plotting all the filters
            for i in range(filters.shape[2]):
                # get the filters
                #filt = filters[i, :, :, :]
                # plotting each of the channel, color image RGB channels
                for j in range(filters.shape[3]):
                    ax = plt.subplot(
                        filters.shape[2], filters.shape[3], filter_cnt)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.axis('off')
                    plt.imshow(filters[:, :, i, j], cmap='gray')
                    filter_cnt += 1
                    #print(filters[j, :, :])
                    # return

            # plt.show()
            plt.savefig('visualization/' + name + '.png', bbox_inches='tight')
            # return

    return


def detect2(source, weights, conf=0.5, iou_thres=0.45, device='', save_txt=False, save_conf=False, save_img=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default=weights, help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=source,
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=conf, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=iou_thres, help='IOU threshold for NMS')
    parser.add_argument('--device', default=device,
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    #parser.add_argument('--view-img', default=save_img, help='display results')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', default=save_txt,
                        help='save results to *.txt')
    #parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', default=save_conf,
                        help='save confidences in --save-txt labels')
    #parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements()

    # source, weights, view_img, save_txt, save_conf img_size, conf, iou_thresh, device,  = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size

    return detect(opt, save_img=save_img)


source_img = 'inference_pics/GOPR0212_16050881947191.JPG'
source_img2 = 'inference_pics/IMG_20210303_104504.jpg'
weights = 'best.pt'

# conf is the confidence threshold of the detection
# iou_threshold is the area of overlap
# device, set to '' for gpu, 'cpu' for cpu
# save_txt=True saves a text file with coordinates
# save_conf=True adds the confidence level to the coordinate output
# save_img=True saves the image with boundingboxes

# for fastest result use device='', save_txt=False, save_conf=True, save_img=False
coords = detect2(source_img, weights, conf=0.7, iou_thres=0.45,
                 device='cpu', save_txt=False, save_conf=True, save_img=False)
print(coords)
