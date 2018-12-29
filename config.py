# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 17:22:57 2018

@author: LongJun
"""
import os
import numpy as np

PASCAL_PATH = 'dataset'

CACHE_PATH = 'annotation_cache'

BATCH_SIZE = 1

CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

#CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
#           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
#           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
#           'train', 'tvmonitor']

target_size = 600

max_size = 1000

#FLIPPED = False

MAX_ITER = 80000

lr_change_ITER = 60000

LEARNING_RATE = [0.001, 0.0001]

SUMMARY_ITER = 10

SAVE_ITER = 5000

overlaps_max = 0.7

overlaps_min = 0.3

OUTPUT_DIR = os.path.join('output', '2018_12_27_21_00')

momentum = 0.9

GPU_ID = '0,1'

anchor_scales = [128,256,512]

anchor_ratios = [0.5,1,2]

anchor_batch = 256

weight_path = os.path.join('model_pretrained', 'vgg_16.ckpt')

weigt_output_path = OUTPUT_DIR

test_output_path = 'test_output'

feat_stride = 16

PIXEL_MEANS = np.array([[[122.7717, 115.9465, 102.9801]]])

max_rpn_input_num = 12000

max_rpn_nms_num = 2000

test_max_rpn_input_num = 6000

test_max_rpn_nms_num = 300

nms_thresh = 0.7

dect_train_batch = 128

dect_fg_rate = 0.25

bbox_nor_target_pre = True

bbox_nor_mean = (0.0, 0.0, 0.0, 0.0)

bbox_nor_stdv = (0.1, 0.1, 0.2, 0.2)

roi_input_inside_weight = (1.0, 1.0, 1.0, 1.0)

POOLING_SIZE = 7

fg_thresh = 0.5

bg_thresh_hi = 0.5

bg_thresh_lo = 0.02

test_nms_thresh = 0.4

test_fp_tp_thresh = 0.5

test_max_per_image = 10

image_output_dir = os.path.join(test_output_path, 'image_output')
