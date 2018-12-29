# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 09:54:12 2018

@author: LongJun
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import config as cfg
import os
import pascal_voc as pascl
import numpy as np
import tensorflow as tf
import network
import datetime
import cv2
from nms import py_cpu_nms
#Solver类 网络训练用的类 暂时不用管
class Val_test(object):   
    def __init__(self, net ,val_data):
        self.net = net
        self.val_data = val_data
        self.overlaps_max = cfg.overlaps_max
        self.overlaps_min = cfg.overlaps_min
        self.ckpt_filename = tf.train.latest_checkpoint(cfg.OUTPUT_DIR)
        self.test_output_dir = cfg.test_output_path
        self.image_output_dir = cfg.image_output_dir
        txtname = os.path.join(self.val_data.data_path, 'ImageSets', 'Main', self.val_data.phase+'.txt')
        with open(txtname) as f:
            self.image_index = [x.strip() for x in f.readlines()]

    def test_model(self):
        saver = tf.train.Saver()
        _rois_coord = self.net.rois_coord[:,1:5]
        #rois_coord = self.net.rois_coord
        _pred_box = self.net.bbox_pred
        _pred_score = self.net.cls_prob
        #_pred_box_score_arg = tf.argmax(_pred_score, axis=1)
        dect_total_result = [[[] for i in range(self.val_data.num_gtlabels)] for j in range(self.net.num_classes)]
        with tf.Session() as sess:
            saver.restore(sess, self.ckpt_filename)
            for i in range (self.val_data.num_gtlabels):
                print (i)            
                train_data = self.val_data.get()
                image_height = np.array(train_data['image'].shape[1])
                image_width = np.array(train_data['image'].shape[2])
                feed_dict = {self.net.image: train_data['image'], self.net.image_width: image_width,\
                             self.net.image_height: image_height}
                                
                rois_coord, pred_box, pred_score= sess.run([_rois_coord, _pred_box, _pred_score],\
                                                                        feed_dict=feed_dict) #rois_coord roi的四个坐标:[n, 4]   pred_box预测的Tx Ty Tw Th:[n, 4*num_cls]   
                                                                                              #pred_score 每一个pred_box对应的score :[n, num_cls]  
                #pred_box_score_arg = pred_box_score_arg.astype(np.int32)
                #num_pred = pred_box_score_arg.shape[0]  #num_pred 一张图中所有预测的

                #pred_box_gather = np.empty([num_pred, 4], dtype = np.float32)
                #pred_score_gather = np.empty(num_pred)
                
                #for j in range(num_pred):
                #    pred_box_gather[j, :] = pred_box[j, 4*pred_box_score_arg[j]:4*(pred_box_score_arg[j]+1)]
                #    pred_score_gather[j] = pred_score[j, pred_box_score_arg[j]]
                    
                #pred_box_gather = pred_box_gather * np.array(cfg.bbox_nor_stdv) + np.array(cfg.bbox_nor_mean)
                #pre_box_coord = self.coord_transform_inv(rois_coord, pred_box_gather.astype(np.float32))
                #pre_box_coord = pre_box_coord/train_data['scale']

                for k in range(1, self.net.num_classes):    #修改成每一个类别的概率只要大于0就算预测
                    pre_class_arg = np.where(pred_score[:,k]>0)[0]
                    cls_pred_box_target = pred_box[pre_class_arg, k*4:(k+1)*4]
                    cls_pred_box_target = cls_pred_box_target * np.array(cfg.bbox_nor_stdv) + np.array(cfg.bbox_nor_mean)
                    cls_pred_box_coord = self.coord_transform_inv(rois_coord, cls_pred_box_target.astype(np.float32))
                    cls_pred_box_coord = cls_pred_box_coord/train_data['scale'] + 1.0
                    cls_pred_score = pred_score[pre_class_arg, k]
                    #print(cls_pred_box_coord.shape, cls_pred_score.shape)
                    cls_pred_score = cls_pred_score[:, np.newaxis]  #增加一个维度
                    cls_pred_target = np.concatenate((cls_pred_box_coord, cls_pred_score), axis=1)
                    keep = py_cpu_nms(cls_pred_target, cfg.test_nms_thresh)
                    cls_pred_target = cls_pred_target[keep, :]
                    dect_total_result[k][i] = cls_pred_target
                   # print (cls_pred_target)
                image_scores = np.hstack([dect_total_result[j][i][:, -1] for j in range(1, self.net.num_classes)]) #一张图片所有类别的分数的集合
                if len(image_scores) > cfg.test_max_per_image:
                    image_thresh = np.sort(image_scores)[-cfg.test_max_per_image] #找到阈值 也就是说每一个张图片上最多有100个预测值
                    for j in range(1, self.net.num_classes):
                        keep = np.where(dect_total_result[j][i][:, -1] >= image_thresh)[0]
                        dect_total_result[j][i] = dect_total_result[j][i][keep, :] #滤除掉多余的
            mean_ap = self.map_compute(dect_total_result)
            print ('the mean_ap of pascal_voc 2007 is', mean_ap)
            #    imname = train_data['imname']
            #    im_scale = train_data['scale']
            #    image = cv2.imread(imname)
            #    im = cv2.resize(image, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
            #    im = self.draw_result(im, dect_total_result)
                 #cv2.imshow('Image',im)
                 #cv2.waitKey(0)
            #    im_save_path = os.path.join(self.image_output_dir, '{:d}'.format(i)+'.jpg')
            #    cv2.imwrite

    def draw_result(self, img, result):
        for i in range(1, self.net.num_classes):
            #if result[i][0]
            print (111) 
            for j in range(result[i][0].shape[0]):
                print(111)
                x1 = int(result[i][0][j][0])
                y1 = int(result[i][0][j][1])
                x2 = int(result[i][0][j][2])
                y2 = int(result[i][0][j][3])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                cv2.rectangle(img, (x1, y1-20),(x2, y1), (125, 125, 125), -1)
                lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA
                cv2.putText(img, cfg.CLASSES[i] + ' : %.2f' % result[i][0][j][4],\
                (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,\
                (0, 0, 0), 1, lineType)                
        return img
        
        
    def coord_transform_inv (self, anchors, boxes):
        anchors = anchors.astype(np.float32)
        anchors = np.reshape(anchors, [-1,4])
        anchor_x = (anchors[:,2] + anchors[:,0]) * 0.5
        anchor_y = (anchors[:,3] + anchors[:,1]) * 0.5
        acnhor_w = (anchors[:,2] - anchors[:,0]) + 1.0
        acnhor_h = (anchors[:,3] - anchors[:,1]) + 1.0
        boxes = np.reshape(boxes, [-1,4])
        boxes_x = boxes[:,0]*acnhor_w + anchor_x
        boxes_y = boxes[:,1]*acnhor_h + anchor_y
        boxes_w = np.exp(boxes[:,2])*acnhor_w
        boxes_h = np.exp(boxes[:,3])*acnhor_h
        coord_x1 = boxes_x - boxes_w*0.5
        coord_y1 = boxes_y - boxes_h*0.5
        coord_x2 = boxes_x + boxes_w*0.5
        coord_y2 = boxes_y + boxes_h*0.5
        coord_result = np.stack([coord_x1, coord_y1, coord_x2, coord_y2], axis=1)
        return coord_result              
    
    def map_compute(self, dect_boxes):
        ap = []
        for cls_ind, cls in enumerate(self.val_data.classes):
            cls_obj = {}
            num_cls_obj = 0
            if cls == 'background':
                 continue
            cls_filename = os.path.join(self.test_output_dir, cls+'.txt')
            with open(cls_filename, 'w') as f:
                 for img_ind_dex, image_ind in enumerate(self.image_index):
                      dect_box = dect_boxes[cls_ind][img_ind_dex]
                      if dect_box == []:
                           continue
                      for i in range(dect_box.shape[0]):
                           f.write('{:s} {:2f} {:2f} {:2f} {:2f} {:3f} \n'.format\
                                   (image_ind, dect_box[i][0], dect_box[i][1], dect_box[i][2],\
                                    dect_box[i][3], dect_box[i][4]))
                           
            for gt_label in self.val_data.gt_labels:
                 gt_label_cls_ind = np.where(gt_label['gt_classs']==cls_ind)[0]
                 gt_label_pick_box = gt_label['boxes'][gt_label_cls_ind, :]
                 gt_label_pick_cls = gt_label['gt_classs'][gt_label_cls_ind]
                 diff_pick = gt_label['diff'][gt_label_cls_ind].astype(np.bool)
                 dec_id = [False] * gt_label_cls_ind.size
                 num_cls_obj = num_cls_obj + sum(~diff_pick)
                 cls_obj[gt_label['image_index']] = {'bbox': gt_label_pick_box,\
                                                      'cls': gt_label_pick_cls,\
                                                      'dec_id': dec_id, 'diff': diff_pick}
                 #print (num_cls_obj)
            with open(cls_filename, 'r') as f:
              lines = f.readlines()
            splitlines = [x.strip().split(' ') for x in lines] #每一行的信息都是indx+类别概率+四个坐标
            image_ids = [x[0] for x in splitlines]  # 找到所有的图片名称 注意 有重复的 有重复的 总数量是obj的数量 而不是image的数量
            confidence = np.array([float(x[5]) for x in splitlines]) #找到所有图片该类别的置信度
            BB = np.array([[float(z) for z in x[1:5]] for x in splitlines]) #bounding box                 
            
            nd = len(image_ids) #有的图片有有的图片没有                  
            tp = np.zeros(nd)         
            fp = np.zeros(nd)
            
            if BB.shape[0] > 0:
                 sorted_ind = np.argsort(-confidence) #不是一张图里的也排序？
                 BB = BB[sorted_ind, :]
                 image_ids = [image_ids[x] for x in sorted_ind] #图片名称也进行更改
                 for d in range(nd): #每一个测试值计算TP和FP
                      R = cls_obj[image_ids[d]] #找到该预测值 真值的gt信息
                      bb = BB[d, :].astype(float) #找到该预测值
                      ovmax = -np.inf #默认是错的
                      BBGT = R['bbox'].astype(float) #一张图中该类别的 gt bbox
                      
                      if BBGT.size > 0: #可能是空值 如果是空值也就是说ovmax的值默认就是错的
                           #计算所有的gt_box与预测值之间的IOU
                           ixmin = np.maximum(BBGT[:, 0], bb[0])
                           iymin = np.maximum(BBGT[:, 1], bb[1])
                           ixmax = np.minimum(BBGT[:, 2], bb[2])
                           iymax = np.minimum(BBGT[:, 3], bb[3])
                           iw = np.maximum(ixmax - ixmin + 1., 0.)
                           ih = np.maximum(iymax - iymin + 1., 0.)
                           inters = iw * ih
                           uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +\
                                  (BBGT[:, 2] - BBGT[:, 0] + 1.) *\
                                  (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
                           overlaps = inters / uni
                           ovmax = np.max(overlaps) #找到最大的IOU
                           jmax = np.argmax(overlaps) #找到最大IOU对应的obj
                           
                      if ovmax > cfg.test_fp_tp_thresh: #判断预测对还是预测错 
                           if not R['diff'][jmax]:
                               if not R['dec_id'][jmax]: #重复预测也算错
                                    tp[d] = 1.
                                    R['dec_id'][jmax] = 1 #
                               else:
                                    fp[d] = 1.
                      else:
                           fp[d] = 1. #小于阈值也算错
            fp = np.cumsum(fp) #计算所有的正确率 加和排序
            tp = np.cumsum(tp) #计算所有的错误了 加和排序
            rec = tp / float(num_cls_obj) #除所有的obj 不管是否是一张图片 召回率计算
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps) #计算准确率
            ap.append(self.val_data.voc_ap(rec, prec)) #将召回率与准确率传入	
            print (np.mean(ap))
        return sum(ap)/(self.net.num_classes - 1.0)
    
    
#获取需要restore 变量scope的名称                
    def get_var_list(self, global_variables, ckpt_variables):
        variables_to_restore = []
        for key in global_variables:
            if key.name.split(':')[0] in ckpt_variables:
                variables_to_restore.append(key) 
        return variables_to_restore
    
    

#主体代码的测试 测试能生成labels， 并进行筛选 选出256个positive 和negative的labels
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU_ID
    net = network.Net(is_training=False)
    val_data = pascl.pascal_voc('test', fliped=False)
    test = Val_test(net, val_data)
    print ('start training')
    test.test_model()
