import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pickle
import copy
import config as cfg


class pascal_voc(object):
    def __init__(self, phase, fliped, rebuild=False):
        self.devkil_path = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit')   #pasval_voc路径
        self.data_path = os.path.join(self.devkil_path, 'VOC2007')  #pascal_voc 2007路径
        self.cache_path = cfg.CACHE_PATH    #缓存路径
        self.batch_size = cfg.BATCH_SIZE    #batch_size
        self.target_size = cfg.target_size  #图片的最小尺寸
        self.max_size = cfg.max_size    #图片的最大尺寸
        self.classes = cfg.CLASSES  #类别信息  ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus'....]
        self.pixel_means = cfg.PIXEL_MEANS  #背景像素
        self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))   #构造class字典
        self.flipped = fliped  #图片是否翻转
        self.phase = phase  #ImageSet 的名称
        self.rebuild = rebuild   #是否重新简历缓存
        self.cursor = 0    #当前游标
        self.epoch = 1     #当前的epoch
        #self.gt_labels = None
        self.prepare()
        self.num_gtlabels = len(self.gt_labels)
    
    def image_read(self, imname, flipped=False):
        image = cv2.imread(imname)  #opencv 中默认图片色彩格式为BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) #将图片转成RGB格式
        if flipped:
            image = image[:, ::-1, :]
        return image
    
    def get(self): #在get中完成 self.epoch+1的操作
        #images = np.zeros((self.batch_size, self.image_size, self.image_size, 3))
        #gt_box = np.zeros((self.batch_size, 4), dtype=np.uint16)
        #gt_cls = np.zeros((num_objs), dtype=np.int32)
        count = 0
        tf_blob = {}
        assert self.batch_size == 1, "only support single batch" 
        while count < self.batch_size:
            imname = self.gt_labels[self.cursor]['imname']
            flipped = self.gt_labels[self.cursor]['flipped']
            image = self.image_read(imname, flipped=flipped)
            image, image_scale = self.prep_im_for_blob(image, self.pixel_means, self.target_size, self.max_size)#resize后的image
            image = np.reshape(image, (self.batch_size, image.shape[0], image.shape[1], 3)) #将image 转化成tensorflow输入的形式
            gt_box = self.gt_labels[self.cursor]['boxes'] * image_scale #将gt_box sclae与scale相乘 boxes.shape=[num_obj,4]
            gt_cls = self.gt_labels[self.cursor]['gt_classs']
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.gt_labels):
                np.random.shuffle(self.gt_labels)
                self.cursor = 0
                self.epoch += 1
        tf_blob = {'image':image, 'scale':image_scale, 'cls':gt_cls, 'box': gt_box, 'imname': imname}
        return tf_blob #返回的image.shape=[batch,size,size,3] image_scale, gt_box.shape=[num_objs,4]

    

    def prepare(self):
        gt_labels = self.load_labels()
        if self.flipped:
            print('Appending horizontally-flipped training examples ...') #{'boxes':boxes, 'gt_classs':gt_classes, 'imname':imname}组成的list
            gt_labels_cp = copy.deepcopy(gt_labels) #很重要
            for idx in range(len(gt_labels_cp)):
                gt_labels_cp[idx]['flipped'] = True
                width_pre = copy.deepcopy(gt_labels_cp[idx]['boxes'][:,0])
                gt_labels_cp[idx]['boxes'][:,0] = gt_labels_cp[idx]['image_size'][0] - gt_labels_cp[idx]['boxes'][:,2]
                gt_labels_cp[idx]['boxes'][:,2] = gt_labels_cp[idx]['image_size'][0] - width_pre
#                gt_labels_cp[idx]['boxes'][:,[0,2]] = gt_labels_cp[idx]['image_size'][0] - gt_labels_cp[idx]['boxes'][:,[0,2]][:,::-1]
            gt_labels += gt_labels_cp
        if self.phase == 'train':
            np.random.shuffle(gt_labels)
        self.gt_labels = gt_labels
        #return gt_labels

    def load_labels(self):
        cache_file = os.path.join(
            self.cache_path, 'pascal_' + self.phase + '_gt_labels.pkl')

        if os.path.isfile(cache_file) and not self.rebuild:
            print('Loading gt_labels from: ' + cache_file)
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)  #从.pkl文件中反序列对象
            return gt_labels

        print('Processing gt_labels from: ' + self.data_path)

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        if self.phase == 'train':
            txtname = os.path.join(
                self.data_path, 'ImageSets', 'Main', 'trainval.txt')
        else:
            txtname = os.path.join(
                self.data_path, 'ImageSets', 'Main', self.phase + '.txt')
            #self.flipped = False
        with open(txtname, 'r') as f:
            self.image_index = [x.strip() for x in f.readlines()]

        gt_labels = []
        for index in self.image_index:
            gt_label = self.load_pascal_annotation(index) #groundtruth_roidb 包括objet box坐标信息 以及类别信息(转换成dict后的)
            gt_labels.append(gt_label)
        print('Saving gt_labels to: ' + cache_file)
        with open(cache_file, 'wb') as f:
            pickle.dump(gt_labels, f)
        return gt_labels

    def load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        image_size = tree.find('size')
        size_info = np.zeros((2,), dtype=np.float32)
        size_info[0] = float(image_size.find('width').text)
        size_info[1] = float(image_size.find('height').text)
        num_objs = len(objs) #object的数量
        boxes = np.zeros((num_objs, 4), dtype=np.float32) #boxes 坐标 (num_objs,4)个 dtype=np.uint16
        gt_classes = np.zeros((num_objs), dtype=np.int32) #class 的数量num_objs个 dtype=np.int32 应该是groundtruth中读到的class
        difficult = np.empty((num_objs))
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self.class_to_ind[obj.find('name').text.lower().strip()] #找到class对应的类别信息
            boxes[ix, :] = [x1, y1, x2, y2] #注意boxes是一个np类的矩阵 大小为[num_objs,4]
            gt_classes[ix] = cls #将class信息存入gt_classses中，注意gt_classes也是一个np类的矩阵 大小为[num_objs] 是int值 对应于name
            imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
            difficult[ix] = int(obj.find('difficult').text)
        return {'boxes':boxes, 'gt_classs':gt_classes, 'imname':imname, 'flipped':False, 'image_size':size_info, 'image_index': index, 'diff': difficult}
    
    def prep_im_for_blob(self, im, pixel_means, target_size, max_size): #传入image 背景 600 1000
            im = im.astype(np.float32, copy=False)
            im -= pixel_means #去掉背景
            im_shape = im.shape
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])
            im_scale = float(target_size) / float(im_size_min) #600/最短边
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > max_size:
                im_scale = float(max_size) / float(im_size_max)
            im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,interpolation=cv2.INTER_LINEAR)
            return im, im_scale #返回im 和 im_scale

    def voc_ap(self, rec, prec): #使用10年之后的pascal_voc的map计算方式
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i-1] = np.maximum(mpre[i-1], mpre[i])
        
        i = np.where(mrec[1:] != mrec[:-1])[0] #取所有与取倒数第一个之间
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]) #计算ap
        
        return ap
        
if __name__ == '__main__':
    pascal = pascal_voc('test')
    tf_blob = pascal.get()
    print (len(pascal.gt_labels))
    #print (len(pascal.gt_labels))
