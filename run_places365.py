# -*- coding:utf-8 -*-
import _init_paths
import numpy as np
import sys
import caffe
import pickle
import cv2
import os
from io import BytesIO
import urllib
import urllib2
import time

fpath_design = './deploy_alexnet_places365.prototxt'
fpath_weights = './alexnet_places365.caffemodel'
ilsvrc_2012_mean = './ilsvrc_2012_mean.npy'
fpath_labels = './labels.pkl'

fpath_design = '/media/wac/backup/places365/caffe_relate/deploy_alexnet_places365.prototxt'
fpath_weights = '/media/wac/backup/places365/caffe_relate/alexnet_places365.caffemodel'
# ilsvrc_2012_mean = '/media/wac/backup/places365/places365CNN_mean_224.binaryproto'

def get_places365_net():
    # fpath_design = './deploy_alexnet_places365.prototxt'
    # fpath_weights = './alexnet_places365.caffemodel'

    # initialize net
    net = caffe.Net(fpath_design, fpath_weights, caffe.TEST)

    # load input and configure preprocessing
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    # transformer.set_mean('data', np.load('./ilsvrc_2012_mean.npy').mean(1).mean(1))
    transformer.set_mean('data', np.load(ilsvrc_2012_mean).mean(1).mean(1))
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_channel_swap('data', (2, 1, 0))
    transformer.set_raw_scale('data', 255.0)

    # since we classify only one image, we change batch size from 10 to 1
    net.blobs['data'].reshape(1, 3, 227, 227)

    return net, transformer

places365_net, places365_transformer = get_places365_net()

def places365_detection(im_Bytes, net=places365_net, transformer=places365_transformer):
    # im_Bytes 可以是本地图片地址，或者是从url中读取的图片
    im = caffe.io.load_image(im_Bytes)

    # load the image in the data layer
    net.blobs['data'].data[...] = transformer.preprocess('data', im)
    out = net.forward()
    fpath_labels = './labels.pkl'
    with open(fpath_labels, 'rb') as f:
        labels = pickle.load(f)
        # print labels
        prob = net.blobs['prob'].data[0].flatten()
        top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-3:-1]
        # print top_k

        result = []
        for i, k in enumerate(top_k):
            result.append((labels[k], prob[k]))
            # print i, labels[k], prob[k]
        # result格式 [('lecture_room', 0.084788859), ('conference_center', 0.063725658)]  两个set里面前面是label，后面是概率
        return result

if __name__ == '__main__':
    # test images directory
    test_dir = './酒吧/'
    files = os.listdir(test_dir)
    result = dict()
    for file in files:
        start = time.time()
        result[file] = places365_detection(test_dir + file)
        end = time.time()
        print '%s\t%s\t%f\t%f'%(file, result[file][0][0], result[file][0][1], end - start)




