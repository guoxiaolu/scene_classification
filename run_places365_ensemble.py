# -*- coding:utf-8 -*-
import os
os.environ['KERAS_BACKEND']='tensorflow'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import keras.backend as K
K.set_image_dim_ordering('tf')
from keras.applications import *
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import h5py
import time
from scipy.misc import imread, imresize

Resnet50_weights_path = 'models_ResNet50_prune_retrain/weights.00009.hdf5'
InceptionV3_weights_path = 'models_inception_prune_retrain/weights.00007.hdf5'
Ensemble_weights_path = 'models_ensemble(inception+resnet50)_prune_retrain/weights.00299.hdf5'
categories = '/media/wac/backup/places365_Standard/filelist_places365-standard/categories_places365_pure.txt'

f = open(categories, mode='r')
classes = [line.strip().split(' ')[0] for line in f.readlines()]

nb_classes = 365

def ResNet50_load(weights_path):
    basemodel = ResNet50()
    basemodel.summary()
    input = basemodel.input
    x = basemodel.get_layer('flatten_1').output
    output = Dense(nb_classes, activation='softmax', name='prediction')(x)
    basemodel_365 = Model(input, output)

    basemodel_365.load_weights(weights_path)

    input = basemodel_365.input
    x = basemodel_365.get_layer('avg_pool').output
    output = GlobalAveragePooling2D()(x)
    model = Model(input, output)
    return model

def InceptionV3_load(weights_path):
    basemodel = InceptionV3()
    basemodel.summary()
    input = basemodel.input
    x = basemodel.get_layer('avg_pool').output
    output = Dense(nb_classes, activation='softmax', name='prediction')(x)
    basemodel_365 = Model(input, output)
    basemodel_365.load_weights(weights_path)

    input = basemodel_365.input
    x = basemodel_365.get_layer('mixed10').output
    output = GlobalAveragePooling2D()(x)
    model = Model(input, output)
    return model

def Ensemble_load(weights_path):
    input_tensor = Input((4096,))
    x = input_tensor
    x = Dropout(0.5)(x)
    x = Dense(nb_classes, activation='softmax')(x)
    model = Model(input_tensor, x)
    model.load_weights(weights_path)
    return model

ResNet50_model = ResNet50_load(Resnet50_weights_path)
InceptionV3_model = InceptionV3_load(InceptionV3_weights_path)
ensemble_model = Ensemble_load(Ensemble_weights_path)
test_dir = '/media/wac/backup1/places365_project/咖啡店/'
files = os.listdir(test_dir)
for file in files:
    start = time.time()
    f = os.path.join(test_dir, file)
    img = imread(f)
    ResNet50_img = imresize(img, (224,224))
    ResNet50_output = ResNet50_result = ResNet50_model.predict(ResNet50_img.reshape(1, ResNet50_img.shape[0], ResNet50_img.shape[1], ResNet50_img.shape[2]))

    InceptionV3_img = imresize(img, (299, 299))
    InceptionV3_output = InceptionV3_model.predict(InceptionV3_img.reshape(1, InceptionV3_img.shape[0], InceptionV3_img.shape[1], InceptionV3_img.shape[2]))

    output = ensemble_model.predict(np.concatenate((InceptionV3_output, ResNet50_output), axis=1))
    id = np.argmax(output)
    end = time.time()
    print '%s\t%s\t%f\t%f'%(file, classes[id], output[0, id], end - start)