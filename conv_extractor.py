import os
os.environ['KERAS_BACKEND']='tensorflow'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import keras.backend as K
K.set_image_dim_ordering('tf')
from keras.applications import *
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import h5py

train_img_path = '/media/wac/backup/places365_Standard/places365_data'
val_img_path = '/media/wac/backup/places365_Standard/val_data'
categories = '/media/wac/backup/places365_Standard/filelist_places365-standard/categories_places365_pure.txt'

f = open(categories, mode='r')
classes = [line.strip().split(' ')[0] for line in f.readlines()]

nb_classes = 365
# image_size = (299, 299)
image_size = (224, 224)

# basemodel = InceptionV3()
basemodel = ResNet50()
input = basemodel.input
# x = basemodel.get_layer('avg_pool').output
x = basemodel.get_layer('flatten_1').output
output = Dense(nb_classes, activation='softmax', name='prediction')(x)
basemodel_365 = Model(input, output)
# basemodel_365.load_weights('models_inception_prune_retrain/weights.00007.hdf5')
basemodel_365.load_weights('models_ResNet50_prune_retrain/weights.00009.hdf5')

input = basemodel_365.input
# x = basemodel_365.get_layer('mixed10').output
x = basemodel_365.get_layer('avg_pool').output
output = GlobalAveragePooling2D()(x)
model = Model(input, output)

gen = ImageDataGenerator()

data_generator = gen.flow_from_directory(train_img_path, image_size, classes=classes, shuffle=False,
                                             batch_size=32)

outs = model.predict_generator(data_generator, data_generator.samples/32)
print outs.shape

with h5py.File('gap_ResNet50_365_train_prune_retrain.h5') as h:
    h.create_dataset('data', data=outs)
    h.create_dataset('label', data=data_generator.classes)
    h.create_dataset('fname', data=data_generator.filenames)