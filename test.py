import os
os.environ['KERAS_BACKEND']='tensorflow'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import keras.backend as K
K.set_image_dim_ordering('tf')
from keras.applications import *
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import h5py

train_img_path = '/media/wac/backup/places365_Standard/places365_data'
val_img_path = '/media/wac/backup/places365_Standard/places365_data_tmp'
categories = '/media/wac/backup/places365_Standard/filelist_places365-standard/categories_places365_pure.txt'

f = open(categories, mode='r')
classes = [line.strip().split(' ')[0] for line in f.readlines()]
nb_classes = 365
image_size = (299, 299)
# image_size = (224, 224)


basemodel = InceptionV3()
# basemodel = ResNet50()
input = basemodel.input
x = basemodel.get_layer('avg_pool').output
# x = basemodel.get_layer('flatten_1').output
output = Dense(nb_classes, activation='softmax', name='prediction')(x)
model = Model(input, output)
model.load_weights('models_inception/weights.00034.hdf5')


gen = ImageDataGenerator()

data_generator = gen.flow_from_directory(val_img_path, image_size, classes=classes, shuffle=False,
                                             batch_size=32)

outs = model.predict_generator(data_generator, data_generator.samples/32)
print outs.shape

for i in range(outs.shape[0]):
    fname = data_generator.filenames[i]
    gt = data_generator.classes[i]
    label = np.argmax(outs[i])