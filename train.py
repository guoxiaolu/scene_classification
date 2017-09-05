import os
os.environ['KERAS_BACKEND']='tensorflow'
import keras.backend as K
K.set_image_dim_ordering('tf')
from keras.applications import *
from keras.models import Model
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from time import time


# train_file = '/media/wac/backup/places365_Standard/filelist_places365-standard/places365_train_standard_filter.txt'
# val_file = '/media/wac/backup/places365_Standard/filelist_places365-standard/places365_val.txt'
train_img_path = '/media/wac/backup/places365_Standard/places365_data'
val_img_path = '/media/wac/backup/places365_Standard/val_data'
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

model.summary()

gen = ImageDataGenerator()
train_generator = gen.flow_from_directory(train_img_path, target_size=image_size, classes=classes, shuffle=True,
                                          batch_size=32)

val_generator = gen.flow_from_directory(val_img_path, target_size=image_size, classes=classes, shuffle=True,
                                          batch_size=32)

sgd = SGD(lr=0.001, momentum=0.9, decay=1e-3, nesterov=False)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.load_weights('./models_inception/weights.00034.hdf5')
# tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
# mc = ModelCheckpoint('./models/weights.{epoch:05d}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
# model.fit_generator(train_generator, train_generator.classes.size/32, nb_epoch=100, callbacks=[tb, mc], validation_data=val_generator, validation_steps=val_generator.classes.size/32, initial_epoch=26)
#
# model.save('InceptionV3_model.h5')

start = time()
score = model.evaluate_generator(val_generator, val_generator.classes.size/32)
cost = time() - start
print('Test score:', score[0])
print('Test accuracy:', score[1])
print('Time:', cost)

####################
# sparse
weight_th = 0.005
for i, layer in enumerate(model.layers):
    weights = model.layers[i].get_weights()

    prune_weights_list = []
    for i, w in enumerate(weights):
        weights_copy = w[:]
        weights_copy[abs(weights_copy) < weight_th] = 0
        prune_weights = weights_copy
        prune_weights_list.append(prune_weights)

    layer.set_weights(prune_weights_list)

model.save('./models_inception_00034_prune.hdf5')

# start = time()
# score = model.evaluate_generator(train_generator, train_generator.classes.size/32)
# cost = time() - start
# print('Train score (prune):', score[0])
# print('Train accuracy (prune):', score[1])
# print('Time (prune):', cost)

start = time()
score = model.evaluate_generator(val_generator, val_generator.classes.size/32)
cost = time() - start
print('Test score (prune):', score[0])
print('Test accuracy (prune):', score[1])
print('Time (prune):', cost)


# retrain
tb = TensorBoard(log_dir='./logs_inception_prune_retrain', histogram_freq=0, write_graph=True, write_images=False)
mc = ModelCheckpoint('./models_inception_prune_retrain/weights.{epoch:05d}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
model.fit_generator(train_generator, train_generator.classes.size/32, nb_epoch=10, callbacks=[tb, mc], validation_data=val_generator, validation_steps=val_generator.classes.size/32)

model.save('./models_inception_00034_prune_retrain10.hdf5')

# start = time()
# score = model.evaluate_generator(train_generator, train_generator.classes.size/32)
# cost = time() - start
# print('Train score (prune retrain):', score[0])
# print('Train accuracy (prune retrain):', score[1])
# print('Time (prune retrain):', cost)

start = time()
score = model.evaluate_generator(val_generator, val_generator.classes.size/32)
cost = time() - start
print('Test score (prune retrain):', score[0])
print('Test accuracy (prune retrain):', score[1])
print('Time (prune retrain):', cost)