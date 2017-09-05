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
from scipy.misc import imread, imresize
from random import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten


def img_gen(filename, path, batch_size=32, image_size=(256,256)):
    f = open(filename, mode='r')
    img_list = []
    label_list = []
    all_lines = f.readlines()
    print filename, len(all_lines)
    shuffle(all_lines)

    for i, line in enumerate(all_lines):
        content = line.strip().split(' ')
        if content[0][0] == '/':
            img_name = path+content[0]
        else:
            img_name = os.path.join(path, content[0])
        img = imread(img_name, mode='RGB')
        img = imresize(img, image_size)
        label = np.zeros((365,))
        label[int(content[1])] = 1
        img_list.append(img)
        label_list.append(label)

        # if len(img_list) == batch_size:
        #     yield np.array(img_list), np.array(label_list)
        #     print i
        #     img_list = []
        #     label_list = []
    f.close()
    if len(img_list) > 0:
        return (np.array(img_list), np.array(label_list))
        # yield np.array(img_list), np.array(label_list)

# train_file = '/media/wac/backup/places365_Standard/filelist_places365-standard/places365_train_standard_filter.txt'
# val_file = '/media/wac/backup/places365_Standard/filelist_places365-standard/places365_val.txt'
train_img_path = '/home/wac/places365_Standard/places365_data'
val_img_path = '/home/wac/places365_Standard/val_data'
categories = '/home/wac/places365_Standard/filelist_places365-standard/categories_places365_pure.txt'

f = open(categories, mode='r')
classes = [line.strip().split(' ')[0] for line in f.readlines()]

nb_classes = 365
# image_size = (299, 299)
image_size = (224, 224)

# train_generator = img_gen(train_file, train_img_path, image_size=image_size)
# val_generator = img_gen(val_file, val_img_path, image_size=image_size)

# image_size = (224, 224)

# basemodel = InceptionV3()
basemodel = ResNet50()
input = basemodel.input
# x = basemodel.get_layer('avg_pool').output
x = basemodel.get_layer('flatten_1').output
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
model.load_weights('./models/weights.00019.hdf5')
tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
mc = ModelCheckpoint('./models/weights.{epoch:05d}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
model.fit_generator(train_generator, train_generator.classes.size/32, nb_epoch=100, callbacks=[tb, mc], validation_data=val_generator, validation_steps=val_generator.classes.size/32, initial_epoch=20)

# epoch = 400
# for e in range(epoch):
#     for [imgs, labels] in img_gen(val_file, val_img_path, image_size=image_size):
#         model.fit(imgs, labels, batch_size=32, nb_epoch=1, shuffle=True, callbacks=[tb, mc])

model.save('ResNet50_model.h5')
