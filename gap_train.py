import os
os.environ['KERAS_BACKEND']='tensorflow'
import keras.backend as K
K.set_image_dim_ordering('tf')
import h5py
import numpy as np
from sklearn.utils import shuffle
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.utils.io_utils import HDF5Matrix
from keras.layers import Input, Dropout, Dense
from keras.models import Model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

np.random.seed(2017)

# X_train = []
# X_test = []
nb_classes = 365


# X_inception_train = HDF5Matrix('gap_InceptionV3_365_train_prune_retrain.h5', 'data')
# X_ResNet50_train = HDF5Matrix('gap_ResNet50_365_train_prune_retrain.h5', 'data')
# y_train = HDF5Matrix('gap_InceptionV3_365_train_prune_retrain.h5', 'label', end=X_inception_train.shape[0])
#
# X_train = np.concatenate((X_inception_train, X_ResNet50_train), axis=1)
# print X_train.shape, y_train.shape
#
# # X_train, y_train = shuffle(X_train, y_train)
# y_train = np_utils.to_categorical(y_train, nb_classes)

X_inception_val = HDF5Matrix('gap_InceptionV3_365_val_prune_retrain.h5', 'data')
X_ResNet50_val = HDF5Matrix('gap_ResNet50_365_val_prune_retrain.h5', 'data')
y_val = HDF5Matrix('gap_InceptionV3_365_val_prune_retrain.h5', 'label', end=X_inception_val.shape[0])

X_val = np.concatenate((X_inception_val, X_ResNet50_val), axis=1)
print X_val.shape, y_val.shape

# X_val, y_val = shuffle(X_val, y_val)
y_val = np_utils.to_categorical(y_val, nb_classes)

input_tensor = Input(X_val.shape[1:])
x = input_tensor
x = Dropout(0.5)(x)
x = Dense(nb_classes, activation='softmax')(x)
model = Model(input_tensor, x)

# model.load_weights('./models1/weights.01199-1.10.hdf5')
sgd = SGD(lr=0.001, momentum=0.9, decay=1e-4, nesterov=False)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
mc = ModelCheckpoint('./models/weights.{epoch:05d}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=10)
model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=512, nb_epoch=300, callbacks=[tb, mc])

model.save('model.h5')

re = model.evaluate(X_val, y_val)
print re