import os
os.environ['KERAS_BACKEND']='tensorflow'
import keras.backend as K
K.set_image_dim_ordering('tf')
from keras.applications.inception_v3 import InceptionV3

model = InceptionV3()