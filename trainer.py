import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import csv
from matplotlib import pyplot as plt
import numpy as np
import cv2

import keras
from keras.layers import Dense, Convolution2D, UpSampling2D, MaxPooling2D, ZeroPadding2D, Flatten, Dropout, Reshape
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

from keras.layers import Dense , Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD , Adam
from keras.layers import Conv2D , BatchNormalization
from keras.layers import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

from sklearn.externals import joblib


reshape_size = (48,48)


def GetData(filename):
    X = []
    Y = []
    X_test = []
    Y_test = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)  # skip the headers
        for row in csv_reader:
            pixels = np.fromstring(row[1], dtype=int, sep=' ')
            pixels = pixels.reshape( reshape_size )
            emotion = int(row[0])
            if row[2] == "Training":    
                X.append(pixels)
                Y.append(emotion)
            else:
                X_test.append(pixels)
                Y_test.append(emotion)
                
    return [X,Y, X_test, Y_test]

def ChangeTo4D(data):
    ip_shape = data.shape
    ip_shape = (ip_shape[0],1,ip_shape[1],ip_shape[2])
    return data.reshape(ip_shape)

[train_X, train_Y, test_X, test_Y] = GetData('fer2013.csv')
train_X = np.array(train_X)
train_Y = np.array(train_Y)
test_X = np.array(test_X)
test_Y = np.array(test_Y)

train_X_4d = ChangeTo4D(train_X)
train_X_4d = train_X_4d/255
test_X_4d = ChangeTo4D(test_X)
test_X_4d = test_X_4d/255


from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        featurewise_center=True,  
        samplewise_center=False,  
        featurewise_std_normalization=True,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=20,  
        zoom_range = 0.0,  
        width_shift_range=0.2,  
        height_shift_range=0.2,  
        horizontal_flip=True, 
        vertical_flip=False)  

datagen.fit(train_X_4d)

model = Sequential()

model.add(Conv2D(64, (5, 5), activation='relu', padding="same", input_shape=(1,48,48)))
model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(64, (5, 5), activation='relu', padding="same"))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(128, (4, 4), activation='relu', padding="same"))

model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(7 , activation='softmax'))

print(model.summary())

train_Y_one_hot_encoding = keras.utils.to_categorical(train_Y);
test_Y_one_hot_encoding = keras.utils.to_categorical(test_Y);

batch_size = 128
epochs = 20
model.compile(loss='binary_crossentropy', optimizer='adam' , metrics=['accuracy'])

from keras.callbacks import ReduceLROnPlateau
lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, min_delta=0.0001, patience=1, verbose=1)

history = model.fit_generator(datagen.flow(train_X_4d, train_Y_one_hot_encoding, batch_size=batch_size),
                    steps_per_epoch= train_X_4d.shape[0] // batch_size,
                    callbacks=[lr_reduce],
                    epochs = epochs, verbose = 2)

test_eval = model.evaluate(test_X_4d, test_Y_one_hot_encoding, verbose =1);
print('test loss and acc is', test_eval[0], test_eval[1]);

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")