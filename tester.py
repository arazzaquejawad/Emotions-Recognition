import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import csv
from matplotlib import pyplot as plt
import numpy as np
import cv2
import sys
from keras.models import model_from_json

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

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def ChangeTo4D(data):
    ip_shape = data.shape
    ip_shape = (ip_shape[0],1,ip_shape[1],ip_shape[2])
    return data.reshape(ip_shape)   



video_capture = cv2.VideoCapture(0)

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
    


while True:
    images = []
    
    
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3, 5)
    for (x,y,w,h) in faces:
        gray = gray[y:y+h, x:x+w]
    gray = cv2.resize(gray,(48,48))
    
    images.append(gray)
    images = np.array(images)
    img_4d = ChangeTo4D(images)
     
    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    prediction = np.argmax(loaded_model.predict(img_4d))
    
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
    cv2.putText(frame, str(prediction), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), lineType=cv2.LINE_AA)     
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()


