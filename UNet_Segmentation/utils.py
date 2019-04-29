from glob import glob
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
import cv2
import os
import random

image_list = []
mask_list = []

def next_batch(num, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def get_data(directory):
    for img_file in glob(directory+"/Original/*.tif"):
        img = cv2.imread(img_file)
        img = cv2.resize(img, (256, 256))
        image_list.append(img)
    for mask_file in glob(directory+"/Ground Truth/*.tif"):
        mask = cv2.imread(mask_file, 0)
        mask = cv2.resize(mask, (256, 256))
        mask_list.append(mask)
    assert len(mask_list) == len(image_list)
    np.save('images.npy', image_list)
    np.save('masks.npy', mask_list)

def train_test_split_data(directory):
    if os.path.isfile('images.npy') and os.path.isfile('masks.npy'):
        print('DATA EXISTS!')
        images = np.load('images.npy')
        masks = np.load('masks.npy')
        x_train, x_val, y_train, y_val = train_test_split(images, masks, test_size=0.15, random_state=42)
        x_train = np.asarray(x_train)
        x_val = np.asarray(x_val)
        y_val = np.asarray(y_val)
        y_train = np.asarray(y_train)
        y_train = np.reshape(y_train, (-1, 256, 256, 1))
        y_val = np.reshape(y_val, (-1, 256, 256, 1))
        print("Done Data Formation.")
        print('X_TRAIN, Y_TRAIN Shape', x_train.shape, y_train.shape)
        print('X_VAL, Y_VAL Shape', x_val.shape, y_val.shape)
        return x_train, x_val, y_train, y_val
    else:
        get_data(directory)
        print('Call TRAIN_TEST_SPLIT_DATA AGAIN!')
        train_test_split_data(directory)