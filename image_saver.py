import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from tensorflow.keras import layers
import keras
import time
import tensorflow as tf
import h5py
import skimage
import sys
import random
import pickle
from IPython import display
from keras.utils.vis_utils import plot_model

def save_images(model,number,directory,m_or_n='swimseg'):
    if m_or_n == 'swimseg':
        test_input = tf.random.normal([number, 128])
        test_labels = np.random.randint(1,2,size=number)
        one_hot_test_labels = keras.utils.to_categorical(test_labels,num_classes=2)
        predictions = model([test_input,one_hot_test_labels], training=False)
    elif m_or_n == 'swinseg':
        test_input = tf.random.normal([number, 128])
        test_labels = np.random.randint(0,1,size=number)
        one_hot_test_labels = keras.utils.to_categorical(test_labels,num_classes=2)
        predictions = model([test_input,one_hot_test_labels], training=False)
    else:
        print('please use either swimseg or swinseg')
    predictions = predictions.numpy().astype(int)
    for i in range(number):
        print(i)
        rgb = predictions[i,:,:,:3]
        gt_map = predictions[i,:,:,3]
        gt_map = 255*np.ceil(gt_map/255)
        out_dim = np.shape(gt_map)[0]
        GT_map = np.zeros((out_dim,out_dim,3))
        GT_map[:,:,0] = gt_map
        GT_map[:,:,1] = gt_map
        GT_map[:,:,2] = gt_map
        GT_map = GT_map.astype(np.uint8)
        
        #create directories if they dont exist
        if not os.path.exists(directory+'images/'):
            os.makedirs(directory+'images/')
            
        if not os.path.exists(directory+'GTmaps/'):
            os.makedirs(directory+'GTmaps/')
        
        #save images
        im = Image.fromarray(rgb.astype(np.uint8))
        im.save(directory+'images/'+str(i)+'.png')
        
        #save maps
        im = Image.fromarray(GT_map)
        im.save(directory+'GTmaps/'+str(i)+'.png')
        
SWIMNGEN = tf.keras.models.load_model('/home2/vcqf59/Swimseg_2/SWIMSEG_PLUS/final_model/') 
save_images(SWIMNGEN,100,'/home2/vcqf59/Swimseg_2/SWIMSEG_PLUS/Swimseg_aug/')
save_images(SWIMNGEN,100,'/home2/vcqf59/Swimseg_2/SWIMSEG_PLUS/Swinseg_aug/',m_or_n='swinseg')