# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 00:12:39 2019

@author: cagri
"""
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, AveragePooling2D, Activation, Flatten
from keras import losses
from keras.optimizers import Adam
from keras.models import Model
from collections import deque
from keras.callbacks import TensorBoard
import sys
import glob
import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import math
import tensorflow as tf
from tqdm import tqdm
import keras.backend.tensorflow_backend as backend
from threading import Thread
from gym_torcs_new import TorcsEnv
from keras import backend as K
from keras.models import model_from_json
from tensorflow.python.keras.backend import set_session
import keras.backend.tensorflow_backend as backend
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gdk
from gi.repository import GdkPixbuf as Pixbuf
import json
from keras.utils.vis_utils import plot_model
from logfunctions import LogQValues,LogReward

IM_HEIGHT = 84
IM_WIDTH = 84
MEMORY_SIZE = 5000
MIN_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 16
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
DISCOUNT = 0.99
UPDATE_TARGET_EVERY =5
epsilon = 1
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001
AGGREGATE_STATS_EVERY = 10
SAVE_MODEL_EVERY = 50
MODEL_NAME = 'CG_Track2_no_transferlearning'
SHOW_EVERY=1000
MIN_REWARD = -1
PREDICTION_BATCH_SIZE = 1

frame_shape= (84,84)
input_shape =(84,84,4)

def process_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (frame_shape), interpolation=cv2.INTER_AREA)
    return frame

config = tf.ConfigProto(device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

def array_from_pixbuf(p):
    " convert from GdkPixbuf to numpy array"
    w,h,c,r=(p.get_width(), p.get_height(), p.get_n_channels(), p.get_rowstride())
    assert p.get_colorspace() == Pixbuf.Colorspace.RGB
    assert p.get_bits_per_sample() == 8
    if  p.get_has_alpha():
        assert c == 4
    else:
        assert c == 3
    assert r >= w * c
    a=np.frombuffer(p.get_pixels(),dtype=np.uint8)
    if a.shape[0] == wch:
        return a.reshape( (h, w, c) )
    else:
        b=np.zeros((h,wc),'uint8')
        for j in range(h):
            b[j,:]=a[rj:rj+wc]
        return b.reshape( (h, w, c) )
def  screen_getter():
    screen = Gdk.get_default_root_window().get_screen()
    w = screen.get_active_window()
    x , y , u , d = w.get_geometry()
    pb = Gdk.pixbuf_get_from_window(w, x , y , u , d)
    state = array_from_pixbuf(pb)
    state = cv2.resize(state,(IM_WIDTH,IM_HEIGHT))
    return state
class DQNAgent:
    def __init__(self,load_model,model_name):
        self.num_actions= 12
        self.training_model = self.create_model(load_model)
        self.target_model = self.create_model(load_model)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}")
        self.target_model.set_weights(self.training_model.get_weights())
        self.replay_memory = deque(maxlen=MEMORY_SIZE)
        self.update_counter = 0
        self.terminate = False
        self.episode=0
        self.step = 0
        self.last_logged_episode = 0
        self.training_initialized = False
        self.logq_values = LogQValues(model_name)                 #Q değerlerinin saklanması için kullanılacak
        self.log_rewards = LogReward(model_name)
        
    def create_model(self,load_model):
        if load_model == True:
            json_file = open("CG_Track2_no_transferlearning/step162560.json", 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights("CG_Track2_no_transferlearning/step162560.h5")
            loaded_model.compile(loss="mse", optimizer=Adam(lr=1e-4), metrics=["accuracy"])
            print(loaded_model.summary)
            return loaded_model
        else:
            model = Sequential()
            model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(IM_WIDTH,IM_HEIGHT,3)))  #80*80*4
            model.add(Activation('relu'))
            
            model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
            model.add(Activation('relu'))
            
            model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
            model.add(Activation('relu'))
            
            model.add(Flatten())
            model.add(Dense(512))
            model.add(Activation('relu'))
            model.add(Dense(self.num_actions, activation="linear"))

            adam = Adam(lr=1e-4)
            model.compile(loss='mse',optimizer=adam)
 
 
            
            print("creating new model")
            return model


