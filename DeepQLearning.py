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

    def stack_frames(self,state,reset=False):
        """
        Alınan bir frame'nin işlenerek 4'lü frameStack'in oluşturulması için kullanılır.
        Reset=True durumunda tura yeni başlandığı için aynı frame 4 kere stack'e basılmaktadır.
        return edilen deque objesi stackFrame numpy array'e dönüştürülerek yapay sinir ağı beslenmektedir
        """
        state = process_frame(state)

        if reset:
            self.stack = deque([np.zeros(frame_shape, dtype=np.uint8) for i in range(4)], maxlen=4)
            for _ in range(4):
                self.stack.append(state)            
        else:
            self.stack.append(state)
        state_stack = np.stack(self.stack,axis=2)
        return state_stack
    def make_action(self,epsilon,state):
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(state))
        else:
            action = np.random.randint(0,self.num_actions)
        return action
            
    def update_memory(self, transition):
        self.replay_memory.append(transition)
    
    def train(self):
        global sess
        if len(self.replay_memory) < MIN_MEMORY_SIZE:
            return
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_states = np.array([transition[0] for transition in minibatch])/255
        try:
            with sess.as_default():
                with sess.graph.as_default():
                    #print("trainstarted")
                    current_qs_list = self.training_model.predict(current_states,PREDICTION_BATCH_SIZE)
        except Exception as ex:
             print('Train ilk')

        new_current_states = np.array([transition[3] for transition in minibatch])/255
        try:
            with sess.as_default():
                with sess.graph.as_default():
                    new_qs_list = self.target_model.predict(new_current_states,PREDICTION_BATCH_SIZE)
        except Exception as ex:
            print('Train iki')


        X = []
        y = []
        for index, (current_state,action,reward,_,done) in enumerate(minibatch):
            if not done:
                max_new_q = np.max(new_qs_list[index])
                new_q = reward + DISCOUNT*max_new_q
            else:

                new_q = reward
            
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)
        self.logq_values.write(self.episode,self.step,current_qs)
        log_it = False

        if self.tensorboard.step > self.last_logged_episode:
            log_it = True
            self.last_log_episode = self.tensorboard.step
        try:
            with sess.as_default():
                with sess.graph.as_default():
                    self.training_model.fit(np.array(X)/255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_it else None)
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

        if self.episode > self.last_logged_episode:
            log_it = True
            self.last_log_episode = self.episode


        if self.episode % UPDATE_TARGET_EVERY == 0:
            try:
                with sess.as_default():
                    with sess.graph.as_default():
                        return self.target_model.set_weights(self.training_model.get_weights())
            except Exception as exAalborg_transfer_learning:
                print("update counter")
    def get_qs(self, state):
        #print("get qs in")
        try:
            with sess.as_default():
                with sess.graph.as_default():
                    return self.training_model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]
        except Exception as ex:
            print("get qs")

    def train_in_loop(self):
        
        global sess
        
        backend.set_session(tf.Session(config=tf.ConfigProto()))
        X = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH,3)).astype(np.float32)
        y = np.random.uniform(size=(1, self.num_actions)).astype(np.float32)
        try:
            with sess.as_default():
                with sess.graph.as_default():
                    #set_session(sess)
                    self.training_model.fit(X,y, verbose=False, batch_size=1)
        except Exception as ex:
             print('train in loop')

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

if name == 'main':

    savedstepnumber = 162560
    num_of_episodes = None
    LOAD_MODEL = True
    model_name = "CG_Track2_no_transferlearning"
    if not os.path.isdir(model_name):
        os.makedirs(model_name)
    if num_of_episodes is None:
        num_of_episodes = float('inf')
    if LOAD_MODEL is True:
        with open(f"CG_Track2_no_transferlearning/info_step{savedstepnumber}.txt") as json_file:
            data = json.load(json_file)
            episode = data['episode']
            epsilon = data['epsilon']
            step = data['step']
    else:
        episode = 0
        epsilon = 1
        step = 0
    agent = DQNAgent(LOAD_MODEL,model_name)
    env = TorcsEnv(vision=True, throttle=True)
    random.seed(1)
    np.random.seed(1)
    ep_rewards = []
    tf.set_random_seed(1)
    MIN_EPSILON = 0.001
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)
    agent.get_qs(np.ones((IM_HEIGHT, IM_WIDTH,3)))
    while episode <= num_of_episodes:
        episode +=1
        agent.tensorboard.step = episode
        episode_reward = 0
        agent.episode = episode
        agent.step = step
        observation = env.reset()
        episode_start = time.time()
        current_state = screen_getter()
        done = False
        while not done:
            action = agent.make_action(epsilon,current_state)
            new_observation, reward, done, _ = env.step(action)
            new_state = screen_getter()
            episode_reward +=reward
            agent.update_memory((current_state,action,reward,new_state,done))
            step +=1
            current_state = new_state
        ep_rewards.append(episode_reward)
        reward_mean=np.mean(ep_rewards[-30:])
        msg = "{0:4}:{1}\t Epsilon: {2:4.2f}\t Reward: {3:.1f}\t Episode Mean: {4:.1f}"
        print(msg.format(episode,step,epsilon,episode_reward, reward_mean))
        agent.log_rewards.write(count_episodes=episode,count_states=step,reward_episode=episode_reward,reward_mean=reward_mean)
        
        if epsilon > MIN_EPSILON:
            epsilon *=EPSILON_DECAY
            epsilon = max(MIN_EPSILON,epsilon)


