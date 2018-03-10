from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import *
from keras.layers.pooling import GlobalAveragePooling1D
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import keras
import numpy as np
import gym
import sys
import copy
import argparse
from collections import deque
import random
import cv2


def get_action_image(model, state):
        '''
	    Given a state in form of image and model number (two models are used for SpaceInvaders), return the optimal action as per model.           Parameters:
	        state (type numpy matrix): contains 4 consecutive video frames.
		model_num (type int): model number from which action has to be taken (two models for SpaceInvaders).
	'''
        state_inp = np.zeros((1, 84, 84, 4), dtype='float')
        state_inp[0,:,:,:] = state[:,:,:]
        pred_action = model.predict(state_inp)
        return np.argmax(pred_action[0])

env = gym.wrappers.Monitor(gym.make('SpaceInvaders-v0'), './video', video_callable=lambda episode_id: True, force=True)
model = load_model('model_init_SpaceInvaders-v0.h5')
avg_reward = 0
test_iters = 10
for given_episode in range(0, test_iters):
            given_reward = 0
            state = env.reset()
            state = cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY), (84, 84))
            current_states_queue = deque(maxlen=4)
            next_states_queue = deque(maxlen=4)
            for frame_counter in range(4):
                current_states_queue.append(state)
                action = np.random.randint(0, env.action_space.n, 1)[0]
                next_state, reward, done, _ = env.step(action)
                next_state = cv2.resize(cv2.cvtColor(next_state, cv2.COLOR_RGB2GRAY), (84, 84))
                state = next_state
            next_states_queue = current_states_queue
            while True:
                action = get_action_image(model, np.stack(current_states_queue, axis=2))
                next_state, reward, done, _ = env.step(action)
                next_state = cv2.resize(cv2.cvtColor(next_state, cv2.COLOR_RGB2GRAY), (84, 84))
                next_states_queue.append(next_state)
                current_states = np.stack(current_states_queue, axis=2)
                next_states = np.stack(next_states_queue, axis=2)
                next_states_queue.append(next_state)
                current_states = np.stack(current_states_queue, axis=2)
                given_reward += reward
                current_states_queue = next_states_queue
                if done: break
            avg_reward += given_reward
print(float(avg_reward)/float(test_iters))

