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



env = gym.wrappers.Monitor(gym.make('SpaceInvaders-v0'), './video', video_callable=lambda episode_id: True, force=True)
model = load_model('/home/yash/Sem2/DeepRL/space/model_init_SpaceInvaders-v0.h5')
for i in range(10):
	state = env.reset()
	done = False
	total_reward = 0
	while not done:
    		env.render()
    		state = np.reshape(state, [1, 2])
    		action = np.argmax(model.predict(state)[0])
    		next_state, reward, done, _ = env.step(action)
    		total_reward += reward
    		state = next_state
	print(total_reward)
