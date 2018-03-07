# !/usr/bin/env python
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

class QNetwork():

    def __init__(self, environment_name, model_type):

        env = gym.make(environment_name)
        env.reset()
        self.num_actions = env.action_space.n
        self.state_size = env.observation_space.shape[0]
        if environment_name == 'SpaceInvaders-v0':
          self.model_1 = self.LinearDQN_initialize(model_type)
          self.model_2 = self.LinearDQN_initialize(model_type)
        del env

    def save_model_weights(self, suffix):
        self.model.save(suffix)

    def load_model(self, model_file):
        self.model = load_model(model_file)

    def load_model_weights(self, weight_file):
        pass

    def LinearDQN_initialize(self, model_type):
        if model_type == 'linear_dqn':
            model = Sequential()
            model.add(Dense(self.num_actions, input_dim=self.state_size))

        if model_type == 'dqn':
            model = Sequential()
            model.add(Dense(16, input_dim=self.state_size, activation='relu'))
            model.add(Dense(16, input_dim=self.state_size, activation='relu'))
            model.add(Dense(self.num_actions, activation='linear'))

        if model_type == 'ddqn':
            input_layer = Input(shape=(self.state_size,))
            x = Dense(16, activation='relu')(input_layer)
            state_val = Dense(1, activation='linear')(x)
            y = Dense(16, activation='relu')(input_layer)
            y = Dense(16, activation='relu')(y)
            adv_vals = Dense(self.num_actions, activation='linear')(y)
            policy = keras.layers.merge([adv_vals, state_val], mode=lambda x: x[0]-K.mean(x[0])+x[1], output_shape = (self.num_actions,))
            model = Model(input=[input_layer], output=[policy])

        if model_type == 'dqn_space_invaders':
            inp_layer = Input(shape=(84, 84, 4))
            x = Conv2D(filters=16, kernel_size=(8, 8), strides=4, activation='relu')(inp_layer)
            x = Conv2D(filters=32, kernel_size=(4, 4), strides=2, activation='relu')(x)
            x = Flatten()(x)
            x = Dense(256, activation='relu')(x)
            x = Dense(self.num_actions, activation='linear')(x)
            model = Model(inp_layer, x)
        model.summary()
        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=0.001))
        return model

    def get_action_image(self, state, model_num):
        state_inp = np.zeros((1, 84, 84, 4), dtype='float')
        state_inp[0,:,:,:] = state[:,:,:]
        if model_num == 1:
            pred_action = self.model_1.predict(state_inp)
        else:
            pred_action = self.model_2.predict(state_inp)
        return np.argmax(pred_action[0])

    def get_action_prob(self, state, model_num):
        state_inp = np.zeros((1, 84, 84, 4), dtype='float')
        state_inp[0,:,:,:] = state[:,:,:]
        if model_num == 1:
            pred_action = self.model_1.predict(state_inp)[0,:]
        else:
            pred_action = self.model_2.predict(state_inp)[0,:]
        return pred_action

    def get_action(self, state):
        pred_action = self.model.predict(state)
        return np.argmax(pred_action[0])

    def train(self, state, action, reward, next_state, done, gamma):
        target = reward
        if not done:
            target = (reward + gamma * np.amax(self.model.predict(next_state)[0]))
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)

    def train_batch(self, states, actions, rewards, next_states, dones, gamma):
        batch_size = states.shape[0]
        targets = rewards
        targets_f = np.zeros((batch_size, self.num_actions), dtype='float')
        for i in range(0, batch_size):
            if not dones[i]:
                targets[i] = float(rewards[i] + gamma*np.amax(self.model.predict(np.reshape(next_states[i], [1,self.state_size]))[0]))
            targets_f[i][:] = self.model.predict(np.reshape(states[i], [1,self.state_size]))[0]
            targets_f[i][int(actions[i])] = targets[i]
        self.model.fit(states, targets_f, epochs=1, verbose=1)

    def train_batch_space_invaders(self, states, actions, rewards, next_states, dones, gamma, model_num):
        batch_size = states.shape[0]
        targets = rewards
        targets_f = np.zeros((batch_size, self.num_actions), dtype='float')
        if model_num == 1: model_target_num = 2
        elif model_num == 2: model_target_num = 1
        for i in range(0, batch_size):
            if not dones[i]:
                targets[i] = float(rewards[i] + gamma*np.amax(self.get_action_prob(next_states[i], model_target_num)))
            targets_f[i][:] = self.get_action_prob(states[i], model_target_num)
            targets_f[i][int(actions[i])] = targets[i]
        if model_num == 1:
            self.model_1.fit(states, targets_f, epochs=1, verbose=1)
        elif model_num == 2:
            self.model_2.fit(states, targets_f, epochs=1, verbose=1)
        else:
            print('UNKNOWN MODEL NUMBER !')
            sys.exit(1)

class Replay_Memory():
    def __init__(self, env, batch_size, memory_size=32, burn_in=10000):
        self.memory = deque(maxlen=memory_size)
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.num_iterations = 1000000
        self.env = env
        self.batch_size = batch_size
        self.num_frames = 4

    def fill_random_transitions(self, env_name):
        if env_name == 'CartPole-v0':
            random_transition_counter = 0
            num_actions = self.env.action_space.n
            state_size = self.env.observation_space.shape[0]
            for e in range(self.burn_in):
                state = self.env.reset()
                state = np.reshape(state, [1,state_size])
                for given_iter in range(self.num_iterations):
                    action = np.random.randint(0, num_actions, 1)[0]
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = np.reshape(next_state, [1, state_size])
                    self.append([state, action, reward, next_state, done])
                    state = next_state
                    random_transition_counter += 1
                    if done:
                        break
                    if random_transition_counter > self.memory_size:
                        return

        elif env_name == 'MountainCar-v0':
            random_transition_counter = 0
            num_actions = self.env.action_space.n
            state_size = self.env.observation_space.shape[0]
            for e in range(self.burn_in):
                state = self.env.reset()
                state = np.reshape(state, [1, state_size])
                action_prob = np.array([1,0,1], dtype='float')
                action_prob = action_prob/np.sum(action_prob)
                set_actions = range(0, self.env.action_space.n)
                mod_fact = 40
                for given_iter in range(self.num_iterations):
                    action = np.random.choice(set_actions, size=1, replace=False, p=action_prob)[0]
                    if action == 0 and given_iter%mod_fact == 0:
                         action_prob[2] = 100
                         action_prob[0] = 1
                    if action == 2 and given_iter%mod_fact == 0:
                         action_prob[2] = 1
                         action_prob[0] = 100
                    if action == 0 and given_iter%mod_fact != 0:
                         action_prob[0] *= 2
                    if action == 2 and given_iter%mod_fact != 0:
                         action_prob[2] *= 2
                    action_prob = action_prob / np.sum(action_prob)
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = np.reshape(next_state, [1, state_size])
                    self.append([state, action, reward, next_state, done])
                    state = next_state
                    random_transition_counter += 1
                    if done:
                        break
                    if random_transition_counter > self.memory_size:
                        return

        elif env_name == 'SpaceInvaders-v0':
            random_transition_counter = 0
            num_actions = self.env.action_space.n
            for e in range(self.burn_in):
                state = self.env.reset()
                state = cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY), (84, 84))
                current_states_queue = deque(maxlen=4)
                next_states_queue = deque(maxlen=4)
                for frame_counter in range(self.num_frames):
                    current_states_queue.append(state)
                    action = np.random.randint(0, num_actions, 1)[0]
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = cv2.resize(cv2.cvtColor(next_state, cv2.COLOR_RGB2GRAY), (84, 84))
                    state = next_state
                next_states_queue = current_states_queue
                for given_iter in range(self.num_iterations):
                    action = np.random.randint(0, num_actions, 1)[0]
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = cv2.resize(cv2.cvtColor(next_state, cv2.COLOR_RGB2GRAY), (84, 84))
                    next_states_queue.append(next_state)
                    current_states = np.stack(current_states_queue, axis=2)
                    next_states = np.stack(next_states_queue, axis=2)
                    self.append([current_states, action, reward, next_states, done])
                    random_transition_counter += 1
                    current_states_queue = next_states_queue
                    if done:
                        break
                    if random_transition_counter > self.memory_size:
                        return

    def sample_batch(self, batch_size=32):
        mini_batch = random.sample(self.memory, batch_size)
        return mini_batch

    def append(self, transition):
        self.memory.append(transition)

class DQN_Agent():

    def __init__(self, environment_name, render=False):
        self.env_name = environment_name
        self.train_type = 'use_replay_memory_space_invaders_v0'
        self.env = gym.make(environment_name)
        self.env.reset()
        if self.train_type == 'use_replay_memory':
            self.batch_size = 32
            self.replay_memory = self.burn_in_memory()
            self.eps = 1
            self.eps_decay_fact = 0.99
        if self.train_type == 'no_replay_memory':
            self.eps = 1
            self.eps_decay_fact = 0.99
            self.batch_size = 1
        if self.train_type == 'use_replay_memory_space_invaders_v0':
            self.batch_size = 32
            self.replay_memory = self.burn_in_memory()
            self.eps = 1
            self.eps_decay_fact = 0.99
        self.model_type = 'dqn'
        if environment_name == 'CartPole-v0':
            self.gamma = float(0.99)
        if environment_name == 'MountainCar-v0':
            self.gamma = float(1)
        if environment_name == 'SpaceInvaders-v0':
            self.gamma = float(1)
            self.model_type = 'dqn_space_invaders'
            self.train_type = 'use_replay_memory_space_invaders_v0'
        self.num_actions = self.env.action_space.n
        self.state_size = self.env.observation_space.shape[0]
        self.model = QNetwork(environment_name, self.model_type)
        self.num_iterations = 1000000
        self.num_episodes = 2000
        self.num_frames = 4

    def epsilon_greedy_policy(self, q_values):
        pass

    def greedy_policy(self, q_values):
        pass

    def train(self):

        if self.train_type == 'no_replay_memory':
            for given_episode in range(0, self.num_episodes):
                print('Train episode: ' + str(given_episode))
                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])
                rand_thresh = 1
                for given_iter in range(0, self.num_iterations):
                    rand_thresh *= self.eps_decay_fact
                    rand_thresh = max(rand_thresh, 0.1)
                    rand_num = np.random.uniform(low=0, high=1)
                    if rand_num < rand_thresh:
                        action = np.random.randint(0, self.num_actions, 1)[0]
                    else:
                        action = self.model.get_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = np.reshape(next_state, [1, self.state_size])
                    self.model.train(state, action, reward, next_state, done, self.gamma)
                    state = next_state
                    if done:
                        break

        if self.train_type == 'use_replay_memory':
            for given_episode in range(0, self.num_episodes):
                print('Train episode: ' + str(given_episode))
                state = self.env.reset()
                state = np.reshape(state, [1 , self.state_size])
                rand_thresh = 1
                for given_iter in range(0, self.num_iterations):
                    rand_thresh *= self.eps_decay_fact
                    rand_thresh = max(rand_thresh, 0.1)
                    rand_num = np.random.uniform(low=0, high=1)
                    if rand_num < rand_thresh:
                        action = np.random.randint(0, self.num_actions, 1)[0]
                    else: 
                        action = self.model.get_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = np.reshape(next_state, [1, self.state_size])
                    self.replay_memory.append([state, action, reward, next_state, done])
                    state = next_state
                    if done: break
                given_batch = self.replay_memory.sample_batch(self.batch_size)
                states = np.array([i[0][0] for i in given_batch], dtype='float')
                actions = np.array([i[1] for i in given_batch],dtype='int')
                rewards = np.array([i[2] for i in given_batch], dtype='float')
                next_states = np.array([i[3][0] for i in given_batch], dtype='float')
                dones = np.array([i[4] for i in given_batch], dtype='bool')
                self.model.train_batch(states, actions, rewards, next_states, dones, self.gamma)

        if self.train_type == 'use_replay_memory_space_invaders_v0':
            given_batch = self.replay_memory.sample_batch(self.batch_size)
            for given_episode in range(0, self.num_episodes):
                print('Train episode: ' + str(given_episode))
                state = self.env.reset()
                state = cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY), (84, 84))
                current_states_queue = deque(maxlen=4)
                next_states_queue = deque(maxlen=4)
                for frame_counter in range(self.num_frames):
                    current_states_queue.append(state)
                    action = np.random.randint(0, self.num_actions, 1)[0]
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = cv2.resize(cv2.cvtColor(next_state, cv2.COLOR_RGB2GRAY), (84, 84))
                    state = next_state
                next_states_queue = current_states_queue
                rand_thresh = 1
                for given_iter in range(0, self.num_iterations):
                    rand_thresh *= self.eps_decay_fact
                    rand_thresh = max(rand_thresh, 0.1)
                    rand_num = np.random.uniform(low=0, high=1)
                    if rand_num < rand_thresh:
                        action = np.random.randint(0, self.num_actions, 1)[0]
                    else:
                        if np.random.randint(low=0, high=10, size=1)[0]%2 == 0:
                            action = self.model.get_action_image(np.stack(current_states_queue, axis=2), model_num = 1)
                        else:
                            action = self.model.get_action_image(np.stack(current_states_queue, axis=2), model_num = 2)
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = cv2.resize(cv2.cvtColor(next_state, cv2.COLOR_RGB2GRAY), (84, 84))
                    next_states_queue.append(next_state)
                    current_states = np.stack(current_states_queue, axis=2)
                    next_states = np.stack(next_states_queue, axis=2)
                    self.replay_memory.append([current_states, action, reward, next_states, done])
                    current_states_queue = next_states_queue
                    if done: break
                given_batch = self.replay_memory.sample_batch(self.batch_size)
                states = np.array([i[0] for i in given_batch], dtype='float')
                actions = np.array([i[1] for i in given_batch],dtype='int')
                rewards = np.array([i[2] for i in given_batch], dtype='float')
                next_states = np.array([i[3] for i in given_batch], dtype='float')
                dones = np.array([i[4] for i in given_batch], dtype='bool')
                if int(given_episode/100)%2 == 0:
                    self.model.train_batch_space_invaders(states, actions, rewards, next_states, dones, self.gamma, model_num = 1)
                else:
                    self.model.train_batch_space_invaders(states, actions, rewards, next_states, dones, self.gamma, model_num = 2)
            self.model.save_model_weights('model_space.h5')
    
    def test(self, model_file=None):
        avg_reward = 0
        for given_episode in range(0, 200):
            state = self.env.reset()
            done = False
            total_reward = 0
            while done is False:
                state = np.reshape(state, [1, self.state_size])
                action = self.model.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                state = next_state
            avg_reward += total_reward
        print(float(avg_reward) / float(200))


    def burn_in_memory(self):
        memory = Replay_Memory(self.env, self.batch_size)
        memory.fill_random_transitions(self.env_name)
        return memory


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env', dest='env', type=str)
    parser.add_argument('--render', dest='render', type=int, default=0)
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--model', dest='model_file', type=str)
    return parser.parse_args()


def main(args):
    args = parse_arguments()
    environment_name = args.env
    agent = DQN_Agent(environment_name)
    agent.train()


if __name__ == '__main__':
    main(sys.argv)
