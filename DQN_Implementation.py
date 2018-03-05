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

class QNetwork():

    def __init__(self, environment_name, model_type):

        env = gym.make(environment_name)
        env.reset()
        self.num_actions = env.action_space.n
        self.state_size = env.observation_space.shape[0]
        self.model = self.LinearDQN_initialize(model_type)
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
        model.summary()
        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=0.001))
        return model

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

class Replay_Memory():
    def __init__(self, env, batch_size, memory_size=50000, burn_in=10000):
        self.memory = deque(maxlen=memory_size)
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.num_iterations = 1000000
        self.env = env
        self.batch_size = batch_size

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
                action_prob = np.ones((self.env.action_space.n, 1), dtype='float')[:,0]
                action_prob = action_prob/np.sum(action_prob)
                set_actions = range(0, self.env.action_space.n)
                for given_iter in range(self.num_iterations):
                    action = np.random.choice(set_actions, size=1, replace=False, p=action_prob)[0]
                    action_prob[action] += 1
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
             
    def sample_batch(self, batch_size=32):
        mini_batch = random.sample(self.memory, batch_size)
        return mini_batch

    def append(self, transition):
        self.memory.append(transition)


class DQN_Agent():

    def __init__(self, environment_name, render=False):
        self.env_name = environment_name
        self.train_type = 'use_replay_memory'
        self.env = gym.make(environment_name)
        self.env.reset()
        if self.train_type == 'use_replay_memory':
            self.batch_size = 32
            self.replay_memory = self.burn_in_memory()
            self.eps = 0.75
            self.eps_decay_fact = 0.90
        if self.train_type == 'no_replay_memory':
            self.eps = 1
            self.eps_decay_fact = 0.99
            self.batch_size = 1
        self.model_type = 'dqn'
        if environment_name == 'CartPole-v0':
            self.gamma = float(0.99)
        if environment_name == 'MountainCar-v0':
            self.gamma = float(1)
        self.num_actions = self.env.action_space.n
        self.state_size = self.env.observation_space.shape[0]
        self.model = QNetwork(environment_name, self.model_type)
        self.num_iterations = 1000000
        self.num_episodes = 1000

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
                rewards_final = np.zeros((self.batch_size,1), dtype='float')
                self.model.train_batch(states, actions, rewards_final, next_states, dones, self.gamma)

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
    agent.test()


if __name__ == '__main__':
    main(sys.argv)
