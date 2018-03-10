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
    '''
        This class contains neural network to approximate the Q-function.
	Following are the inputs requited to initialize this class:
	    (1). Environment name: this is required because the SpaceInvaders environment uses two models.
	         (a). First model is the last checkpoint of training model and is used to generate target labels.
		 (b). Second model is the main model which is being trained.
            (2). Model type: this parameter takes care of different architecture requirements in the assignments. This variale can be:
	         (a) "linear_dqn": Single layer linear network with no activations.
		 (b) "dqn": Multi layer MLP with activations.
		 (c) "ddqn": Dueling arhcitecture consisting of two branches (state value function, advantage functions).
	Following are the functionalies included in this class:
	    (1). Model Class Initialization.
	    (2). Checkpoint Swap: This is saving the current model as latest checkpoint (used for SpaceInvaders).
	    (3). Save Model Weights.
	    (4). Load model from a file.
	    (5). Model architecture initialization.
	    (6). Get the action as per trained model if an image is given as input. (Used for SpaceInvaders)
	    (7). Get the Q(s,a) estimated by model given state as input. (output of network).
	    (8). Get action as per state parameters. (used for CartPole and MountainCar).
	    (9). Train the model stochastically if state parameters are given as input (used for CartPole, MountainCar when NO memory replay is used).
	    (10). Train the model at mini-batch level if state parameters are given as input (used for CartPole when memory replay is used)
	    (11). Train the model at mini-batch level if state parameters are given as input for MountainCar (as reaching 0.5 is the goal, a saperate function is required for this).
	    (12). Train the model at mini-batch level if images are given as input (used for SpaceInvaders).
    '''
    def __init__(self, environment_name, model_type):
        '''
	    Initialize the class as per input parameters.
            Parameters:
	        environment_name (type string): Name of environment to be used. Possible values: 'CartPole-v0', 'MountainCar-v0', 'SpaceInvaders-v0'
	        model_type (type string): Name of model type to be used. Possible valies: 'linear_dqn', 'dqn', 'ddqn', 'dqn_space_invaders'
	'''
        env = gym.make(environment_name)
        env.reset()
        self.num_actions = env.action_space.n
        self.state_size = env.observation_space.shape[0]
        if environment_name == 'SpaceInvaders-v0':
          self.model_1 = self.LinearDQN_initialize(model_type)
          self.save_model_weights(self.model_1, 'model_init_'+environment_name+'.h5')
          self.model_2 = load_model('model_init_'+environment_name+'.h5')
        else:
          self.model = self.LinearDQN_initialize(model_type)
        del env

    def checkpoint_swap(self, environment_name):
        '''
	    Given an environment name save the recent model as checkpoint.
	    Parameters: 
	        environment_name (type string): Name of the environment.
	'''
        print('Reloading model....')
        self.model_1 = load_model('model_init_'+environment_name+'.h5')
        
    def save_model_weights(self, model, suffix):
        '''
	    Given a model and string with model name, saves the model weights and defination.
	    Parameters: 
	        model (type keras model): Model to be saved.
		suffix (type string): Output path to which model is to be saved.
	'''
        print('Saving model....')
        model.save(suffix)

    def load_model(self, model_file):
        '''
	    Given a model file name, loads the model and returns it.
	    Parameters: 
	        model_file (type string): Path of model file to be loaded.
	'''
        print('Loading model....')
        model = load_model(model_file)
        return model

    def load_model_weights(self, weight_file):
        '''
	    This function is not used anywhere because both model defination and weights are saved as a .h5 file by keras.
	'''
        pass

    def LinearDQN_initialize(self, model_type):
        '''
	    Given the type of model. This function initializes the model architecture.
	    Parameters:
	        model_type (type string): Specifies what model to initialize.
	'''
        if model_type == 'linear_dqn':
            model = Sequential()
            model.add(Dense(self.num_actions, input_dim=self.state_size))

        if model_type == 'dqn':
            model = Sequential()
            model.add(Dense(32, input_dim=self.state_size, activation='relu'))
            model.add(Dense(32, input_dim=self.state_size, activation='relu'))
            model.add(Dense(self.num_actions, activation='linear'))

        if model_type == 'ddqn':
            input_layer = Input(shape=(self.state_size,))
            x = Dense(32, activation='relu')(input_layer)
            state_val = Dense(1, activation='linear')(x)
            y = Dense(32, activation='relu')(input_layer)
            y = Dense(32, activation='relu')(y)
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
        '''
	    Given a state in form of image and model number (two models are used for SpaceInvaders), return the optimal action as per model.           Parameters:
	        state (type numpy matrix): contains 4 consecutive video frames.
		model_num (type int): model number from which action has to be taken (two models for SpaceInvaders).
	'''
        state_inp = np.zeros((1, 84, 84, 4), dtype='float')
        state_inp[0,:,:,:] = state[:,:,:]
        if model_num == 1:
            pred_action = self.model_1.predict(state_inp)
        else:
            pred_action = self.model_2.predict(state_inp)
        return np.argmax(pred_action[0])

    def get_action_prob(self, state, model_num):
        '''
	    Given a state in form of image and model number (two models are used for SpaceInvaders), return the value function for each action .       Parameters:
	        state (type numpy matrix): contains 4 consecutive video frames.
		model_num (type int): model number from which action probability has to be generated (two models for SpaceInvaders).
	'''
        state_inp = np.zeros((1, 84, 84, 4), dtype='float')
        state_inp[0,:,:,:] = state[:,:,:]
        if model_num == 1:
            pred_action = self.model_1.predict(state_inp)[0,:]
        else:
            pred_action = self.model_2.predict(state_inp)[0,:]
        return pred_action

    def get_action(self, state):
        '''
	    Given a state in form of state varialbes, returns the optimal action as per current model.
	    Parameters:
	        state (type numpy array): contains array of size as number of environment state variables.
	'''
        pred_action = self.model.predict(state)
        return np.argmax(pred_action[0])

    def train(self, state, action, reward, next_state, done, gamma):
        '''
	    Given a state, action, reward, next_state and done flag train the model stochastically for given input.
	'''
        target = reward
        if not done:
            target = (reward + gamma * np.amax(self.model.predict(next_state)[0]))
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)

    def train_batch(self, states, actions, rewards, next_states, dones, gamma):
        '''
	    Given sampled states, actions, rewards, next_states and done flags train the model at mini-batch level for given input.
	'''
        batch_size = states.shape[0]
        targets = rewards
        targets_f = np.zeros((batch_size, self.num_actions), dtype='float')
        for i in range(0, batch_size):
            if not dones[i]:
                targets[i] = float(rewards[i] + gamma*np.amax(self.model.predict(np.reshape(next_states[i], [1,self.state_size]))[0]))
            targets_f[i][:] = self.model.predict(np.reshape(states[i], [1,self.state_size]))[0]
            targets_f[i][int(actions[i])] = targets[i]
        self.model.fit(states, targets_f, epochs=1, verbose=0)

    def train_batch_mountain_car(self, states, actions, rewards, next_states, dones, gamma):
        '''
	    Given sampled states, actions, rewards, next_states and done flags train the model for MountainCar at mini-batch level for given input. Note that a different function is required because if height>=0.5, then target label changes.
	'''
        batch_size = states.shape[0]
	targets = rewards
	targets_f = np.zeros((batch_size, self.num_actions), dtype='float')
	for i in range(0, batch_size):
	    if not dones[i]:
	        targets[i] = float(rewards[i] + gamma*np.amax(self.model.predict(np.reshape(next_states[i], [1,self.state_size]))[0]))
            elif dones[i] and next_states[i][0] >= 0.5:
	        targets[i] = float(rewards[i])
	    targets_f[i][:] = self.model.predict(np.reshape(states[i], [1,self.state_size]))[0]
	    targets_f[i][int(actions[i])] = targets[i]
	self.model.fit(states, targets_f, epochs=1, verbose=0)

    def train_batch_space_invaders(self, states, actions, rewards, next_states, dones, gamma, model_num):
        '''
	    Given sampled states, actions, rewards, next_states and done flags train the model for MountainCar at mini-batch level for given input. Here training is done on top of images as input representation of state.
	'''
        batch_size = states.shape[0]
        targets = rewards
        targets_f = np.zeros((batch_size, self.num_actions), dtype='float')
        model_target_num = 1
        for i in range(0, batch_size):
            if not dones[i]:
                targets[i] = float(rewards[i] + gamma*np.amax(self.get_action_prob(next_states[i], model_target_num)))
            targets_f[i][:] = self.get_action_prob(states[i], model_target_num)
            targets_f[i][int(actions[i])] = targets[i]
        if model_num == 1:
            self.model_1.fit(states, targets_f, epochs=1, verbose=0)
        elif model_num == 2:
            self.model_2.fit(states, targets_f, epochs=1, verbose=0)
        else:
            print('UNKNOWN MODEL NUMBER !')
            sys.exit(1)

class Replay_Memory():
    '''
        This class handles the experience table buffer.
	Following are the inputs requited to initialize this class:
	    (1). OpenAI gym environment. This environment is used to take random action and record state, reward, done flags, next state transition values.
	    (2). Memory size, this is the maximum number of entries a table can hold.
	    (3). Burn in, maximum number of episodes to fill the table initially.
	Following are the functionalities included in this clasS:
	    (1). Given the size of experience table, initially fill the transitions randomly.
	    (2). Given the mini-batch size, randomly sample the transitions for training.
	    (3). Given a new transition as input, append the transition in table and remove the oldest transition (a queue does it by default).
    '''
    def __init__(self, env, batch_size, memory_size=50000, burn_in=10000):
        self.memory = deque(maxlen=memory_size)
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.num_iterations = 1000000
        self.env = env
        self.batch_size = batch_size
        self.num_frames = 4

    def fill_random_transitions(self, env_name):
        '''
            Given an environment name, fill the experience table by taking random actions initially.
        '''
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
                action_prob = np.array([1,1,1], dtype='float')
                action_prob = action_prob/np.sum(action_prob)
                set_actions = range(0, self.env.action_space.n)
                for given_iter in range(self.num_iterations):
                    action = np.random.choice(set_actions, size=1, replace=False, p=action_prob)[0]
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
        '''
           Given a batch size randomly sample a mini-batch from created table of experiences.
        '''
        mini_batch = random.sample(self.memory, batch_size)
        return mini_batch

    def append(self, transition):
        '''
           Given a new transition which consists of states, actions, next states, done flags, rewards add the tranition to experience table
        '''
        self.memory.append(transition)

class DQN_Agent():
    '''
        This class contains the main functionalies required for an Agent to learn. It also contains the training and testing loops.
	Following are the inputs requited to initialize this class:
	    (1). Environment Name.
	    (2). Render: We always keep this as false.
	Following are the functionalies included in this class:
	    (1). Given an environment name, initialize the appropirate set of hyperparameters.
	    (2). Call the Model class and initalize the neural network.
	    (3). If training type is 'use_memory_replay': initialize the experience table by taking random actions.
	    (4). Train the model depending upon environment, train type and architecture.
	Note: 
	    (1). We are not passing the training type or network architecture as flags, rather we are setting this in class initialization depending upto the environment name.
	    (2). When not using memory reply we make updates to network in stochastic fashion that is one input at a time (not batches).
    '''
    def __init__(self, environment_name, render=False):
        '''
           - If we want to use memory replay. Set "self.train_type='use_replay_memory'", else set is as "self.train_type='no_replay_memory'"
        '''
        self.env_name = environment_name # This can be 'CartPole-v0', 'MountainCar-v0' or 'SpaceInvaders-v0'
        self.env = gym.make(environment_name)
        self.env.reset()
        if environment_name == 'CartPole-v0':
            self.gamma = float(0.99)
            self.model_type = 'ddqn' # This can be 'linear_dqn', 'dqn', 'ddqn'
            self.train_type = 'use_replay_memory' # This can be 'use_replay_memory', 'no_replay_memory'
        if environment_name == 'MountainCar-v0':
            self.gamma = float(1)
            self.train_type = 'no_replay_memory' # This can be 'use_replay_memory', 'no_replay_memory'
            self.model_type = 'dqn' # This can be 'linear_dqn', 'dqn', 'ddqn'
        if environment_name == 'SpaceInvaders-v0':
            self.gamma = float(1)
            self.model_type = 'dqn_space_invaders' # This can be only 'dqn_space_invaders'
            self.train_type = 'use_replay_memory_space_invaders_v0' # This can be only 'use_replay_memory_space_invaders_v0'
            
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
            self.train_type = 'use_replay_memory_space_invaders_v0'
        self.num_actions = self.env.action_space.n
        self.state_size = self.env.observation_space.shape[0]
        self.model = QNetwork(environment_name, self.model_type)
        self.num_iterations = 1000000
        self.num_episodes = 2000
        self.num_frames = 4

    def train(self):
        '''
	    With all the agent class paramaters set, this function trains the network.
	'''
        if self.train_type == 'no_replay_memory':
	    test_rewards = np.zeros((int(self.num_episodes/100)+1,1))
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
	        if given_episode%100 == 0 and given_episode != 0:
		    test_rewards[int(given_episode/100)] = self.test_cartpole(test_iters=1)
		if given_episode%int((self.num_episodes-1)/3) == 0 and given_episode != 0:
		    self.model.save_model_weights(self.model.model, 'model_'+self.env_name+'_'+str(given_episode)+'.h5')
            print(test_rewards)

        elif self.train_type == 'use_replay_memory' and self.env_name == 'MountainCar-v0':
	    rand_thresh = 1
	    test_rewards = np.zeros((int(self.num_episodes/100)+1,1))
            for given_episode in range(0, self.num_episodes):
                print('Train episode: ' + str(given_episode))
                state = self.env.reset()
                state = np.reshape(state, [1 , self.state_size])
		rand_thresh *= self.eps_decay_fact
		rand_thresh = max(rand_thresh, 0.1)
                for given_iter in range(0, self.num_iterations):
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
                    self.model.train_batch_mountain_car(states, actions, rewards, next_states, dones, self.gamma)
                if given_episode%100 == 0 and given_episode != 0:
                    test_rewards[int(given_episode/100)] = self.test(test_iters=1)
		if given_episode%int((self.num_episodes-1)/3) == 0 and given_episode != 0:
		    self.model.save_model_weights(self.model.model, 'model_'+self.env_name+'_'+str(given_episode)+'.h5')
            print(test_rewards)

        elif self.train_type == 'use_replay_memory':
	    test_rewards = np.zeros((int(self.num_episodes/100)+1,1))
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
                if given_episode%100 == 0 and given_episode != 0:
	            test_rewards[int(given_episode/100)] = self.test_cartpole(test_iters=1)
		if given_episode%int((self.num_episodes-1)/3) == 0 and given_episode != 0:
		    self.model.save_model_weights(self.model.model, 'model_'+self.env_name+'_'+str(given_episode)+'.h5')
	    print(test_rewards)

        elif self.train_type == 'use_replay_memory_space_invaders_v0':
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
                        action = self.model.get_action_image(np.stack(current_states_queue, axis=2), model_num = 1)
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
                    self.model.train_batch_space_invaders(states, actions, rewards, next_states, dones, self.gamma, model_num = 2)
                self.model.save_model_weights(self.model.model_2, 'model_init_'+self.env_name+'.h5')
                self.model.checkpoint_swap(self.env_name)
                self.test_image(test_iters=1)

    def test(self, test_iters, model_file=None):
        '''
	    Testing function which runs testing for test_iters number of times and retures the average reward. This is on state parameters.
	'''
        avg_reward = 0
        for given_episode in range(0, test_iters):
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
        print(float(avg_reward) / float(test_iters))
	return (float(avg_reward) / float(test_iters))

    def test_cartpole(self, test_iters, model_file=None):
        '''
	    Testing function which runs testing for test_iters number of times and retures the average reward. This is on state parameters. This function is specific to CartPole environment.
	'''
        avg_reward = 0
	for given_episode in range(0, test_iters):
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
	print(float(avg_reward) / float(test_iters))
	return(float(avg_reward) / float(test_iters))

    def test_image(self, test_iters, model_file=None):
        '''
	    Testing function which runs testing for test_iters number of times and retures the average reward. This is on input images directly . This function is specific to SpaceInvaders environment.
        '''
        avg_reward = 0
        for given_episode in range(0, test_iters):
            given_reward = 0
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
            for given_iter in range(0, self.num_iterations):
                action = self.model.get_action_image(np.stack(current_states_queue, axis=2), model_num = 2)
                next_state, reward, done, _ = self.env.step(action)
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

    def burn_in_memory(self):
        '''
            Initialize the experience table initially with random transitions.
        '''
        memory = Replay_Memory(self.env, self.batch_size)
        memory.fill_random_transitions(self.env_name)
        return memory


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env', dest='env', type=str)
    parser.add_argument('--render', dest='render', type=int, default=0)
    parser.add_argument('--train', dest='train', type=str)
    parser.add_argument('--model', dest='model_file', type=str)
    return parser.parse_args()


def main(args):
    args = parse_arguments()
    environment_name = args.env
    agent = DQN_Agent(environment_name)
    agent.train()
    if environment_name == 'CartPole-v0':
        agent.test_cartpole(test_iters=200)
    elif environment_name == 'MountainCar-v0':
        agent.test(test_iters=200)
    elif environment_name == 'SpaceInvaders-v0':
        agent.test_image(test_iters=200)


if __name__ == '__main__':
    main(sys.argv)

