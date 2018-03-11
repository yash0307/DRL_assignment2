Deep Reinforcement Learning - Homework 2
---------------------------------------------------------------------------------------------------------------------
General Notes:
(1). Entire codebase is in DQN_Implementation.py file.
(2). This script only takes the environment name as input from command line.
(3). All other parameters are set within the script at the time of initializing the DQN_Agent() class.
----------------------------------------------------------------------------------------------------------------------
Q.1.

Running Linear QN for CartPole and no experience replay:
On line 371 set "self.model_type='linear_dqn'"
On line 372 set "self.train_type='no_replay_memory'"
python DQN_Implementation.py --env=CartPole-v0

Running Linear DQN for MountainCar and no experience replay:
On line 375 set "self.train_type='no_replay_memory'"
On line 376 set "self.model_type='linear_dqn'"
python DQN_Implementation.py --env=MountainCar-v0

Whenever replay experience is not used. We do the training of network in a stochastic fashion (one sample at a time, no mini-batch).
-----------------------------------------------------------------------------------------------------------------------
Q.2.

Running Linear QN for CartPole and with experience replay:
On line 371 set "self.model_type='linear_dqn'"
On line 372 set "self.train_type='use_replay_memory'"
python DQN_Implementation.py --env=CartPole-v0

Running Linear DQN for MountainCar and with experience replay:
On line 375 set "self.train_type='use_replay_memory'"
On line 376 set "self.model_type='linear_dqn'"
python DQN_Implementation.py --env=MountainCar-v0

Wheneve replay experience is use. We do the training of network at mini-batch level. Batch size is kept as 32 always.
-----------------------------------------------------------------------------------------------------------------------
Q.3.

Running Deep Q-Network for CartPole with no experience replay:
On line 371 set "self.model_type='dqn'"
On line 372 set "self.train_type='no_replay_memory'"
python DQN_Implementation.py --env=CartPole-v0

Running Deep Q-Network for CartPole and with experience replay:
On line 371 set "self.model_type='dqn'"
On line 372 set "self.train_type='use_replay_memory'"
python DQN_Implementation.py --env=CartPole-v0

Running Deep Q-Network for MountainCar and no experience replay:
On line 375 set "self.train_type='no_replay_memory'"
On line 376 set "self.model_type='dqn'"
python DQN_Implementation.py --env=MountainCar-v0

Running Deep Q-Network for MountainCar and no experience replay:
On line 375 set "self.train_type='use_replay_memory'"
On line 376 set "self.model_type='dqn'"
python DQN_Implementation.py --env=MountainCar-v0

Wheneve replay experience is use. We do the training of network at mini-batch level. Batch size is kept as 32 always.
Whenever replay experience is not used. We do the training of network in a stochastic fashion (one sample at a time, no mini-batch).
-----------------------------------------------------------------------------------------------------------------------
Q.4.

Running Dueling DQN for CartPole with no experience replay:
On line 371 set "self.model_type='ddqn'"
On line 372 set "self.train_type='no_replay_memory'"
python DQN_Implementation.py --env=CartPole-v0

Running Dueling DQN for CartPole and with experience replay:
On line 371 set "self.model_type='ddqn'"
On line 372 set "self.train_type='use_replay_memory'"
python DQN_Implementation.py --env=CartPole-v0

Running Dueling DQN for MountainCar and no experience replay:
On line 375 set "self.train_type='no_replay_memory'"
On line 376 set "self.model_type='ddqn'"
python DQN_Implementation.py --env=MountainCar-v0

Running Dueling DQN for MountainCar and no experience replay:
On line 375 set "self.train_type='use_replay_memory'"
On line 376 set "self.model_type='ddqn'"
python DQN_Implementation.py --env=MountainCar-v0

Wheneve replay experience is use. We do the training of network at mini-batch level. Batch size is kept as 32 always.
Whenever replay experience is not used. We do the training of network in a stochastic fashion (one sample at a time, no mini-batch).
-----------------------------------------------------------------------------------------------------------------------
Q.5.

Running image based DQN for SpaceInvades with experience replay:
python DQN_Implementation.py --env=SpaceInvaders

------------------------------------------------------------------------------------------------------------------------
