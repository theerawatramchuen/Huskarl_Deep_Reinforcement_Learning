# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 08:49:22 2019
Source : https://medium.com/@tensorflow/introducing-huskarl-the-modular-deep-reinforcement-learning-framework-e47d4b228dd3
@author: User
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import huskarl as hk
import gym

# Setup gym environment
create_env = lambda: gym.make('CartPole-v0').unwrapped
dummy_env = create_env()

# Build a simple neural network with 3 fully connected layers as our model
model = Sequential([
  Dense(16, activation='relu', input_shape=dummy_env.observation_space.shape),
  Dense(16, activation='relu'),
  Dense(16, activation='relu'),
])

# Create Deep Q-Learning Network agent
agent = hk.agent.DQN(model, actions=dummy_env.action_space.n, nsteps=2)

# Create simulation, train and then test
sim = hk.Simulation(create_env, agent)
sim.train(max_steps=3000, visualize=True)
print ("Complte train 3000 steps")
print ("Start Test 1000 steps")
sim.test(max_steps=1000)
