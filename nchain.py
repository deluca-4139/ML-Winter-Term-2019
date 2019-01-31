import gym
import numpy as np
import random
import time

env = gym.make("NChain-v0") # Observation space: 5, Action space: 2
env.reset()

# Hyperparameters
max_epochs = 100
alpha = 0.1
gamma = 0.9
epsilon = 0.1

q_table = np.ones([env.observation_space.n, env.action_space.n])

# Training
print("Training...")
for i in range(1, 1000001):
    state = env.reset()

    epochs, reward = 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space through greedy epsilon
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, info = env.step(action) # Take a new action

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        # Q-learning algorithm
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value # Update Q-table

        if epochs == max_epochs:
            done = True

        state = next_state
        epochs += 1

    if i % 1000 == 0:
        print("{}...".format(i))

print("Training finished. Evaluating...")

# Evaluate after training
total_epochs, total_returns, total_rewards = 0, 0, 0
episodes = 1000

for _ in range(episodes):
    state = env.reset()
    epochs, reward = 0, 0

    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)
       
        if reward == 2:
            total_returns += 1
        elif reward == 10:
            total_rewards += 1

        if epochs == max_epochs:
            done = True

        epochs += 1

print("Result after {} episodes:".format(episodes))
print("Average amount of returns:", (total_returns / episodes))
print("Average amount of end chain rewards:", (total_rewards / episodes))
