import gym
import numpy as np
import random
import time

env = gym.make("FrozenLake8x8-v0").env
env.reset()

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.1

q_table = np.ones([env.observation_space.n, env.action_space.n])
counts = np.ones([env.observation_space.n, env.action_space.n])

# Training
print("Training...")
for i in range(1, 1000001):
    
    state = env.reset()

    epochs, penalties, reward = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space through greedy epsilon
        else:
            action = np.argmax(q_table[state])

        c = counts[state, action]
        alpha = 1 / c 

        next_state, reward, done, info = env.step(action) # Take a new action based on what we chose

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        # Q-learning algorithm using ^ values acquired above
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value # Update q-table
        counts[state, action] = c + 1

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

    if i % 1000 == 0:
        print("{}...".format(i))
        
print("Training finished. Evaluating...")

for x in range(env.observation_space.n):
    print(q_table[x])
'''
# Evaluate after training
total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0

    done = False

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)
        env.render()
        
        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print("Results after {} episodes:".format(episodes))
print("Average timesteps per episode:", (total_epochs / episodes))
print("Average penalties per episode:", (total_penalties / episodes))
'''

# Show a test run to the user
state = env.reset()
env.render()
done = False
time.sleep(2)

while not done:
    action = np.argmax(q_table[state])
    state, _, done, _ = env.step(action)
    env.render()
    time.sleep(1.5)

print("")
print("Complete!")

