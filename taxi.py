import gym
import numpy as np
import random
import time
from IPython.display import clear_output

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# Plotting metrics
all_epochs = []
all_penalties = []

env = gym.make("Taxi-v2").env
env.reset()

q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Training!
print("Training...")
for i in range(1, 10001):
    state = env.reset()

    epochs, penalties, reward = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space through greedy epsilon
        else:
            action = np.argmax(q_table[state]) # Exploit learned q-values

        next_state, reward, done, info = env.step(action) # Take a new action based on what we chose

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        # Q-learning algorithm based on ^ values acquired above
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value # Update q-table

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

print("Training finished.")

for x in range(500):
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

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print("Results after {} episodes: ".format(episodes))
print("Average timesteps per episode: ", (total_epochs / episodes))
print("Average penalties per episode: ", (total_penalties / episodes))
'''

# Show a few test runs to the user
for _ in range(5):
    state = env.reset()
    env.render()
    done = False
    time.sleep(1.5)

    while not done:
        action = np.argmax(q_table[state])
        state, _, done, _ = env.step(action)
        env.render()
        time.sleep(1)

    print("Complete!")
    time.sleep(1)
    print("============")
    print("")
    print("")
        
