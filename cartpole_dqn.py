import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

episodes = 1000

# Agent
class dqn:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95 # Discount rate
        self.epsilon = 1.0 # Exploration rate
        self.epsilon_min = 0.0
        self.epsilon_decay = 0.9975
        self.learning_rate = 0.001
        self.model = self._build_model()

    # Neural network model initialization
    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # Q memory handling
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Choose an action
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # Epsilon greedy
        act_values = self.model.predict(state)
        return np.argmax(act_values[0]) # Else, return actual prediction

    # Random sample of memories
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay





# Training

env = gym.make('CartPole-v1')
agent = dqn(env.observation_space.shape[0], env.action_space.n)
state_size = env.observation_space.shape[0]
all_rewards = deque()
all_epsilons = deque()

for e in range(episodes):
    # Reset state
    state = env.reset()
    state = np.reshape(state, [1, state_size])

    # Goal is to keep the pole upright as long as possible until failure or score of 5000
    for time_t in range(5000):
        action = agent.act(state)

        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        reward = reward if not done else -10

        agent.remember(state, action, reward, next_state, done)

        state = next_state

        if done:
            print("Episode: {}/{}, Score: {} ~ e: {:.2}".format(e, episodes, time_t, agent.epsilon))
            all_rewards.append(time_t)
            all_epsilons.append(agent.epsilon)
            break
        
        if len(agent.memory)> 32:
            agent.replay(32)

# Show an example
'''
for _ in range(10):
    new_state = env.reset()
    new_state = np.reshape(new_state, [1, state_size])
    env.render()
    new_done = False
    while not new_done:
        new_action = agent.act(new_state)
        next_new_state, _, new_done, _ = env.step(new_action)
        next_new_state = np.reshape(next_new_state, [1, state_size])
        env.render()
'''

# Plot some data
plt.plot(list(range(1, episodes+1)), all_rewards, 'r-', list(range(1, episodes+1)), all_epsilons, 'b-')
plt.show()
