import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random as rand


# Hyperparameters
beta = 0.9         # Learning rate
gamma = 0.9        # Discount factor
epsilon = 0.1     # Exploration rate

num_of_ind_runs = 15
num_episodes = 1000


averaged_reward = np.zeros(num_episodes)


env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False)
state_size = env.observation_space.n
action_size = env.action_space.n
for run in range(num_of_ind_runs):
    qtable = np.zeros((state_size, action_size))

    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        too_long = False
        t = 0
        rng = np.random.default_rng()
        
        while(not done and not too_long) or t < 200:
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                if np.sum(qtable[state, :]) == 0:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(qtable[state, :])


            next_state, reward, done,too_long , _ = env.step(action)



            qtable[state, action] = qtable[state, action] + beta * (reward + gamma * np.max(qtable[next_state, :]) - qtable[state, action])


            state = next_state
            averaged_reward[episode] += reward
            t += 1
averaged_reward = averaged_reward/(num_of_ind_runs)




env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False)
state_size = env.observation_space.n
action_size = env.action_space.n 
averaged_reward_a = np.zeros(num_episodes)
for run in range(num_of_ind_runs):
    # Initialize Q-table to zeros
    qtable = np.zeros((state_size, action_size))

    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        too_long = False
        rng = np.random.default_rng()
        
        while not done:
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:   
                max_indeces = np.where(qtable[state, :] == qtable[state, :].max())[0]
                action =  rand.choice(max_indeces)

            next_state, reward, terminated,too_long , _ = env.step(action)
            done = terminated or too_long


            if terminated and reward == 0:
                reward = -1
            elif state == next_state:
                reward = -1



            qtable[state, action] +=  beta * (reward + gamma * max(qtable[next_state, :]) - qtable[state, action])



            state = next_state
            if reward == 1:
                averaged_reward_a[episode] += reward

averaged_reward_a = averaged_reward_a/(num_of_ind_runs)


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.plot(averaged_reward, 'b')
plt.plot(averaged_reward_a, 'r')
plt.show()