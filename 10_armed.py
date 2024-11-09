import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_bandits = 10   # num of bandit arms, actions. it is a 10 armed testbed 
n_steps = 10000  # total num of time steps
epsilon = 0.1    # exploration rate for epsilon-greedy - 10% exploration probs.
sigma = 0.01     # Standard deviation for random walk, added at each step, ensuring nonstationarity

# Initialize true action values
q_star = np.zeros(n_bandits)  # represents true reward value for each action (arm)
q_star_walk = np.random.normal(0, sigma, n_bandits)  # random noise sampled from N(0,sigma2)

# Initialize estimated values & metrics
Q_sample = np.zeros(n_bandits)  # stores estimated action values Q(a) for sample-average method
Q_fixed = np.zeros(n_bandits)   # stores estimated Q(a) for fixed step-size method
N = np.zeros(n_bandits)         # tracks how many times each action is selected

# Track performance metrics
rewards_sample = []  # stores rewards obtained at each time step
rewards_fixed = []   # for each method
optimal_sample = []  # track whether optimal action (action with highest Q(a))
optimal_fixed = []   # was selected at each time step for each method

# Simulate the Bandit Problem
for t in range(n_steps):
    # random walk for nonstationary rewards
    q_star += np.random.normal(0, sigma, n_bandits)  # add random noise
    optimal_action = np.argmax(q_star)  # index of action w/ highest true value at current time step

    # Sample AVG method
    if np.random.rand() < epsilon:
        action_sample = np.random.choice(n_bandits)  # Exploration
    else:
        action_sample = np.argmax(Q_sample)  # Exploitation
    
    reward_sample = np.random.normal(q_star[action_sample], 1)  # reward for chosen action, sampled from
                                                                # a normal distrib. around q*(a) with variance 1
    N[action_sample] += 1  # increment action counter
    Q_sample[action_sample] += (reward_sample - Q_sample[action_sample]) / N[action_sample]  # update action value
    rewards_sample.append(reward_sample)  # store reward
    optimal_sample.append(action_sample == optimal_action)  # store whether optimal action was selected

    # Fixed step-size method
    if np.random.rand() < epsilon:
        action_fixed = np.random.choice(n_bandits)  # Exploration
    else:
        action_fixed = np.argmax(Q_fixed)  # Exploitation
    
    reward_fixed = np.random.normal(q_star[action_fixed], 1)  # reward with noise
    Q_fixed[action_fixed] += 0.1 * (reward_fixed - Q_fixed[action_fixed])  # update action value
    rewards_fixed.append(reward_fixed)  # store reward
    optimal_fixed.append(action_fixed == optimal_action)  # store whether optimal action was selected

# Plot results

# compares avg reward over time for both methods. Use cumulative sums to calc running avg.
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(np.cumsum(rewards_sample) / np.arange(1, n_steps+1), label='Sample Averages')
plt.plot(np.cumsum(rewards_fixed) / np.arange(1, n_steps+1), label='Fixed Step Size (alpha=0.1)')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.legend()

# compares % of optimal actions selected over time for both methods
plt.subplot(2, 1, 2)
plt.plot(np.cumsum(optimal_sample) / np.arange(1, n_steps+1), label='Sample Averages')
plt.plot(np.cumsum(optimal_fixed) / np.arange(1, n_steps+1), label='Fixed Step Size (alpha=0.1)')
plt.xlabel('Steps')
plt.ylabel('% Optimal Action')
plt.legend()

plt.show()