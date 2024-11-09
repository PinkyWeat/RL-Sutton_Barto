import numpy as np
import matplotlib.pyplot as plt


def nonstationary_bandit(n_bandits, n_steps, sigma):
    """ Define a nonstationary environment"""
    # Initialize the true action values
    q_star = np.zeros(n_bandits)
    reward = np.zeros(n_steps)

    for t in range(n_steps):
        # add random walk noise
        q_star += np.random.normal(0, sigma, n_bandits)
        yield q_star

def epsilon_greedy(n_bandits, n_steps, alpha, epsilon, environment):
    """ Epsilon-greedy algorithm """
    Q = np.zeros(n_bandits)  # initialize Q values
    rewards = []

    for t, q_star in enumerate(environment):
        # exploration vs explotation
        if np.random.rand() < epsilon:
            action = np.random.choice(n_bandits)  # explores
        else:
            action = np.argmax(Q)  # exploits
    
        # recieve reward
        reward = np.random.normal(q_star[action], 1)
        rewards.append(reward)

        # update Q-value using constant step size
        Q[action] += alpha * (reward - Q[action])

    return np.array(rewards)

def parameter_study(n_bandits=10, n_steps=200000, sigma=0.01, alpha=0.1, epsilons=None, n_runs=50):
    if epsilons is None:
        epsilons = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4]
    
    avg_rewards = []

    for epsilon in epsilons:
        rewards = []
        for _ in range(n_runs):
            # generate the nonstationary environment
            env = nonstationary_bandit(n_bandits, n_steps, sigma)
            run_rewards = epsilon_greedy(n_bandits, n_steps, alpha, epsilon, env)
            rewards.append(np.mean(run_rewards[-100000:]))

        # compute the avg reward over the last 100,000 steps
        avg_rewards.append(np.mean(rewards))
    
    return epsilons, avg_rewards


# visualization
epsilons, avg_rewards = parameter_study(n_runs=50)

plt.figure(figsize=(8, 6))
plt.plot(epsilons, avg_rewards, marker='o')
plt.xscale('log')
plt.xlabel(r"$\epsilon$")
plt.ylabel("Average Reward (Last 100,000 steps)")
plt.title(r"Parameter Study: Average Reward vs $\epsilon$")
plt.grid()
plt.show()    