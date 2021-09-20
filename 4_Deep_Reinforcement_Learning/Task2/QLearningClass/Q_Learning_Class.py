import numpy as np
from collections import namedtuple
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create a named tuple for each available actions
# Adapted from INM707 Reinforcement Module (Class of 2020)
Action = namedtuple('Action', 'name index delta_i delta_j')

up = Action('up', 0, -1, 0)
down = Action('down', 1, 1, 0)
left = Action('left', 2, 0, -1)
right = Action('right', 3, 0, 1)
pickup = Action('pickup', 4, 0, 0)
dropoff = Action('dropoff', 5, 0, 0)

index_to_actions = {}
for action in [up, down, left, right, pickup, dropoff]:
    index_to_actions[action.index] = action

str_to_actions = {}
for action in [up, down, left, right, pickup, dropoff]:
    str_to_actions[action.name] = action


# Create a neccessary function and class
# Adapted from INM707 Reinforcement Module (Class of 2020)
def run_single_exp(env, policy, q_values):
    '''
    A funciton that return a total reward and average reward per step
    when an agent finished one episode in the environment
    '''
    state = env.reset()
    done = False

    total_reward = 0
    step_taken = 0
    while not done:
        action = policy(state, q_values)
        state, reward, done = env.step(action)

        total_reward += reward
        step_taken += 1

    return total_reward, total_reward / step_taken

# Adapted from INM707 Reinforcement Module (Class of 2020)
def run_experiments(env, policy, policy_eval_algo, number_exp):
    '''
    A funciton that return
    1. Total reward
    2. Max reward
    3. Mean reward
    4. Variance
    when the agent run in the environment for many episode
    '''
    all_rewards = []
    reward_per_step_list = []

    for n in range(number_exp):
        final_reward, reward_per_step = run_single_exp(env, policy, policy_eval_algo.q_values)
        all_rewards.append(final_reward)
        reward_per_step_list.append(reward_per_step)

    max_reward = max(all_rewards)
    mean_reward = np.mean(all_rewards)
    var_reward = np.std(all_rewards)

    avg_avg_reward = np.mean(reward_per_step_list)

    return all_rewards, max_reward, mean_reward, var_reward, avg_avg_reward

#%% Create E_Greedy_Policy_Class


class E_Greedy_Policy:
    def __init__(self, epsilon, decay):

        self.epsilon = epsilon
        self.epsilon_start = epsilon

        self.decay = decay
        self.epsilon_min = 0.000001

    def __call__(self, state, q_values):
        # Sample an action from the policy, given a state

        is_greedy = random.random() > self.epsilon

        if is_greedy:
            index_action = np.argmax(q_values[state])
        else:
            index_action = random.randint(0, 5)

        action = index_to_actions[index_action].name
        return action

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.decay

        else:
            self.epsilon = self.epsilon_min

    def reset(self):
        self.epsilon = self.epsilon_start


# Create Q-Learning Class
class Q_Learning:

    def __init__(self, env, alpha, gamma):
        self.size_environment = env.N
        self.alpha = alpha
        self.gamma = gamma
        self.q_values = np.zeros((env.total_possible_states(), 6))

    def update_values(self, s_current, a_next, r_next, s_next):
        a_next_idx = str_to_actions[a_next].index

        self.q_values[s_current, a_next_idx] += self.alpha * (
                r_next + self.gamma * np.max(self.q_values[s_next]) -
                self.q_values[s_current, a_next_idx])

# A helper function that make the agent to adopt Q_Learning in the environment
def q_learning_episode(qleaning_method, env, policy):
    s = env.reset()
    done = False

    while not done:
        action = policy(s, qleaning_method.q_values)
        s_next, r, done = env.step(action)

        qleaning_method.update_values(s, action, r, s_next)
        # e_policy.update_epsilon()

        s = s_next

# %% Define helper function to plot the result

def get_result(env, alpha_list, gamma_list, epsilon, decay, eps):
    '''
    This is a helper function that take in list of alpha and gamma value.
    For each combination of alpha and gamma, this function will
    find the reward after train the environment up to the specify episode.

    Output type: dataframe
    '''
    eps_container = []
    max_container = []
    mean_container = []
    alpha_container = []
    gamma_container = []
    var_container = []
    avg_container = []
    print('Learning')
    for alpha in alpha_list:
        for gamma in gamma_list:

            print('alpha:', alpha, 'gamma:', gamma)
            policy = E_Greedy_Policy(epsilon, decay)
            learning_method = Q_Learning(env, alpha, gamma)

            for episode in range(eps):
                q_learning_episode(learning_method, env, policy)
                policy.update_epsilon()

                if (episode % 50 == 0) | (episode == eps - 1):
                    _, max_rew, mean_rew, var_rew, avg_rew = run_experiments(env, policy, learning_method, 100)

                    eps_container.append(episode)
                    max_container.append(max_rew)
                    mean_container.append(mean_rew)
                    var_container.append(var_rew)
                    alpha_container.append(alpha)
                    gamma_container.append(gamma)
                    avg_container.append(avg_rew)

    dict = {'alpha': alpha_container, 'gamma': gamma_container, 'epsiode': eps_container, 'max_reward': max_container,
            'mean_reward': mean_container,
            'variance': var_container, 'average_reward': avg_container}
    df_result = pd.DataFrame(dict)

    return df_result

def plot_reward_eps(df, alpha_list, gamma_list, fig_size):
    '''
    This is a helper function that take in the dataframe from 'get_result' function
    and plot the average reward against episode
    '''
    nrow = len(alpha_list)
    ncol = len(gamma_list)

    fig, axes = plt.subplots(nrow, ncol, figsize=fig_size)
    for row in range(nrow):
        for col in range(ncol):
            df_temp = df[(df['alpha'] == alpha_list[row]) & (df['gamma'] == gamma_list[col])]
            df_temp_melt = pd.melt(df_temp, id_vars=['epsiode'], value_vars=['max_reward', 'mean_reward'])
            sns.lineplot(x = "epsiode", y = "value", hue = 'variable', legend = 'full', data = df_temp_melt, ax = axes[row, col])

    pad = 5

    for ax, col in zip(axes[0], gamma_list):
        ax.annotate('gamma:' + str(col), xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    for ax, row in zip(axes[:, 0], alpha_list):
        ax.annotate('Alpha:' + str(row) , xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

def plot_var_eps(df, alpha_list, gamma_list, fig_size):
    '''
    This is a helper function that take in the dataframe from 'get_result' function
    and plot the variance against episode
    '''
    nrow = len(alpha_list)
    ncol = len(gamma_list)

    fig, axes = plt.subplots(nrow, ncol, figsize=fig_size)
    for row in range(nrow):
        for col in range(ncol):
            df_temp = df[(df['alpha'] == alpha_list[row]) & (df['gamma'] == gamma_list[col])]
            df_temp_melt = pd.melt(df_temp, id_vars=['epsiode'], value_vars=['variance'])
            df_temp_melt = df_temp_melt.rename(columns={'value': 'variance'})
            df_temp_melt.set_index('epsiode', inplace=True)
            sns.lineplot(data=df_temp_melt['variance'], ax=axes[row, col], color='orange')

    pad = 5

    for ax, col in zip(axes[0], gamma_list):
        ax.annotate('gamma:' + str(col), xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    for ax, row in zip(axes[:, 0], alpha_list):
        ax.annotate('Alpha:' + str(row), xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')


def plot_avg_rew_eps(df, alpha_list, gamma_list, fig_size):
    '''
    This is a helper function that take in the dataframe from 'get_result' function
    and plot the average reward per time step against episode
    '''
    nrow = len(alpha_list)
    ncol = len(gamma_list)

    fig, axes = plt.subplots(nrow, ncol, figsize=fig_size)
    for row in range(nrow):
        for col in range(ncol):
            df_temp = df[(df['alpha'] == alpha_list[row]) & (df['gamma'] == gamma_list[col])]
            df_temp_melt = pd.melt(df_temp, id_vars=['epsiode'], value_vars=['average_reward'])
            df_temp_melt = df_temp_melt.rename(columns={'value': 'average_reward_per_action'})
            df_temp_melt.set_index('epsiode', inplace=True)
            sns.lineplot(data=df_temp_melt['average_reward_per_action'], ax=axes[row, col], color='seagreen')

    pad = 5

    for ax, col in zip(axes[0], gamma_list):
        ax.annotate('gamma:' + str(col), xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    for ax, row in zip(axes[:, 0], alpha_list):
        ax.annotate('Alpha:' + str(row), xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')
        