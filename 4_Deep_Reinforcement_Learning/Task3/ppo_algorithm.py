from Environment.dungeon import *
from neural_network import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


# %% Constructing the PPO algorithm

class PPO:
    def __init__(self, env, params):

        # Extract the environment information
        self.env = env
        self.obs_dim = env.observation_space()  # input size
        self.time_limit = env.time_limit

        # Initialise the values of necessary parameters
        self.total_iteration = 1000
        self.epochs = 5  # Number of times to update actor/critic per iteration
        self.clip = 0.2  # As recommended by the paper
        self.num_exp = 100

        # Hyperparameter tuning considered
        self.n_layers = params['n_layers']
        self.hidden_size = params['hidden_size']
        self.learning_rate = params['learning_rate']
        self.gamma = params['gamma']
        self.timesteps_per_batch = self.time_limit * params['multiplier']

        # Initialise actor and critic neural network
        self.actor = ActorCriticNN(input_size=self.obs_dim, n_layers=self.n_layers,
                                   hidden_size=self.hidden_size, output_size=4).to(device)
        self.critic = ActorCriticNN(input_size=self.obs_dim, n_layers=self.n_layers,
                                    hidden_size=self.hidden_size, output_size=1).to(device)

        # Initialise optimizers
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.learning_rate)

    # Help function to train the PPO algorithm
    def learning(self, save_model=False):

        # Define lists to store the results each episode to plot the graph
        all_iterations = []
        all_avg_rewards = []
        all_var_rewards = []
        all_avg_action_count = []

        # To collect the best average reward per action
        best_avg_reward_per_action = -9999

        # Define early stopping criteria
        early_stopping = 200
        early_stopping_counter = 0

        # Define lists to collect the results
        actor_loss_log = []
        critic_loss_log = []
        t_so_far = 0  # Timesteps simulated so far

        for i in range(1, self.total_iteration+1):  # ALGORITHM STEP 2

            # ALGORITHM STEP 3
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Calculate V_phi_k
            V, _ = self.evaluate(batch_obs, batch_acts)

            # ALGORITHM STEP 5
            # Calculate advantage
            A_k = batch_rtgs - V.detach()
            # Normalize advantages (1e-10 to prevent dividing by 0)
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # ALGORITHM STEP 6 & 7 to Optimise policy
            for _ in range(self.epochs):
                # Calculate V_phi_k and pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # Calculate ratios
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                # Calculate actor loss
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Append the actor and critic loss (log)
                actor_loss_log.append(actor_loss.detach())
                critic_loss_log.append(critic_loss.detach())

            if i % 10 == 0:
                max_reward, avg_reward, var_reward, avg_action_count = self.run_experiments()
                avg_reward_per_action = avg_reward / avg_action_count

                print(f'Iteration: {i:4} | Avg. Reward: {avg_reward:.2f} | Var. Reward: {var_reward:.4f} '
                      f'| Avg. Action Count: {avg_action_count:.2f} | Actor Loss: {actor_loss_log[-1]:.4f} '
                      f'| Critic Loss: {critic_loss_log[-1]:.4f}')
                print(f'>> Avg. Reward per Action: {avg_reward_per_action:.4f}')

                # Append the average reward and action count to lists
                all_iterations.append(i)
                all_avg_rewards.append(avg_reward)
                all_var_rewards.append(var_reward)
                all_avg_action_count.append(avg_action_count)

                if avg_reward_per_action > best_avg_reward_per_action:  # Condition to look at the best score
                    best_avg_reward_per_action = avg_reward_per_action
                    early_stopping_counter = 0
                    if save_model:  # Condition to save the model
                        FILENAME_A = 'Actor_Model_' + str(self.env.size) + '.pth'
                        torch.save(self.actor, FILENAME_A)
                else:
                    early_stopping_counter += 10

                if early_stopping_counter == early_stopping:  # Early stopping criteria for improvement
                    print(f'Early stopping: Have no improvement since {i-200} iterations \n')
                    break

        # Use the best average reward per action to find the best optimal hyperparameters
        return best_avg_reward_per_action, all_iterations, all_avg_rewards, all_var_rewards, all_avg_action_count

    # Help function to create set of trajectories (batch)
    def rollout(self):
        # Batch Data
        batch_obs = []          # batch observations
        batch_acts = []         # batch actions
        batch_log_probs = []    # log probability of each action
        batch_rews = []         # batch rewards
        batch_rtgs = []         # batch rewards-to-go
        batch_lens = []         # episodic lengths in batch

        t = 0  # Keeps track of how many timesteps we've run so far this batch
        ep_t = 0

        while t < self.timesteps_per_batch:
            ep_rews = []  # Rewards this episode
            obs = self.env.reset()

            for ep_t in range(self.time_limit):

                # Increment timesteps ran this batch so far
                t += 1

                # Collect observation
                batch_obs.append(obs)

                index_action, log_prob = self.get_action(obs)
                action = index_to_actions[index_action].name
                obs, rew, done = self.env.step(action)

                # Collect reward, action, and log prob
                ep_rews.append(rew)
                batch_acts.append(index_action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1)  # plus 1 because timesteps starts at 0
            batch_rews.append(ep_rews)

        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float).to(device)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float).to(device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).to(device)
        # ALGORITHM STEP 4
        batch_rtgs = self.compute_rtgs(batch_rews)

        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    # Help function to get the action index and its log probability
    def get_action(self, obs):
        dist = Categorical(self.actor(obs))
        action_index = dist.sample()
        # Calculate log probability
        log_prob = torch.squeeze(dist.log_prob(action_index)).item()
        action_index = torch.squeeze(dist.sample()).item()

        return action_index, log_prob

    # Help function to compute reward to go
    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (number of timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode backwards to maintain same order in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0  # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).to(device)

        return batch_rtgs

    # Help function to calculate value function
    def evaluate(self, batch_obs, batch_acts):
        # Query Critic network for a value (V) for each obs in batch_obs.
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        dist = Categorical(self.actor(batch_obs))
        log_probs = torch.squeeze(dist.log_prob(batch_acts))

        return V, log_probs

    # Help function to run several experiments
    def run_experiments(self):
        self.actor.eval()

        with torch.no_grad():
            all_rewards = []
            all_action_counts = []

            for _ in range(self.num_exp):
                obs = self.env.reset()
                done = False

                total_reward = 0
                action_count = 0

                while not done:
                    dist = Categorical(self.actor(obs))
                    action_index = torch.squeeze(dist.sample()).item()
                    action_name = index_to_actions[action_index].name
                    obs, reward, done = self.env.step(action_name)

                    total_reward += reward
                    action_count += 1

                all_rewards.append(total_reward)
                all_action_counts.append(action_count)

            max_reward = max(all_rewards)
            avg_reward = np.mean(all_rewards)
            var_reward = np.std(all_rewards)
            avg_action_count = np.mean(all_action_counts)

        self.actor.train()
        self.critic.train()

        return max_reward, avg_reward, var_reward, avg_action_count
