from ppo_algorithm import *
import pickle


# %% Testing of the PPO algorithm

# Help function to run several experiments
def run_experiments(env, actor_model, num_exps):
    with torch.no_grad():
        all_rewards = []
        all_action_counts = []

        for _ in range(num_exps):
            obs = env.reset()
            done = False

            total_reward = 0
            action_count = 0

            while not done:
                dist = Categorical(actor_model(obs))
                action_index = torch.squeeze(dist.sample()).item()
                action_name = index_to_actions[action_index].name
                obs, reward, done = env.step(action_name)
                total_reward += reward
                action_count += 1

            all_rewards.append(total_reward)
            all_action_counts.append(action_count)

        max_reward = max(all_rewards)
        avg_reward = np.mean(all_rewards)
        var_reward = np.std(all_rewards)
        avg_action_count = np.mean(all_action_counts)

    return max_reward, avg_reward, var_reward, avg_action_count


if __name__ == '__main__':

    # Import the Actor model
    PATH_MLP = 'Actor_Model_10.pth'
    model = torch.load(PATH_MLP)
    model.eval()

    # Import the same environment as training
    FILENAME = open('dungeon_10.p', 'rb')
    dungeon = pickle.load(FILENAME)
    FILENAME.close()

    dungeon.reset()
    dungeon.display()

    # Running the 1000 experiments
    max_r, avg_r, var_r, avg_ac = run_experiments(dungeon, model, 1000)
    avg_r_per_ac = avg_r / avg_ac

    print(f'The environment size ({dungeon.size} x {dungeon.size}) of the dungeon shows the following result:')
    print(f'Maximum Reward: {max_r:0.2f} | Average Reward: {avg_r:0.2f} | Variance Reward: {var_r:0.2f} | '
          f'Average Action: {avg_ac:0.2f} | Average Reward Per Action: {avg_r_per_ac:0.2f}')
