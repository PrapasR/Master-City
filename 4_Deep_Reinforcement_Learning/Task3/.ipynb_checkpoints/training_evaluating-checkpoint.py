from ppo_algorithm import *
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import optuna
from optuna.visualization import plot_optimization_history


# %% The function to train and tune hyperparameters

# Help function to plot experiments of the Q-Learning training
def visualise_training(all_iterations, all_avg_rewards, all_var_rewards, all_avg_action_counts, best_params):
    # Crete dataframe
    df = pd.DataFrame(zip(all_iterations, all_avg_rewards, all_var_rewards, all_avg_action_counts),
                      columns=['iteration', 'avg_reward', 'var_reward', 'avg_action_count'])

    # Define the optimal parameters from tuning
    best_n_layers = best_params['n_layers']
    best_hidden_size = best_params['hidden_size']
    best_learning_rate = best_params['learning_rate']
    best_gamma = best_params['gamma']
    best_multiplier = best_params['multiplier']

    # Plot the PPO training graphs (the values from running experiments 100 times at every 10 iterations)
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 10))
    fig.suptitle('PPO Learning Plots with the Optimal Hyperparameters', fontsize=22)

    axs[0].plot(df['iteration'], df['avg_reward'])
    axs[0].set_title(f'(Hidden Layers: {best_n_layers}, Hidden Size: {best_hidden_size}, '
                     f'Learning Rate: {best_learning_rate:.4f}, Gamma: {best_gamma:.4f}, '
                     f'Multiplier: {best_multiplier})', fontsize=14)
    axs[0].set_ylabel('Avg. Reward', fontsize=14)

    axs[1].plot(df['iteration'], df['var_reward'])
    axs[1].set_ylabel('Var. Reward', fontsize=14)

    axs[2].plot(df['iteration'], df['avg_action_count'])
    axs[2].set_ylabel('Avg. Action C.', fontsize=14)
    axs[2].set_xlabel('Iterations', fontsize=14)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':

    # Define an objective function to tune hyperparameters by Optuna framework
    # (similar to random search but it's better)
    def objective(trial):
        # Define tuning parameters
        params = {
            'n_layers': trial.suggest_int('n_layers', 1, 2),
            'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256, 512]),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
            'gamma': trial.suggest_uniform('gamma', 0.2, 0.8),
            'multiplier': trial.suggest_int('multiplier', 3, 5)
        }

        best_avg_reward_per_action, _, _, _, _ = PPO(dungeon, params).learning(save_model=False)

        return best_avg_reward_per_action

    # Define the environment
    dungeon = IceDungeonPO(10)
    dungeon.reset()

    # Save the environment for evaluating the results
    FILENAME_ENV = open('dungeon_' + str(dungeon.size) + '.p', 'wb')
    pickle.dump(dungeon, FILENAME_ENV)
    FILENAME_ENV.close()

    # Create a study object and optimise the objective function
    study = optuna.create_study(direction='maximize')

    # Limit tuning iteration at 5 times
    study.optimize(objective, n_trials=5)

    print('\nBest trial:')
    trial_ = study.best_trial

    print('Best parameters:', trial_.params)

    # Training with the optimal hyperparameters
    _, iter_lst, avg_r_lst, var_r_lst, avg_ac_lst = PPO(dungeon, trial_.params).learning(save_model=True)

    # Visualise the optimisation history
    plot_optimization_history(study)

    # Visualise experiments of the Q-Learning training
    visualise_training(iter_lst, avg_r_lst, var_r_lst, avg_ac_lst, trial_.params)
