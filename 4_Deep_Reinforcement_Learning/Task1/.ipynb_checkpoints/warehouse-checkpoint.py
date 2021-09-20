import numpy as np
import random
import copy

# Task1: Develop a tabular Reinforcement Learning environment that follows a Markov Decision Process.
# Adapted from  'INM-707 Deep Reinforcement Learning Exercise' (Class of 2021).

# %% Warehouse Environment

'''
Rules:
    - Constructing the environment with specifying the size of its, and random placing obstacles, 
      a parcel and a destination at beginning.
    - There are 3 types of obstacles, which are stationary obstacles (N/2), humans (N/10) and a box.
    - All humans will move within the boundary 2x2 to the left of their starting position and only move to free spaces; 
      humans have 60% of not moving, and the rest will be shared among the remaining valid actions.
    - The box is moved toward free space, when the agent accidentally touches it.
    - Randomly locating the position of agent around the edge of the environment when resetting the environment.
    - There are six actions of the agent, which are up, down, left, right, pick up and drop off.
    - The agent cannot move when it crashes into the stationary obstacles.
    - [Empty, Stationary Obstacles, Box, Human, Parcel, Destination, Agent] > [0, 1, 2, 3, 4, 5, 6].

Reward Conditions:
    - Each action (-1).
    - Crashing stationary obstacles (-5).
    - Crashing a box (-20). --> huge punishment because of possible damage to goods
    - Crashing humans (-N^2 / 2).
    - Picking a parcel at valid location (N^2 / 2).
    - Picking a parcel at invalid locations (-10).
    - Dropping-off the parcel at a correct destination (N^2 / 2).
    - Dropping-off the parcel at a incorrect destination (-10).
    
Terminate Conditions:
    - Running out of battery (Battery life time: N^2).
    - Crashing the humans (safety first).
    - Reaching the destination.

'''


class Warehouse:

    def __init__(self, N):
        # Define warehouse size and the number of environment and action
        self.N = N
        self.warehouse = np.zeros((N, N), dtype=np.int8)

        # Define the edge of environment
        self.warehouse[0, :] = self.warehouse[-1, :] = self.warehouse[:, 0] = self.warehouse[:, -1] = 1

        # Random placing stationary obstacles (N/2)
        empty_coordinates = self.get_placing_coordinates('empty', int(self.N / 2))
        self.warehouse[empty_coordinates[0], empty_coordinates[1]] = 1

        # Random placing a box (Only 1)
        empty_coordinates = self.get_placing_coordinates('empty', 1)
        self.warehouse[empty_coordinates[0], empty_coordinates[1]] = 2

        # Random placing humans (N/10)
        empty_coordinates = self.get_placing_coordinates('empty', int(self.N / 10))
        self.warehouse[empty_coordinates[0], empty_coordinates[1]] = 3
        self.origin_human_coordinates = copy.deepcopy(empty_coordinates)
        self.now_human_coordinates = copy.deepcopy(empty_coordinates)

        # The agent is placed after reset the environment
        self.position_agent = None

        # Initial placing a parcel and a destination
        self.position_parcel = np.asarray(self.get_placing_coordinates('empty', 1))
        self.warehouse[self.position_parcel[0], self.position_parcel[1]] = 4
        self.original_parcel_position = self.position_parcel  # Store the original location of parcel
        self.position_destination = np.asarray(self.get_placing_coordinates('empty', 1))
        self.warehouse[self.position_destination[0], self.position_destination[1]] = 5

        # Define battery condition
        self.battery_elapsed = 0
        self.full_battery = self.N ** 2

        # Define objective boolean
        self.pickup = False
        self.dropoff = False

        # Define parameters to check the values within the boundary of humans' movements
        self.left_value = None
        self.down_value = None
        self.diagonal_value = None
        self.origin_value = None

        # Define a list to collect humans coordinates
        self.human_coordinates_list = []

        # Display help
        self.dict_map_display = {0: '.',
                                 1: 'X',
                                 2: 'B',
                                 3: 'H',
                                 4: 'P',
                                 5: 'D',
                                 6: 'A'}

    # Help function to control agent's movements in the environment
    def step(self, agent_action):
        # At every action, the agent receives a negative reward
        reward = -1
        bump_obstacles = False
        bump_box = False

        # Define the parameters used to check each condition
        next_position = None
        behind_box_position = None
        behind_box_type = None

        # The actions of the agent are 'up', 'down', 'left', 'right','dropoff','pickup'
        if agent_action == 'up':
            next_position = np.array((self.position_agent[0] - 1, self.position_agent[1]))

        if agent_action == 'down':  # If condition prevent the agent looks out the environment
            if (self.position_agent[0] + 1) > (self.N - 1):
                next_position = self.position_agent
            else:
                next_position = np.array((self.position_agent[0] + 1, self.position_agent[1]))

        if agent_action == 'left':
            next_position = np.array((self.position_agent[0], self.position_agent[1] - 1))

        if agent_action == 'right':  # If condition prevent the agent looks out the environment
            if (self.position_agent[1] + 1) > (self.N - 1):
                next_position = self.position_agent
            else:
                next_position = np.array((self.position_agent[0], self.position_agent[1] + 1))

        if agent_action == 'pickup':
            next_position = self.position_agent

        if agent_action == 'dropoff':
            next_position = self.position_agent

        # If the agent bumps into the stationary obstacles, it doesn't move
        if self.warehouse[next_position[0], next_position[1]] == 1:
            bump_obstacles = True

        # If the agent bumps into the box, it move and push the box further (If there are free space behind the box)
        elif self.warehouse[next_position[0], next_position[1]] == 2:
            bump_box = True
            # The actions of the box are 'up', 'down', 'left', or 'right'
            if agent_action == 'up':
                behind_box_position = np.array((next_position[0] - 1, next_position[1]))
            if agent_action == 'down':
                behind_box_position = np.array((next_position[0] + 1, next_position[1]))
            if agent_action == 'left':
                behind_box_position = np.array((next_position[0], next_position[1] - 1))
            if agent_action == 'right':
                behind_box_position = np.array((next_position[0], next_position[1] + 1))
            behind_box_type = self.warehouse[behind_box_position[0], behind_box_position[1]]

        else:
            self.position_agent = next_position

        # If behind the box is free
        if bump_box and (behind_box_type == 0):
            self.position_agent = next_position
            self.warehouse[next_position[0], next_position[1]] = 0
            self.warehouse[behind_box_position[0], behind_box_position[1]] = 2

        # Calculate reward
        if bump_obstacles:  # Crush stationary obstacles
            reward -= 5

        if bump_box:  # Crush a box
            reward -= 20

        current_cell_type = self.warehouse[self.position_agent[0], self.position_agent[1]]

        if current_cell_type == 3:  # Crush humans
            reward -= (self.N ** 2) / 2

        if agent_action == 'pickup':  # Pick up reward's conditions
            if (current_cell_type == 4) & (not self.pickup):
                self.pickup = True
                reward += (self.N ** 2) / 2
                self.warehouse[self.position_parcel[0], self.position_parcel[1]] = 0
            else:
                reward -= 10

        if agent_action == 'dropoff':  # Drop-off reward's conditions
            if (current_cell_type == 5) & self.pickup:
                self.dropoff = True
                reward += (self.N ** 2) / 2
            else:
                reward -= 10

        # Move Human
        self.moving_human()

        # Get current state
        state = self.encode()

        # Update time
        self.battery_elapsed += 1

        # Verify termination condition
        done = False
        if self.battery_elapsed == self.full_battery:
            done = True
        if current_cell_type == 3:
            done = True
        if self.dropoff:
            done = True

        return state, reward, done

    # Help Function to move humans
    def moving_human(self):
        # Define the parameters used to check each condition
        number_humans = None
        relative_y = None
        relative_x = None

        # Check the number of humans
        if (type(self.origin_human_coordinates)) is tuple:
            number_humans = len(self.origin_human_coordinates[0])
        elif (type(self.origin_human_coordinates)) is np.ndarray:
            number_humans = 1

        # Move condition
        for human in range(number_humans):
            action_list = ['up', 'down', 'right', 'left']

            # Define parameters to check for non-valid humans' actions
            remove_up = False
            remove_down = False
            remove_right = False
            remove_left = False

            # Check valid movement constrain by the environment rules
            prev_space_coordinates = copy.deepcopy(self.now_human_coordinates)
            # Calculate relative position
            if (type(self.origin_human_coordinates)) is tuple:
                relative_y = self.origin_human_coordinates[0][human] - self.now_human_coordinates[0][human]
                relative_x = self.origin_human_coordinates[1][human] - self.now_human_coordinates[1][human]

                self.left_value = self.warehouse[
                    self.origin_human_coordinates[0][human], self.origin_human_coordinates[1][human] - 1]
                self.down_value = self.warehouse[
                    self.origin_human_coordinates[0][human] + 1, self.origin_human_coordinates[1][human]]
                self.diagonal_value = self.warehouse[
                    self.origin_human_coordinates[0][human] + 1, self.origin_human_coordinates[1][human] - 1]
                self.origin_value = self.warehouse[
                    self.origin_human_coordinates[0][human], self.origin_human_coordinates[1][human]]

            elif (type(self.origin_human_coordinates)) is np.ndarray:
                relative_y = self.origin_human_coordinates[0] - self.now_human_coordinates[0]
                relative_x = self.origin_human_coordinates[1] - self.now_human_coordinates[1]

                self.left_value = self.warehouse[
                    self.origin_human_coordinates[0], self.origin_human_coordinates[1] - 1]
                self.down_value = self.warehouse[
                    self.origin_human_coordinates[0] + 1, self.origin_human_coordinates[1]]
                self.diagonal_value = self.warehouse[
                    self.origin_human_coordinates[0] + 1, self.origin_human_coordinates[1] - 1]
                self.origin_value = self.warehouse[
                    self.origin_human_coordinates[0], self.origin_human_coordinates[1]]

            # Check condition to move human in valid space
            if (relative_y == 0) & (relative_x == 0):
                remove_up = True
                remove_right = True
                if self.left_value != 0:
                    remove_left = True
                if self.down_value != 0:
                    remove_down = True

            elif (relative_y == 0) & (relative_x == 1):
                remove_up = True
                remove_left = True
                if self.origin_value != 0:
                    remove_right = True
                if self.diagonal_value != 0:
                    remove_down = True

            elif (relative_y == -1) & (relative_x == 1):
                remove_left = True
                remove_down = True
                if self.left_value != 0:
                    remove_up = True
                if self.down_value != 0:
                    remove_right = True

            elif (relative_y == -1) & (relative_x == 0):
                remove_right = True
                remove_down = True
                if self.origin_value != 0:
                    remove_up = True
                if self.diagonal_value != 0:
                    remove_left = True

            # Pop-out movement in action_list for non-valid actions
            if remove_up:
                action_list.remove('up')
            if remove_down:
                action_list.remove('down')
            if remove_left:
                action_list.remove('left')
            if remove_right:
                action_list.remove('right')

            # Calculate probability of action
            if len(action_list) == 0:
                final_human_action = 'nothing'
            else:
                prob_each_action = int(40 / len(action_list))
                available_action_list = ['nothing']
                available_action_list = available_action_list + action_list
                action_prob = [60]

                for i in range(len(action_list)):
                    action_prob.append(prob_each_action)
                final_human_action = random.choices(available_action_list, weights=action_prob)
                final_human_action = final_human_action[0]

            # Change position of humans
            if final_human_action == 'nothing':
                pass
            elif (type(self.origin_human_coordinates)) is tuple:
                if final_human_action == 'up':
                    self.now_human_coordinates[0][human] -= 1
                elif final_human_action == 'down':
                    self.now_human_coordinates[0][human] += 1
                elif final_human_action == 'left':
                    self.now_human_coordinates[1][human] -= 1
                elif final_human_action == 'right':
                    self.now_human_coordinates[1][human] += 1

                self.warehouse[self.now_human_coordinates[0][human], self.now_human_coordinates[1][human]] = 3
                self.warehouse[prev_space_coordinates[0][human], prev_space_coordinates[1][human]] = 0

            elif (type(self.origin_human_coordinates)) is np.ndarray:
                if final_human_action == 'up':
                    self.now_human_coordinates[0] -= 1
                elif final_human_action == 'down':
                    self.now_human_coordinates[0] += 1
                elif final_human_action == 'left':
                    self.now_human_coordinates[1] -= 1
                elif final_human_action == 'right':
                    self.now_human_coordinates[1] += 1

                self.warehouse[self.now_human_coordinates[0], self.now_human_coordinates[1]] = 3
                self.warehouse[prev_space_coordinates[0], prev_space_coordinates[1]] = 0

    # Help function to display
    def display(self):
        env_with_agent = self.warehouse.copy()
        env_with_agent[self.position_agent[0], self.position_agent[1]] = 6

        full_repr = ""

        for r in range(self.N):

            line = ""

            for c in range(self.N):
                string_repr = self.dict_map_display[env_with_agent[r, c]]

                line += "{0:2}".format(string_repr)

            full_repr += line + "\n"

        print(full_repr)

    # Help function to check all blockers before placing the agent
    def check_blockers(self):

        list_blocking_obstacles = []
        check_list = [1, 2, 3]

        # horizontal checking
        for i in [1, self.N - 2]:
            for j in range(1, self.N - 1):
                if int(self.warehouse[i][j]) in check_list:
                    list_blocking_obstacles.append((i + 1, j))
                    list_blocking_obstacles.append((i - 1, j))

        # Vertical checking
        for j in [1, self.N - 2]:
            for i in range(1, self.N - 1):
                if int(self.warehouse[i][j]) in check_list:
                    list_blocking_obstacles.append((i, j + 1))
                    list_blocking_obstacles.append((i, j - 1))

        return list(set(list_blocking_obstacles))

    # Help function to get placing coordinates of each object
    def get_placing_coordinates(self, condition, n_cells):
        """
        :param condition: 'empty' and 'edge'
        :param n_cells: the number of require position
        :return: selected coordinates of the condition location
        """
        if condition == 'empty':
            empty_cells = np.where(self.warehouse == 0)
            random_choice = np.random.choice(np.arange(len(empty_cells[0])), n_cells)
            selected_coordinates = empty_cells[0][random_choice], empty_cells[1][random_choice]
            if n_cells == 1:
                return np.asarray(selected_coordinates).reshape(2, )

            return selected_coordinates

        elif condition == 'edge' and n_cells == 1:
            check_list = self.check_blockers()
            search = True
            while search:
                random_choice_n = np.random.choice(np.arange(1, self.N - 1), n_cells)  # prevent to get stuck in conner
                random_choice_edge = np.random.choice((0, self.N - 1), n_cells)
                chance = random.random() > 0.5

                if chance:  # Horizontal edge cells
                    selected_coordinates = random_choice_n[0], random_choice_edge[0]
                    if selected_coordinates not in check_list:
                        return np.asarray(selected_coordinates).reshape(2, )

                else:  # Vertical edge cells
                    selected_coordinates = random_choice_edge[0], random_choice_n[0]
                    if selected_coordinates not in check_list:
                        return np.asarray(selected_coordinates).reshape(2, )

    # Help function to convert object's coordinates to a position in the environment used for computing state
    def convert_to_position(self, considering_object):
        """
        :param considering_object: 'agent', 'box', 'human' and 'parcel'
        :return: object's position in the environment
        """

        if considering_object == 'agent':
            row, col = self.position_agent[0], self.position_agent[1]
            return (row * self.N) + col  # (between 0 and N*N-1)

        elif considering_object == 'box':
            # Get the box coordinates and convert to position
            box_coordinates = np.where(self.warehouse == 2)
            row, col = box_coordinates[0], box_coordinates[1]
            return (row * self.N) + col  # (between 0 and N*N-1)

        elif considering_object == 'human':
            self.human_coordinates_list = []
            human_position = None

            if (type(self.origin_human_coordinates)) is np.ndarray:
                relative_y = self.origin_human_coordinates[0] - self.now_human_coordinates[0]
                relative_x = self.origin_human_coordinates[1] - self.now_human_coordinates[1]

                if (relative_y == 0) & (relative_x == 0):
                    # Stay on the same cell when it was initiated
                    human_position = 0

                elif (relative_y == 0) & (relative_x == 1):
                    # Stay on the left cell compared to the position it was when the environment initiated
                    human_position = 1

                elif (relative_y == -1) & (relative_x == 1):
                    # Stay on the 'left diagonal'(South-West) cell compared to
                    # the position it was when the environment initiated
                    human_position = 2

                elif (relative_y == -1) & (relative_x == 0):
                    # Stay on the below cell compared to the position it was when the environment initiated
                    human_position = 3

                self.human_coordinates_list.append(human_position)

            elif (type(self.origin_human_coordinates)) is tuple:
                for human in range(len(self.origin_human_coordinates[0])):
                    relative_y = self.origin_human_coordinates[0][human] - self.now_human_coordinates[0][human]
                    relative_x = self.origin_human_coordinates[1][human] - self.now_human_coordinates[1][human]
                    if (relative_y == 0) & (relative_x == 0):
                        # Stay on the same cell when it was initiated
                        human_position = 0

                    elif (relative_y == 0) & (relative_x == 1):
                        # Stay on the left cell compared to the position it was when the environment initiated
                        human_position = 1

                    elif (relative_y == -1) & (relative_x == 1):
                        # Stay on the 'left diagonal'(South-West) cell compared to
                        # the position it was when the environment initiated
                        human_position = 2

                    elif (relative_y == -1) & (relative_x == 0):
                        # Stay on the below cell compared to the position it was when the environment initiated
                        human_position = 3

                    self.human_coordinates_list.append(human_position)

            return self.human_coordinates_list

        elif considering_object == 'parcel':
            if not self.pickup:
                return 0  # Does not pickup
            if self.pickup & (not self.dropoff):
                return 1  # Pick Up
            if self.dropoff:
                return 2  # Drop-off

    # Function to convert current situation of the environment to an unique state number
    def encode(self):
        # Get the position of each object
        agent_position = self.convert_to_position('agent')
        box_position = self.convert_to_position('box')
        human_position = self.convert_to_position('human')
        parcel_position = self.convert_to_position('parcel')

        # Initial state
        state = agent_position

        # The parcel
        state *= 3  # Possible states of the parcel
        state += parcel_position

        # The humans
        for human in range(len(human_position)):
            state *= 4  # Possible states of the humans
            state += human_position[human]

        # The box
        state *= (self.N ** 2)  # Possible states of the box
        state += box_position

        return state  # Unique state

    # Help function to calculate total possible states
    def total_possible_states(self):
        agent_state = self.N * self.N
        box_state = self.N * self.N
        human_state = 4 ** int(self.N / 10)
        parcel_state = 3

        return agent_state * box_state * human_state * parcel_state

    # Help function to calculate observations
    def calculate_observations(self):
        if self.pickup:
            to_parcel_coordinate = np.array([0, 0])
        else:
            to_parcel_coordinate = self.position_parcel - self.position_agent

        to_destination_coordinate = self.position_destination - self.position_agent

        # Padded the boundary
        warehouse_padded = np.ones((self.N + 2, self.N + 2), dtype=np.int8)
        warehouse_padded[1:self.N + 1, 1:self.N + 1] = self.warehouse[:, :]

        surroundings = warehouse_padded[
                       self.position_agent[0] + 1 - 1: self.position_agent[0] + 1 + 2,
                       self.position_agent[1] + 1 - 1: self.position_agent[1] + 1 + 2]

        surroundings[1, 1] = 6

        observations = {'to_parcel_coordinate': to_parcel_coordinate,
                        'to_destination_coordinate': to_destination_coordinate,
                        'surroundings': surroundings}

        return observations

    # Help function to reset the environment
    def reset(self):
        # Reset battery capacity
        self.battery_elapsed = 0

        # Get agent position
        self.position_agent = self.get_placing_coordinates('edge', 1)

        # Reset the placing parcel to the original position and pick up condition
        if self.pickup:
            self.warehouse[self.original_parcel_position[0], self.original_parcel_position[1]] = 4
            self.pickup = False

        # Reset drop off condition
        if self.dropoff:
            self.dropoff = False

        # Calculate observations
        # observations = self.calculate_observations()

        # Get current state
        state = self.encode()

        return state


# %% Check the environment

if __name__ == '__main__':
    env = Warehouse(10)
    env.reset()
    env.display()
    # s, r, d = env.step('up')
    # env.display()
