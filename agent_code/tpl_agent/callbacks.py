import os
import pickle
import random
import copy

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    return np.random.choice(ACTIONS, p=self.model)


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    round_ = game_state['round']
    step = game_state['step']
    field = game_state['field']
    bombs = [g[0] for g in game_state['bombs']]
    bombs_timers = [g[1] for g in game_state['bombs']]
    explosion_map = game_state['explosion_map']
    coins = game_state['coins']
    others = game_state['others']
    n, s, b, (x,y) = game_state['self']

    # ! determine nearest wall, crate, coin, bomb in any direction (99 if no bomb/coin/crate)
    env = [None, None, None, None]
    env[0] = find_items((x,y), 'UP', field, bombs, coins, explosion_map)
    env[1] = find_items((x,y), 'RIGHT', field, bombs, coins, explosion_map)
    env[2] = find_items((x,y), 'DOWN', field, bombs, coins, explosion_map)
    env[3] = find_items((x,y), 'LEFT', field, bombs, coins, explosion_map)
    
    return env

def find_items(coord:(int, int), direction: str, field, bombs, coins, explosion_map, step_limit = 99):
    if direction == 'UP' or direction == 'DOWN':
        temp = copy.copy(coord[1])
        t_coord = (coord[0], temp)
    else:
        temp = copy.copy(coord[0])
        t_coord = (temp, coord[1])

    if direction == 'LEFT' or direction == 'DOWN':
        adjust = lambda x: x - 1
    else:
        adjust = lambda x: x + 1

    nearest_walls = 99
    nearest_crates = 99
    nearest_coins = 99
    nearest_bombs = 99
    nearest_explosions = 99

    found_bomb = False
    found_coin = False
    found_explosion = False

    steps = 0
    while temp != 0 and temp != 16 and field[t_coord] == 0 and steps < step_limit:
        steps += 1
        temp = adjust(temp)
        if direction == "LEFT" or direction == "RIGHT":
            t_coord = (adjust(t_coord[0]), t_coord[1])
        else:
            t_coord = (t_coord[0], adjust(t_coord[1]))

        if found_coin == False and t_coord in coins: 
            found_coin = True 
            nearest_coins = abs(t_coord[0] - coord[0] + t_coord[1] - coord[1])

        if found_bomb == False and t_coord in bombs: 
            found_bomb = True
            nearest_bombs = abs(t_coord[0] - coord[0] + t_coord[1] - coord[1])
            
        if found_explosion == False and explosion_map[t_coord] != 0: 
            found_explosion= True
            nearest_explosions = abs(t_coord[0] - coord[0] + t_coord[1] - coord[1])*explosion_map[t_coord]

        if field[t_coord] == 1:
            nearest_crates = abs(t_coord[0] - coord[0] + t_coord[1] - coord[1])
        elif field[t_coord] == -1:
            nearest_walls = abs(t_coord[0] - coord[0] + t_coord[1] - coord[1])
    
    return nearest_walls, nearest_crates, nearest_coins, nearest_bombs, nearest_explosions





def test_functions():    
    test_state = {
                'round':1, 
                'step':1, 
                'field':np.array([
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [-1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 , 0, -1],
                    [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
                    [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
                    [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
                    [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
                    [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
                    [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
                    [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
                    [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
                    [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
                    [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
                    [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
                    [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
                    [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
                    [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
                ]),
                'bombs':[((12,3), 2), ((14,7), 2), ((11,5),2)],
                'explosion_map':np.array([
                    [0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0. ,0.],
                    [0. ,0. ,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0. ,0. ,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0. ,0. ,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0. ,0. ,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0. ,0. ,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0. ,0. ,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0. ,0. ,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0. ,0. ,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0. ,0. ,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0. ,0. ,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0. ,0. ,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0. ,0. ,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0. ,0. ,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0. ,0. ,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0. ,0. ,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0. ,0. ,0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
                ]),
                'coins':[(5, 1), (3, 10), (5, 14), (9, 1), (9, 7), (10, 11), (11, 3), (13, 9), (13, 14)],
                'self':('haris_agent', 0, True, (11, 7)),
                'others':[('random_agent_0', 0, True, (1, 15)), ('random_agent_1', 0, True, (15, 15)), ('random_agent_2', 0, True, (1, 1))]
    }

    print(state_to_features(test_state))

    
test_functions()