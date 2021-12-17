#for training purposes.

import numpy as np


def setup(self):
    np.random.seed()


def act(agent, game_state: dict):
    agent.logger.info('Pick action at random')

    log = open('training_data.txt', 'a')
    log.write(f'''
                {game_state['round']};
                {game_state['step']};
                {game_state['field']};
                {game_state['bombs']};
                {game_state['explosion_map']};
                {game_state['coins']};
                {game_state['self']};
                {game_state['others']};
            ''')
    return np.random.choice(['RIGHT', 'LEFT', 'UP', 'DOWN'], p=[.25, .25, .25, .25])
