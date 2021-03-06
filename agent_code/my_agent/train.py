import pickle
import random
import os
import numpy as np
from collections import namedtuple, deque
from typing import List
import pandas as pd
from DQN import DQN
import torch
import torch.nn as nn
import time

import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Hyper parameters
RECORD_RANDOM_TRANSITIONS = 0.2
RECORD_ENEMY_TRANSITIONS = 0.3  # record enemy transitions with probability ...
TRAINING_ROUNDS = 10000

# python main.py play --agents my_agent rule_based_agent random_agent rule_based_agent --train 1 --no-gui --n-rounds 500


# Events
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup_training(self):
	"""
	Initialise self for training purpose.

	This is called after `setup` in callbacks.py.

	:param self: This object is passed to all callbacks and you can set arbitrary values.
	"""

	print("Set up Training")


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
	"""
	Called once per step to allow intermediate rewards based on game events.

	When this method is called, self.events will contain a list of all game
	events relevant to your agent that occurred during the previous step. Consult
	settings.py to see what events are tracked. You can hand out rewards to your
	agent based on these events and your knowledge of the (new) game state.

	This is *one* of the places where you could update your agent.

	:param self: This object is passed to all callbacks and you can set arbitrary values.
	:param old_game_state: The state that was passed to the last call of `act`.
	:param self_action: The action that you took.
	:param new_game_state: The state the agent is in now.
	:param events: The events that occurred when going from  `old_game_state` to `new_game_state`
	"""
	self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
	if self.qnn.memory_counter == self.MIN_ENEMY_STEPS:
		self.logger.debug(f'Saved enough information from enemies, start to try the own policy')
		print("Enough learning, I will try on my own now!")


	if old_game_state != None:

		if(self.qnn.memory_counter < self.MEMORY_CAPACITY):
			if np.random.uniform() < RECORD_RANDOM_TRANSITIONS:
				rewards = reward_from_events(self, events)
				self.total_rewards += rewards
				self_mat, old_state = state_to_features(old_game_state)

				_, new_state = state_to_features(new_game_state)
				self.qnn.store_transition(old_state,
				                          np.array([ACTIONS.index(self_action)]),
				                          rewards,
				                          new_state)

		if self.qnn.memory_counter > self.MEMORY_CAPACITY:

			rewards = reward_from_events(self, events)
			self.total_rewards += rewards

			self_mat, old_state = state_to_features(old_game_state)
			# print(self_mat)
			# print(self_action)
			# print(old_game_state['self'][-1])
			# print(new_game_state['self'][-1])
			#

			print("Action {} Added {} to total reward".format(self_action,rewards))


			_, new_state = state_to_features(new_game_state)
			self.qnn.store_transition(old_state,
			                          np.array([ACTIONS.index(self_action)]),
			                          rewards,
			                          new_state)
			self.qnn.learn()



def enemy_game_events_occurred(self, enemy_name: str, old_enemy_game_state: dict, enemy_action: str,
                               enemy_game_state: dict, enemy_events: List[str]):

	if enemy_name != 'random_agent' and self.qnn.memory_counter <= self.MIN_ENEMY_STEPS:
		if old_enemy_game_state != None and enemy_action != None:
			if self.qnn.memory_counter % 500 == 0:
				print("Saved {}th event from enemy {} ".format(self.qnn.memory_counter, enemy_name))

			rewards = reward_from_events(self, enemy_events)
			_, old_state = state_to_features(old_enemy_game_state)
			_, new_state = state_to_features(enemy_game_state)
			self.qnn.store_transition(old_state,
			                          np.array([ACTIONS.index(enemy_action)]),
			                          rewards,
			                          new_state)

			if self.qnn.memory_counter % 500 == 0:
				print("reached {} steps".format(self.qnn.memory_counter))
			if self.qnn.memory_counter == self.MEMORY_CAPACITY:
				print("QNN Start Learning!!!")

			if self.qnn.memory_counter > self.MEMORY_CAPACITY:
				self.qnn.learn()
	else:
		# only record a part of enemy transition randomly after reached min. enemey steps
		if np.random.uniform() < RECORD_ENEMY_TRANSITIONS:
			if old_enemy_game_state != None and enemy_action != None:
				_, old_state = state_to_features(old_enemy_game_state)
				_, new_state = state_to_features(enemy_game_state)
				rewards = reward_from_events(self, enemy_events)

				self.qnn.store_transition(old_state,
				                          np.array([ACTIONS.index(enemy_action)]),
				                          rewards,
				                          new_state)

				if self.qnn.memory_counter > self.MEMORY_CAPACITY:
					self.qnn.learn()


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
	"""
	Called at the end of each game or when the agent died to hand out final rewards.

	This is similar to reward_update. self.events will contain all events that
	occurred during your agent's final step.

	This is *one* of the places where you could update your agent.
	This is also a good place to store an agent that you updated.

	:param self: The same object that is passed to all of your callbacks.
	"""
	self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
	# self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))
	# Store the model
	print("Finished {}th round with {} steps".format(last_game_state['round'], last_game_state['step']))
	local_reward = 0
	if self.qnn.memory_counter > self.MIN_ENEMY_STEPS:
		print(events)

		if last_game_state != None and last_action != None:
			rewards = reward_from_events(self, events)
			self.total_rewards += rewards
			local_reward = self.total_rewards
			print("end state with reward {}".format(rewards))
			print("Finished the round with total rewards {}".format(self.total_rewards))

			_, old_state = state_to_features(last_game_state)
			_, new_state = state_to_features(None)
			self.qnn.store_transition(old_state,
			                          np.array([ACTIONS.index(last_action)]),
			                          reward_from_events(self, events),
			                          new_state)

			if self.qnn.memory_counter == self.MEMORY_CAPACITY:
				print("QNN Start Learning!!!")

			if self.qnn.memory_counter >= self.MEMORY_CAPACITY:
				self.qnn.learn()

			self.record = self.record.append(pd.DataFrame({'round': [int(last_game_state['round'])],
			                                               'step': [int(last_game_state['step'])],
			                                               'loss': [self.qnn.loss],
			                                               'total_rewards': [self.total_rewards]}),
			                                 ignore_index=True)
			self.logger.debug("Model has loss {}".format(self.qnn.loss))

			if (last_game_state['round'] == TRAINING_ROUNDS) or (last_game_state['round'] == int(TRAINING_ROUNDS/2)) or (last_game_state['round'] == int(TRAINING_ROUNDS/4)):
				timestamp = time.strftime("%d_%H_%M_%S", time.localtime())
				recordPath = os.path.join(os.getcwd(), "logs", '{}.csv'.format(timestamp))
				self.record.to_csv(recordPath, index=False)
				print("Saved record in {}".format(recordPath))

		self.total_rewards = 0

	if e.SURVIVED_ROUND in events or last_game_state['round'] % 1000==0 :
		model_num = "model_{}th_round_survive.pt".format(last_game_state['round'])
		save_model_path = os.path.join(os.getcwd(), "models", model_num)
		torch.save(self.qnn.eval_net.state_dict(), save_model_path)
		print("Saved Model with loss {} who survived the round".format(self.qnn.loss))
		self.logger.debug("Saved Model with loss {}".format(self.qnn.loss))


	if self.qnn.memory_counter > self.MEMORY_CAPACITY and\
			((last_game_state['step'] > 50 and local_reward > 0) or
			 last_game_state['step'] > 300):
		model_num = "model_{}th_round.pt".format(last_game_state['round'])
		save_model_path = os.path.join(os.getcwd(), "models", model_num)
		torch.save(self.qnn.eval_net.state_dict(), save_model_path)
		print("Saved Model with loss {}".format(self.qnn.loss))
		self.logger.debug("Saved Model with loss {}".format(self.qnn.loss))

def reward_from_events(self, events: List[str]) -> int:
	"""
	*This is not a required function, but an idea to structure your code.*

	Here you can modify the rewards your agent get so as to en/discourage
	certain behavior.
	"""
	game_rewards = {
		e.BOMB_DROPPED: 0.5,
		e.COIN_COLLECTED: 0.7,

		# e.KILLED_OPPONENT: 5,
		e.INVALID_ACTION: - 0.5,
		e.SURVIVED_ROUND: 10,
		# e.OPPONENT_ELIMINATED: 8,

		e.MOVED_LEFT: - 0.35,
		e.MOVED_RIGHT: - 0.35,
		e.MOVED_UP: - 0.35,
		e.MOVED_DOWN: - 0.35,
		e.WAITED: - 0.4
	}
	reward_sum = 0
	# if (e.MOVED_LEFT in events or e.MOVED_RIGHT in events or e.MOVED_UP in events or e.MOVED_DOWN in events or e.WAITED in events) and e.BOMB_EXPLODED in events:
	if e.BOMB_EXPLODED in events:
		if e.GOT_KILLED not in events and e.KILLED_SELF not in events:
			reward_sum += 0.9
			if  e.CRATE_DESTROYED in events:
				reward_sum += 0.5
			if e.KILLED_OPPONENT in events:
				reward_sum += 0.5
	if e.GOT_KILLED in events or e.KILLED_SELF in events:
		reward_sum -= 0.8
	if e.WAITED and e.GOT_KILLED in events:
		reward_sum -= 0.8

	for event in events:
		if event in game_rewards:
			reward_sum += game_rewards[event]
		self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")

	return reward_sum

# def reward_from_events(self, events: List[str]) -> int:
# 	"""
# 	*This is not a required function, but an idea to structure your code.*
#
# 	Here you can modify the rewards your agent get so as to en/discourage
# 	certain behavior.
# 	"""
# 	game_rewards = {
# 		e.CRATE_DESTROYED: 1,
# 		e.COIN_COLLECTED: 50,
# 		e.COIN_FOUND: 2,
#
# 		e.KILLED_OPPONENT: 10,
# 		e.KILLED_SELF: - 100,
# 		e.GOT_KILLED: - 100,
# 		e.INVALID_ACTION: - 5,
# 		e.SURVIVED_ROUND: 500,
# 		e.OPPONENT_ELIMINATED: 50,
#
# 		e.MOVED_LEFT: - 0.001,
# 		e.MOVED_RIGHT: - 0.001,
# 		e.MOVED_UP: - 0.001,
# 		e.MOVED_DOWN: - 0.001,
# 		e.WAITED: - 0.5
# 	}
# 	reward_sum = 0
# 	if (e.MOVED_LEFT or e.MOVED_RIGHT or e.MOVED_UP or e.MOVED_DOWN or e.WAITED) in events and e.BOMB_EXPLODED in events:
# 		if (e.GOT_KILLED and e.KILLED_SELF) not in events:
# 			reward_sum += 75
#
# 	if (e.INVALID_ACTION or e.WAITED) in events and (e.GOT_KILLED or e.KILLED_SELF) in events:
# 		reward_sum -= 75
#
# 	if e.CRATE_DESTROYED in events and e.KILLED_SELF in events:
# 		reward_sum -= 50
#
# 	for event in events:
# 		if event in game_rewards:
# 			reward_sum += game_rewards[event]
# 		self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
#
# 	return reward_sum

