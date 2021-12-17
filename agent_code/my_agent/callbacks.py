import os
import sys

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("..")

import pickle
import random
from DQN import DQN
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import time

# python main.py play --my-agent my_agent

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

	input_size = 4
	output_size = 8
	kernel_size = 1
	self.total_rewards = 0
	self.record = pd.DataFrame(columns=["round", "steps", "loss", "total_rewards"])

	# training parameters
	self.MIN_ENEMY_STEPS = 15000
	self.MEMORY_CAPACITY = 15000

	self.modelpath = os.path.join(os.getcwd(), "models", 'model.pt')
	self.qnn = DQN(input_size, output_size, kernel_size, self.MEMORY_CAPACITY, self.MIN_ENEMY_STEPS)

	# if self.train or not os.path.isfile(self.modelpath):
	if self.train:
		print("no model, setting up from scratch")
		self.logger.info("Setting up model from scratch.")

	else:
		self.logger.info("Loading model from saved state.")

		state_dict = torch.load((self.modelpath), map_location=lambda storage, loc: storage)
		self.qnn.eval_net.load_state_dict(state_dict)
		print("loaded model from saved state")


def act(self, game_state: dict) -> str:
	"""
	Your agent should parse the input, think, and take a decision.
	When not in training mode, the maximum execution time for this method is 0.5s.

	:param self: The same object that is passed to all of your callbacks.
	:param game_state: The dictionary that describes everything on the board.
	:return: The action to take as a string.
	"""

	self_mat, state_input = state_to_features(game_state)
	if self.train:
		self.logger.debug("Training action")
		choice = self.qnn.choose_action(self_mat, state_input, True)
	else:
		self.logger.debug("Querying model for action.")
		choice = self.qnn.choose_action(self_mat, state_input)

	return ACTIONS[choice[0]]


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def state_to_features(game_state: dict, is_enemy=False) -> np.array:
	"""
	Converts the game state to the input of your model, i.e.
	a feature vector.

	:param game_state:  A dictionary describing the current game board.
	:return: np.array
	"""
	# This is the dict before the game begins and after it ends
	# initial statues
	self_mat = np.ones((1, 6))

	self_field = np.zeros((1, 6)) # mark the barrier
	enemy_field = np.zeros((1,6)) # mark the enemy
	box_field = np.zeros((1, 6))   # mark the box
	bomb_field =np.zeros((1, 6))   # mark the bomb

	if game_state is not None:
		#gaussian filter
		w = np.array([0.41111229, 0.60653066, 0.8007374, 0.94595947, 1., 0.94595947, 0.8007374, 0.60653066, 0.41111229])
		u = w[:, np.newaxis]
		v = w[np.newaxis, :]

		#observe time
		if game_state['step'] == 0 or game_state['step'] ==1:
			self_mat[0,5] = 0
			self_mat[0,4] = 0
		self_loc = game_state['self'][-1]

	# location map
		game_field = game_state['field']

	#mark the bombs
		bombs = [xy[0] for xy in game_state['bombs']]
		for bomb in bombs:
			game_field[bomb] = -1
		others = [xy[-1] for xy in game_state['others']]
		for other in others:
			game_field[other] = -1

		bomb_field, not_wait = get_bomb_field(game_state['bombs'],self_loc,u,v)
		#run!
		if not_wait:
			self_mat[0,4] = 0
		# no bomb-ability
		if not game_state['self'][2]:
			self_mat[0, 5] = 0

	#mark barriers
		barriers = np.zeros((17, 17))
		barriers[game_field != 0] = 1

		#find nearst barrier
		self_field[0,0],self_field[0,1],self_field[0,2],self_field[0,3],wait = find_nearst(self_loc, barriers, 1)
		self_field_n = self_field / 16
		self_field_n[0,4] = wait
	#mark enemy
		enemy_field[0,0],enemy_field[0,1],enemy_field[0,2],enemy_field[0,3] = find_plus(others,self_loc)
		if enemy_field[0,0] ==1 or enemy_field[0,2] == 1 or enemy_field[0,1] == 1 or enemy_field[0,3]==1:
			enemy_field[0,5] = 1
	#mark coins and boxes
		self_loc_lr = self_loc[0]
		self_loc_ud = self_loc[1]

		loc_r = (self_loc_lr + 1, self_loc_ud)
		loc_l = (self_loc_lr - 1, self_loc_ud)
		loc_u = (self_loc_lr, self_loc_ud - 1)
		loc_d = (self_loc_lr, self_loc_ud + 1)

		scope = 4
		box_map =np.zeros((17,17))
		box_map[game_field ==1] = 1
		for coin in game_state['coins']:
			# box_map[coin] = 2
			box_map[coin] = 3


		cut_self_field = cut_field(self_loc, scope, box_map, 0)
		box_field[0,4] = convolve_fast(cut_self_field,u,v)
		box_field[0,5] = convolve_fast(cut_self_field,u,v)

		cut_up_field = cut_field(loc_u, scope, box_map, 0)
		box_field[0,0] = convolve_fast(cut_up_field,u,v)

		cut_right_field = cut_field(loc_r, scope, box_map, 0)
		box_field[0,1] = convolve_fast(cut_right_field,u,v)

		cut_left_field = cut_field(loc_l, scope, box_map, 0)
		box_field[0,3] = convolve_fast(cut_left_field,u,v)

		cut_down_field = cut_field(loc_d, scope, box_map, 0)
		box_field[0,2] = convolve_fast(cut_down_field,u,v)

		if game_field[loc_r] != 0:
			self_mat[0,1] = 0
		if game_field[loc_l] != 0:
			self_mat[0,3] = 0
		if game_field[loc_u] != 0:
			self_mat[0,0] = 0
		if game_field[loc_d] != 0:
			self_mat[0,2] = 0

	enemy_field_n = normalization(enemy_field)
	box_field_n = normalization(box_field)
	# print("once")
	# print(self_field_n)
	# print(bomb_field_n)
	# print(bomb_lev)
	# print(plus_field_n)
	# print(enemy_field_n)
	# print(box_field_n)

	channels = []
	channels.append(self_field)
	channels.append(bomb_field)
	channels.append(enemy_field_n)
	channels.append(box_field_n)

	# concatenate them as a feature tensor (they must have the same shape), ...
	stacked_channels = np.stack(channels)

	return self_mat, stacked_channels

# def get_bomb_field(bombs,loc, u,v ,scope = 3):
def get_bomb_field(bombs,loc, u,v ,scope = 4):
	bomb_map = np.zeros((17,17))
	bomb_field = np.zeros((1,6))
	bomb_level = 0

	for bomb in bombs:
		bomb_loc = bomb[0]
		bomb_step = bomb[1]
		if bomb_step == 3:
			bomb_level = 1
		if bomb_step == 2:
			bomb_level = 3
		if bomb_step ==1:
			bomb_level = 5

		i_l =  max(0,bomb_loc[0] - 3)
		i_r = min(16, bomb_loc[0] + 3)
		i_u = max(0, bomb_loc[1] - 3)
		i_d = min(16, bomb_loc[1] + 3)

		lr = i_l
		ud = i_u
		while lr <= i_r:
			bomb_map[lr, bomb_loc[1]] = bomb_level
			lr = lr+1
		while ud <= i_d:
			bomb_map[bomb_loc[0],ud] = bomb_level
			ud = ud + 1
	loc_r = (loc[0] + 1, loc[1])
	loc_l = (loc[0] - 1, loc[1])
	loc_u = (loc[0], loc[1] - 1)
	loc_d = (loc[0], loc[1] + 1)
	cut_self_field = cut_field(loc, scope, bomb_map, 0)
	bomb_field[0,4] = convolve_fast(cut_self_field,u,v)

	cut_up_field = cut_field(loc_u, scope, bomb_map, 0)
	bomb_field[0,0] = convolve_fast(cut_up_field,u,v)

	cut_right_field = cut_field(loc_r, scope, bomb_map, 0)
	bomb_field[0,1] = convolve_fast(cut_right_field,u,v)

	cut_left_field = cut_field(loc_l, scope, bomb_map, 0)
	bomb_field[0,3] = convolve_fast(cut_left_field,u,v)

	cut_down_field = cut_field(loc_d, scope, bomb_map, 0)
	bomb_field[0,2] = convolve_fast(cut_down_field,u,v)

	bomb_field_n = normalization(bomb_field)
	not_wait = False
	if bomb_map[loc] == 5:
		not_wait = True
	# 	idx_l = 0
	# 	idx_r = 0
	# 	idx_u = 0
	# 	idx_d = 0
	# 	if bomb_map[loc[0], loc[1]]!= 0:
	# 		bomb_lev_field[0,0] = bomb_map[loc[0], loc[1]-1] #up
	# 		bomb_lev_field[0,1] = bomb_map[loc[0]+1, loc[1]] #right
	# 		bomb_lev_field[0,2] = bomb_map[loc[0], loc[1]+1] #down
	# 		bomb_lev_field[0,3] = bomb_map[loc[0]-1, loc[1]] #left
	# 		bomb_lev_field[0,4] = bomb_map[loc[0],loc[1]] #current
	# 		for i in range(5):
	# 			if bomb_lev_field[0,i]!=0:
	# 				bomb_field[0,i] = 0
	# 			else:
	# 				bomb_field [0,i] = 1
	# 	else:
	# 		while loc[0] - idx_l > 0 and bomb_map[loc[0]-idx_l, loc[1]]==0:
	# 			if idx_l > 3:
	# 				break
	# 			bomb_lev_field[0, 3] = bomb_map[loc[0] - idx_l, loc[1]]
	# 			idx_l += 1
	#
	# 		while loc[0] + idx_r <16 and bomb_map[loc[0]+idx_r, loc[1]]==0:
	# 			if idx_r > 3:
	# 				break
	# 			bomb_lev_field[0, 1] = bomb_map[loc[0]+idx_r, loc[1]]
	# 			idx_r +=1
	#
	# 		while loc[1] - idx_u >0 and bomb_map[loc[0], loc[1]-idx_u]==0:
	# 			if idx_u > 3:
	# 				break
	# 			bomb_lev_field[0, 0] = bomb_map[loc[0], loc[1]-idx_u]
	# 			idx_u +=1
	#
	# 		while loc[1] + idx_d <16 and bomb_map[loc[0],loc[1]+idx_d]==0:
	# 			if idx_d > 3:
	# 				break
	# 			bomb_lev_field[0, 2] = bomb_map[loc[0],loc[1]+idx_d]
	# 			idx_d +=1
	#
	#
	# 	bomb_field[0,0] = (16 - idx_u)/16
	# 	bomb_field[0,1] = (16 - idx_r)/16
	# 	bomb_field[0,2] = (16 - idx_d)/16
	# 	bomb_field[0,3] = (16 - idx_l)/16
	#
	# 	bomb_lev_field[0,0] = bomb_map[loc[0], loc[1]-idx_u]
	# 	bomb_lev_field[0,1] = bomb_map[loc[0]+idx_r, loc[1]]
	# 	bomb_lev_field[0,2] = bomb_map[loc[0],loc[1]+idx_d]
	# 	bomb_lev_field[0,3] = bomb_map[loc[0]-idx_l, loc[1]]
	# 	bomb_lev_field[0,4] = bomb_map[loc[0], loc[1]]
	# 	if bomb_lev_field[0,4] != 0:
	# 		bomb_field[0,4] = 0
	# print(bomb_field)
	# print(bomb_map)
	# print(bomb_lev_field)
	return bomb_field_n,not_wait

def find_nearst(loc,field, threshold):
	idx_r = 0
	idx_l = 0
	idx_u = 0
	idx_d = 0

	if(field[loc]!=0):
		wait = 0
		field[loc] = 0
	else: wait = 1
	#right for barriers threshold = -1
	while field[loc[0]+idx_r,loc[1]] != threshold:
		idx_r += 1
	while field[loc[0] - idx_l,loc[1]] != threshold:
		idx_l += 1

	#up
	while field[loc[0] , loc[1] - idx_u] != threshold:
		idx_u += 1
	while field[loc[0] , loc[1] + idx_d] != threshold:
		idx_d += 1
	return idx_u - 1, idx_r-1, idx_d-1 , idx_l-1,wait

def find_plus(object_list, loc):

	idx_r = 16
	idx_l = 16
	idx_u = 16
	idx_d = 16
	if len(object_list) != 0:
		for obj in object_list:

			diff_lr = obj[0] - loc[0]
			if diff_lr < 0 and -diff_lr < idx_l:
				idx_l = - diff_lr

			if diff_lr > 0 and diff_lr < idx_r:
				idx_r = diff_lr

			diff_ud = obj[1] - loc[1]
			if diff_ud < 0 and -diff_ud < idx_u:
				idx_u = -diff_ud

			if diff_ud > 0 and diff_ud < idx_d:
				idx_d = diff_ud

	return idx_u, idx_r,idx_d,idx_l


def cut_field(self_loc, scope, field, padding_c):
	# left-right
	padding_left = 0
	padding_right = 0
	padding_top = 0
	padding_down = 0

	if self_loc[1] - scope < 0:
		padding_left = scope - self_loc[1]
		idx_left = 0
		idx_right = self_loc[1] + scope + 1
	elif self_loc[1] + scope > 16:
		padding_right = self_loc[1] + scope - 16
		idx_right = 17
		idx_left = self_loc[1] - scope
	else:
		idx_left = self_loc[1] - scope
		idx_right = self_loc[1] + scope + 1

	if self_loc[0] - scope < 0:
		padding_top = scope - self_loc[0]
		idx_down = self_loc[0] + scope + 1
		idx_top = 0
	elif self_loc[0] + scope > 16:
		padding_down = self_loc[0] + scope - 16
		idx_top = self_loc[0] - scope
		idx_down = 17
	else:
		idx_top = self_loc[0] - scope
		idx_down = self_loc[0] + scope + 1

	loc_game = field[:, idx_left:idx_right]
	loc_game = loc_game[idx_top:idx_down, :]
	final_field = np.pad(loc_game, ((padding_top, padding_down), (padding_left, padding_right)), 'constant',
	                     constant_values=padding_c)
	return final_field


def convolve_fast(arr, K_u, K_v):
	'''
	Convolve the array using kernel K_u and K_v.

	arr -- 2D array that gets convolved
	K_u -- kernel u
	K_v -- kernel v
	'''
	arr_pad = np.pad(arr, (int((K_u.shape[0] - 1) / 2), int((K_v.shape[1] - 1) / 2)))

	# flip the kern
	u_flip = np.flip(np.flip(K_u, 0), 1)
	v_flip = np.flip(np.flip(K_v, 0), 1)

	i = int(arr.shape[0] / 2)
	j = int(arr.shape[0] / 2)
	arr_conv = v_flip.dot(arr_pad[i: i + u_flip.shape[0], j: j + v_flip.shape[1]].dot(u_flip))

	return arr_conv[0][0]

def normalization(x):

	if x.max()!= x.min():
		res = (x-x.min())/(x.max()-x.min())
		return res
	elif x.max()!=0:
		return x/x.max()
	else:
		return x


# def state_to_features(game_state: dict, is_enemy=False) -> np.array:
# 	"""
# 	Converts the game state to the input of your model, i.e.
# 	a feature vector.
#
# 	:param game_state:  A dictionary describing the current game board.
# 	:return: np.array
# 	"""
# 	# This is the dict before the game begins and after it ends
# 	# initial statues
# 	self_field = np.zeros((1, 6))
# 	self_mat = np.ones((1, 6))
# 	plus_field = np.zeros((1, 6))
# 	ex_field = np.zeros((1, 6))
#
# 	# w = np.array([0.011109, 0.13533528, 0.60653066, 1., 0.60653066, 0.135335280, .011109])
# 	# w = np.array([0.7548396, 0.8824969, 0.96923323, 1., 0.96923323, 0.8824969, 0.7548396])
# 	w = np.array([0.72614904, 0.83527021, 0.92311635, 0.98019867, 1., 0.98019867,0.92311635, 0.83527021, 0.72614904])
# 	u = w[:, np.newaxis]
# 	v = w[np.newaxis, :]
#
# 	if game_state is not None:
# 		# location map
# 		game_field = game_state['field']
# 		explosion_field = game_state['explosion_map']
#
# 		barriers = np.zeros((17, 17))
# 		#mark barrier
# 		barriers[game_field != 0] = 1
#
# 		#mark the bombs
# 		bombs = [xy[0] for xy in game_state['bombs']]
# 		for bomb in bombs:
# 			game_field[bomb] = -1
# 			explosion_field[bomb] = 4
# 		# explosion_field[game_field != 0] = -1
#
# 		# mark all the crates and coins
# 		coins_field = np.zeros((17, 17))
# 		coins_field[game_field == 1] = 3
# 		for coin in game_state['coins']:
# 			coins_field[coin] = 6
#
# 		#mark enemy
# 		others = [xy[-1] for xy in game_state['others']]
# 		for other in others:
# 			game_field[other] = 1
#
#
#
# 		self_loc = game_state['self'][-1]
# 		# print("self location {}".format(self_loc))
# 		self_loc_lr = self_loc[0]
# 		self_loc_ud = self_loc[1]
#
# 		scope = 4 #how far the agent can see
#
# 		loc_r = (self_loc_lr + 1, self_loc_ud)
# 		loc_l = (self_loc_lr - 1, self_loc_ud)
# 		loc_u = (self_loc_lr, self_loc_ud - 1)
# 		loc_d = (self_loc_lr, self_loc_ud + 1)
# 		#nearst wall:
# 		self_field[0], self_field[1],self_field[2],self_field[3] = find_nearst(self_loc,)
# 		#env self
# 		cut_self_field = cut_field(self_loc,scope,barriers,1)
# 		self_env = convolve_fast(cut_self_field,u,v)
# 		self_field[0,5] = self_env
#
# 		cut_self_plus = cut_field(loc_r,scope,coins_field, 0)
# 		# plus_field[0,4] = convolve_fast(cut_self_plus,u,v)
# 		plus_field[0,5] = convolve_fast(cut_self_plus,u,v)
#
# 		cut_self_ex = cut_field(self_loc,scope,explosion_field,0)
# 		ex_field[0,4] = convolve_fast(cut_self_ex,u,v)
#
# 		#env right
# 		if game_field[loc_r] != 0 or ((self_loc_lr + 2, self_loc_ud) in bombs):
# 			self_mat[0, 1] = 0  # not right
# 			self_field[0,1] = 65 #blocked
# 		else:
# 			cut_right_field = cut_field(loc_r,scope,barriers,1)
# 			self_field[0,1] = convolve_fast(cut_right_field,u,v)
#
# 			cut_right_plus = cut_field(loc_r, scope, coins_field, 0)
# 			# run with the gaussian filter -- the far features have less influence
# 			plus_field[0, 1] = convolve_fast(cut_right_plus, u, v)
#
# 			cut_right_ex = cut_field(loc_r, scope, explosion_field, 0)
# 			ex_field[0, 1] = convolve_fast(cut_right_ex, u, v)
#
# 		#env left
# 		if int(game_field[loc_l])!= 0 or ((self_loc_lr-2,self_loc_ud) in bombs):
# 			self_mat[0, 3] = 0  # not left
# 			self_field[0,3] = 65
# 		else:
# 			cut_left_field = cut_field(loc_l, scope, barriers, 1)
# 			self_field[0, 3] = convolve_fast(cut_left_field, u, v)
#
# 			cut_left_plus = cut_field(loc_l, scope, coins_field, 0)
# 			plus_field[0, 3] = convolve_fast(cut_left_plus, u, v)
#
# 			cut_left_ex = cut_field(loc_l, scope, explosion_field, 0)
# 			ex_field[0, 3] = convolve_fast(cut_left_ex, u, v)
#
# 		#env down
# 		if game_field[loc_d] != 0 or ((self_loc_lr,self_loc_ud +2) in bombs):
# 			self_mat[0, 2] = 0  # not down
# 			self_field[0,2] = 65
#
# 		else:
# 			cut_down_field = cut_field(loc_d, scope, barriers, 1)
# 			self_field[0, 2] = convolve_fast(cut_down_field, u, v)
#
# 			cut_down_plus = cut_field(loc_d, scope, coins_field, 0)
# 			plus_field[0, 2] = convolve_fast(cut_down_plus, u, v)
#
# 			cut_down_ex = cut_field(loc_d, scope, explosion_field, 0)
# 			ex_field[0, 2] = convolve_fast(cut_down_ex, u, v)
#
# 		#env up
# 		if game_field[loc_u] != 0 or ((self_loc_lr,self_loc_ud - 2) in bombs):
# 			self_mat[0, 0] = 0  # not up
# 			self_field[0,0] = 65
# 		else:
# 			cut_up_field = cut_field(loc_u, scope, barriers, 1)
# 			self_field[0, 0] = convolve_fast(cut_up_field, u, v)
#
# 			cut_up_plus = cut_field(loc_u, scope, coins_field, 0)
# 			plus_field[0, 0] = convolve_fast(cut_up_plus, u, v)
#
# 			cut_up_ex = cut_field(loc_u, scope, explosion_field, 0)
# 			ex_field[0, 0] = convolve_fast(cut_up_ex, u, v)
#
# 		#not wait near bomb
# 		if (explosion_field[loc_r] == 4) or\
# 				(explosion_field[loc_l] == 4) or\
# 				(explosion_field[loc_d] == 4) or\
# 				(explosion_field[loc_u] == 4 )or (explosion_field[self_loc]!= 0):
# 			self_mat[0, 4] = 0
#
# 		if not game_state['self'][2]:
# 			self_mat[0, 5] = 0  # no bombs
#
# 		# give some indicate for boxes but still the robot has to learn that this behavior has more rewards
# 		# elif game_field[self_loc_lr, self_loc_ud - 1] == 1 or game_field[self_loc_lr, self_loc_ud + 1] == 1 or \
# 		# 		game_field[self_loc_lr - 1, self_loc_ud] == 1 or game_field[loc_r] == 1 :
# 		# 	self_field[0, 5] = 2
# 	# print("new")
# 	# print(self_field)
# 	# print(ex_field)
# 	# print(plus_field)
# 	self_field_n = normalization(self_field)
# 	# For example, you could construct several channels of equal shape, ...
# 	channels = []
# 	channels.append(self_field_n)
# 	channels.append(plus_field)
# 	channels.append(ex_field)
# 	# concatenate them as a feature tensor (they must have the same shape), ...
# 	stacked_channels = np.stack(channels)
#
# 	return self_mat, stacked_channels
