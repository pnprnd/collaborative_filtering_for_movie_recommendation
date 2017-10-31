# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def get_user_item_mat():
	"""
	generate a user-item pair matrix from ml-100k/u.data
	"""

	# read data from u.data
	col_title = ['user_id', 'item_id', 'rating', 'timestamp']
	raw_dat = pd.read_csv('ml-100k/u.data', sep='\t', names=col_title, encoding='latin-1')
	dat_num = raw_dat.shape[0]

	# # of users and # of items
	u_num = 943
	i_num = 1682

	# buffer for the user-item pair matrix (rating matrix)
	ratings = np.zeros((u_num, i_num))

	# buffer for the most recent time when an item is rated by a user
	times = np.zeros((u_num, i_num))

	# iterate over each rating record
	for row in range(dat_num):

		# obtain the user index, item index, rating and unix timestamp
		u_idx = raw_dat['user_id'][row]-1
		i_idx = raw_dat['item_id'][row]-1
		rating = raw_dat['rating'][row]
		time = raw_dat['timestamp'][row]

		# store only most recent data
		if ratings[u_idx, i_idx] != 0 and times[u_idx, i_idx] > time:
			continue
		ratings[u_idx, i_idx] = rating
		times[u_idx, i_idx] = time

	# return the generated rating matrix
	return ratings


def get_mat(path):
	"""
	obtain a matrix from a file
	"""
	mat = pd.read_csv(path, sep='\t', header=None, encoding='utf-8')
	return np.matrix(mat.values)


def get_item_titles():
	"""
	obtain all the item titles
	"""
	mat = pd.read_csv('ml-100k/u.item', sep='|', header=None, encoding='latin-1')
	return [item.encode('utf-8') for item in mat.values[:,1]]

