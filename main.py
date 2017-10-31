# -*- coding: utf-8 -*-
"""
Implementation of a recommendation system using techniques proposed in [1]
Training data from MovieLens 100M Dataset [2]

References:
[1] Hu, Yifan, Yehuda Koren, and Chris Volinsky. "Collaborative filtering for implicit feedback datasets." 
Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on. Ieee, 2008.
[2] https://grouplens.org/datasets/movielens/
"""

import numpy as np
import os

import data_input as di
import mat_fac_implicit as mfi
import recommendation as rec


def recommend_items(user_id, rec_item_num, include_seen, latent_feature_num, iterations, regularization):
	"""
	recommend K items for a user U based on past behavior of the given users toward the given items

	@param user_id {int}			: user id U (1-943)
	@param rec_item_num {int}		: # of recommendations for the user K
	@param include_seen {bool}		: if False, only unseen (unrated) items will be recommended
	@param latent_feature_num {int}	: # of latent factor
	@param iterations {int}			: # of iterations
	@param regularization {float}	: constant for regularization penalty
	"""

	# get rating data (user-item pairs)
	print '[data input]'
	all_dat = di.get_user_item_mat()

	# obtain user factor and item factor from files
	u_fac_path = 'tmp/u_fac.tmp'
	i_fac_path = 'tmp/i_fac.tmp'

	# if the files exist, use the old data
	if os.path.isfile(u_fac_path) and os.path.isfile(i_fac_path):
		print '[use previous user and item factors]'
		u_factor = di.get_mat(u_fac_path)
		i_factor = di.get_mat(i_fac_path)

	# if not, generate user factors and item factors
	else:
		print '[compute user and item factors]'
		[u_factor, i_factor] = mfi.factor_mat(all_dat, latent_feature_num, 
											  iterations, regularization)

	# sort the predicted preference values and select K recommended items
	print '[recommend %d items for user_id %d]' % (rec_item_num, user_id)
	recommendations = rec.recommend(rec_item_num, u_factor[user_id-1,:], i_factor, 
								   all_dat[user_id-1], include_seen)

	# return the items (id-title pairs)
	return recommendations



# usage example
user_id = 1
rec_item_num = 5
include_seen = False
latent_feature_num = 10
iterations = 20
regularization = 0.001

recommendations = recommend_items(user_id, rec_item_num, include_seen, 
								  latent_feature_num, iterations, regularization)
print recommendations






