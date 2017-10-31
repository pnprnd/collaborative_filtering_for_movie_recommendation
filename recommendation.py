# -*- coding: utf-8 -*-

import numpy as np
import data_input as di

def recommend(rec_item_num, u_factor, i_factors, input_ratings, include_seen):
	"""
	return a list of K recommended items

	@param rec_item_num {int}	: # of items to be selected K
	@param u_factor {1xF}		: user factor
	@param i_factors {FxN}		: item factor matrix
	@param input_ratings {1xN}	: rating data of the user
	@param include_seen {bool}	: False if recommend only unseen items (rating = 0)
	"""

	# compute the predicted preference
	preference = u_factor * i_factors

	# ignore the seen items
	if not include_seen:
		input_ratings = np.matrix(input_ratings) # logical indexing requires np.matrix
		preference[input_ratings != 0] = -1
	
	# get the indices of the top K items with highest preference
	rec_item_ids = np.argsort(preference)[0, -1:-rec_item_num-1:-1]

	# get the titles from the item indices
	item_titles = di.get_item_titles()

	# store the item id and the title of each selected item in a list
	recommendations = []
	for i in range(rec_item_num):
		recommendations.append([rec_item_ids[0, i]+1, item_titles[rec_item_ids[0, i]]])

	# return the list
	return recommendations

