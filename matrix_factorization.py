# -*- coding: utf-8 -*-
"""
Implementation of matrix factorization using Gradient Boosting [1]

Reference:
[1] Nguyen, Jennifer, and Mu Zhu. "Content‐boosted matrix factorization techniques for 
recommender systems." Statistical Analysis and Data Mining: The ASA Data Science 
Journal 6.4 (2013): 286-301.
"""

import numpy as np
import pandas as pd

def factor_mat(all_dat, f_num, max_itr, learn_rate, regularization):
	"""
	estimate an MxF user factor and an FxN iten factor from the MxN rating matrix

	@param all_dat {MxN}			: rating matrix (row: user, column: item)
	@param f_num {int}				: # of latent features F
	@param max_itr {int}			: iteration limitation
	@param learn_rate {double}		: learning rate
	@param regularization {double}	: constant for regularization penalty (to avoid overfitting)
	"""

	# get # of users and # of items
	[u_num, i_num] = all_dat.shape

	# init user factor and item factor with random values
	u_fac = np.random.rand(u_num, f_num)
	i_fac = np.random.rand(f_num, i_num)

	# error threshold
	max_err = 0.5

	# weight of penalti on items
	i_weight = 1
	
	for itr in range(max_itr+1):

		# total errors
		sum_err = 0

		# iterate over each rating element
		for u in range(u_num):
			for i in range(i_num):

				# calculate error
				err = all_dat[u,i] - np.dot(u_fac[u,:], i_fac[:,i])
				sum_err = sum_err + err**2 + regularization*(np.linalg.norm(u_fac[u,:])**2 + np.linalg.norm(i_fac[:,i])**2)

				# update factor elements
				u_fac[u,:] = u_fac[u,:] - learn_rate * (-err*i_fac[:,i] + regularization*u_fac[u,:])
				i_fac[:,i] = i_fac[:,i] - learn_rate * (-err*u_fac[u,:] + regularization*i_weight*i_fac[:,i])


		if sum_err < max_err: # converge
			break
		elif itr == max_itr: # fail to converge
			print 'Failed to converge in %d loops' % max_itr

	print 'Iterations: ', itr
	print 'Error: ', sum_err

	# save the output
	#df = pd.DataFrame(u_fac)
	#df.to_csv("tmp/u_fac.tmp", index=False, header=False, sep='\t', encoding='utf-8')
	#df = pd.DataFrame(i_fac)
	#df.to_csv("tmp/i_fac.tmp", index=False, header=False, sep='\t', encoding='utf-8')

	return [u_fac, i_fac]


"""
# for testing
dat = np.matrix('1 4 3; 6 5 2; 1 5 3; 7 9 5')
[u_fac, i_fac] = factor_mat(dat, 2, 5000, 0.001, 0.001)

print "user factors:"
print u_fac
print "item factors:"
print i_fac
print "predicted ratings:"
print np.matmul(u_fac, i_fac)
"""

