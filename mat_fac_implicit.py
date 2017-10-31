# -*- coding: utf-8 -*-
"""
Implementation of maxtrix factorization using techniques proposed in "Collaborative Filtering 
for Implicit Feedback Datasets" [1]

Reference:
[1] Hu, Yifan, Yehuda Koren, and Chris Volinsky. "Collaborative filtering for implicit feedback datasets." 
Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on. Ieee, 2008.
"""

import numpy as np
import pandas as pd

def factor_mat(all_dat, f_num, iterations, regularization):
	"""
	estimate an MxF user factor matrix and an FxN item factor matrix from the MxN rating matrix

	@param all_dat {MxN}			: rating matrix (row: user, column: item)
	@param f_num {int}				: # of latent features F
	@param iterations {int}			: # of iterations
	@param regularization {double}	: constant for regularization penalty (to avoid overfitting)
	"""

	# get # of users and # of items
	[u_num, i_num] = all_dat.shape

	# init user factors and item factors with random values
	u_fac = np.matrix(np.random.rand(u_num, f_num))	# MxF
	i_fac = np.matrix(np.random.rand(i_num, f_num))	# NxF

	# calculate the preference matrix
	preference = cal_preference(all_dat)

	# calculate the confidence matrix
	confidence = cal_confidence(all_dat)
	
	# recalculate the user factors and item factors using the alternating least square method
	for itr in range(iterations):
		u_fac = alternate_ls(u_num, i_fac, preference, confidence, regularization)
		#print itr, "u_fac"
		i_fac = alternate_ls(i_num, u_fac, preference.T, confidence.T, regularization)
		#print itr, "i_fac"
	
	# save the output
	df = pd.DataFrame(u_fac)
	df.to_csv("tmp/u_fac.tmp", index=False, header=False, sep='\t', encoding='utf-8')
	df = pd.DataFrame(i_fac.T)
	df.to_csv("tmp/i_fac.tmp", index=False, header=False, sep='\t', encoding='utf-8')

	# an MxF user factor matrix and an FxN item factor matrix
	return [u_fac, i_fac.T]


def cal_confidence(dat):
	"""
	calculate the confidence of each user-item pair

	@param dat {MxN}: rating matrix (row: user, column: item)
	"""

	alpha = 40.0
	confidence = np.zeros(dat.shape)
	confidence = 1 + alpha * dat
	return np.matrix(confidence)

def cal_preference(dat):
	"""
	calculate the preference of each user-item pair

	@param dat {MxN}: rating matrix (row: user, column: item)
	"""

	preference = np.ones(dat.shape)
	preference[dat == 0] = 0
	return np.matrix(preference)

def alternate_ls (u_num, Y, P, C, reg):
	"""
	calculate latent factors using the alternating least square method
	applicable to computing both user factors and item factors

	@param u_num {int}: # of user/item factors
	@param Y {MxF / NxF}: fixed item/user factors
	@param P {MxN}: preference matrix
	@param C {MxN}: confidence matrix
	@param reg {double}: regularization penalty (to avoid overfitting)
	"""

	# get # of items/users and # of latent factors
	[i_num, f_num] = Y.shape

	# output buffer
	X = np.zeros((u_num, f_num))

	# precalculate YtY to improve the performance
	YtY = Y.T * Y

	# iterate over each user/item
	for u in range(u_num):

		# store the diagonal elements of the matrix Cu discussed in the paper in a vector
		Cu = C[u,:]

		# store the coresponding row/column of the preference matrix
		Pu = P[u,:]

		# compute Cu-I
		Cu_I = Cu - 1

		# calculate Yt(Cu-I)Y
		YtCu_IY = np.zeros((f_num, f_num))
		CuIY = np.multiply(Y, Cu_I.T) # weight each row of Y with Cu-I
		for row in range(f_num):
			for col in range(f_num):
				YtCu_IY[row,col] =  Y[:,row].T * CuIY[:,col]
		
		# left term : ((YtCuY + regI)^(-1)) = (YtY + Yt(Cu-I)Y + regI)^(-1)
		left_inv = YtY + YtCu_IY + reg*np.eye(f_num)
		left = np.linalg.inv(left_inv)

		# right term : YtCuPu
		right = Y.T * np.multiply(Cu.T, Pu.T)

		# compute the latent factor of the user/item
		x = left * right
		
		# store it in a matrix
		X[u,:] = x.T

	# return an MxF or NxF matrix
	return np.matrix(X)


"""
# for testing
dat = np.matrix([[1, 2, 3], [4, 5, 0], [1, 5, 3], [7, 3, 9], [0, 4, 3]])
[u_fac, i_fac] = factor_mat(dat, 2, 50, 0.001)
print "user factors:"
print u_fac
print "item factors:"
print i_fac
print "predicted preference:"
print np.matmul(u_fac, i_fac)
"""

