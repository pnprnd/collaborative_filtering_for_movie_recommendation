# Recommendation System

This project shows the implementation of a recommendation system based on the **collaborative filtering method** proposed in [1]. **MovieLens 100M Dataset** [2] is used as the training data set. The main function, which is *recommend_items()* in *main.py*, takes 6 parameters as the input: 1) the user ID 2) the number of recommended items 3) the recommendation mode (recommend only unseen items or not) 4) the number of latent factor 5) the number of iterations for the matrix factorization method 6) the regularization constant for avoiding overfitting. The algorithm starts by reading the data from *u.data* and generating a user-item pair matrix (rating matrix). This rating matrix is then decomposed into two latent factor matrices (user factor matrix and item factor matrix) using [1]. These matrices are stored in *tmp/u_fac.tmp* and *tmp/i_fac.tmp*, so that when these files exist the latent factors are not recomputed. Once a user factor matrix and an item factor matrix are obtained, the preference of each user to each item is calculated. For the indicated user, items with high preference are selected as recommended items.

The main program includes *main.py*, *data_input.py*, *mat_fac_implicit.py* and *recommendation.py*. However, in the first attempt, the recommendation system was implimented using matrix factorization and gradient boosting [3], which requires more computation time and tends to assume unrated items as dislike items. The code is also available in *matrix_factorization.py*.


## References:
[1] Hu, Yifan, Yehuda Koren, and Chris Volinsky. "Collaborative filtering for implicit feedback datasets." 
Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on. Ieee, 2008.

[2] https://grouplens.org/datasets/movielens/

[3] Nguyen, Jennifer, and Mu Zhu. "Content‐boosted matrix factorization techniques for 
recommender systems." Statistical Analysis and Data Mining: The ASA Data Science 
Journal 6.4 (2013): 286-301.
