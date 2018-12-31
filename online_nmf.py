# def split_set(data):
#     import pandas as pd
#     # users_list = data['userId'].unique()
#     train_df = data
#     test_df = pd.DataFrame({'userId' : [], 'movieId' : [], 'rating' : []})
#     last_rows_idx = []
#
#     for i in data.index[1:]:
#         # mark the index number of the last movie of each user
#         if data.loc[i]['userId'] != data.loc[i-1]['userId']:
#             last_rows_idx.append(i-1)
#
#     # send the last movie of each user to test and erase from train
#     for i in last_rows_idx:
#         test_df = test_df.append(data.loc[i])
#         train_df.drop(i, axis=0, inplace=True)
#
#     test_df.reset_index(inplace=True)
#     train_df.reset_index(inplace=True)
#     test_df.drop('index',axis=1,inplace=True)
#     train_df.drop('index',axis=1,inplace=True)
#
#     return train_df, test_df



# def split_set(R_df):
#     import pandas as pd
#     from numpy.random import choice
#
#     train_df = R_df
#     test_movies = choice(data.columns, size=int(data.shape[1]*.02), replace=False)
#     # test_movies = choice(data.columns, size=15, replace=False)
#     test_ratings = []
#     test_users = []
#
#     for tm in test_movies:
#         fm = R_df[tm].nonzero()[0][0] # Movie
#         test_users.append(R_df.index[fm]) # user (got from its index, just in case)
#         test_ratings.append( R_df[tm][R_df.index[fm]] ) # column = Movie, row = index of first nonzero rating for Movie
#         train_df[tm][R_df.index[fm]] = 0
#
#     # data_df has some erased entries
#     # R_df is the same as before
#
#
#     return train_df,test_movies, test_users, test_ratings




def training(n_ratings, latent_factors, n_iterations):
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import NMF, TruncatedSVD
    from sklearn.externals.joblib import parallel_backend
    # from time import clock
    '''
    Read data from "ratings.csv" and use SVD or NMF to factorize the users-items
    matriz for recommendations
    '''

    # read data from file to a table
    data = pd.read_csv("ratings.csv")
    data = data.iloc[0 : n_ratings]
    data = data[['userId', 'movieId', 'rating']]

    # split data in train and test
    # train_df, test_df = split_set(data) #NEW

    # R_df = data.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
    # R_test = R_df
    R_df = data.pivot(index='userId', columns ='movieId', values='rating').fillna(0) #NEW

    # train_df, test_df = split_set(R_df) #NEW
    train_df = 0
    test_df = 0
    # R_test = test_df.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0) #NEW
    # n_total = R_df.shape[0]
    # n_test = int(n_total*0.)
    #
    # train = R_df.iloc[:n_total-n_test]
    # test = R_df.iloc[n_total-n_test:]


    # factorize matrix (saved in "model")
    # model = NMF(n_components=latent_factors, init='random', random_state=0)
    model = TruncatedSVD(n_components=latent_factors, n_iter=n_iterations, random_state=0)
    with parallel_backend('threading'):
        # users
        # model = model.fit(train)
        # model = model.fit(R_df)#NEW
        U = model.fit_transform(R_df)
    # movies
    H = model.components_

    return data, model, H, R_df, test_df, U
