def training():
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import NMF, TruncatedSVD
    from sklearn.externals.joblib import parallel_backend
    # from time import clock
    '''
    Read data from "ratings.csv" and use SVD or NMF to factorize the users-items
    matriz for recommendations
    '''

    # the number of ratings to read, recommendation to make and
    # latent dimensions for factorization
    n_ratings = 300000
    numberOfRecos = 10
    latent_factors = 15

    # read data from file to a table
    data = pd.read_csv("ratings.csv")
    data = data.iloc[0 : n_ratings]
    data = data[['userId', 'movieId', 'rating']]

    R_df = data.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)

    # split data in train and test (currently uses everything as train)
    n_total = R_df.shape[0]
    n_test = int(n_total*0.)

    train = R_df.iloc[:n_total-n_test]
    test = R_df.iloc[n_total-n_test:]


    # factorize matrix (saved in "model")
    # model = NMF(n_components=latent_factors, init='random', random_state=0)
    model = TruncatedSVD(n_components=latent_factors, n_iter=5, random_state=0)
    with parallel_backend('threading'):
        # users
        model = model.fit(train)
    # movies
    H = model.components_

    return model, H, R_df
