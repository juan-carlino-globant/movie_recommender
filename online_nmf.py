def training():
    import numpy as np
    import pandas as pd
    from utils import Timer
    from sklearn.decomposition import NMF, TruncatedSVD
    from sklearn.externals.joblib import parallel_backend

    '''
    Read data from "ratings.csv" and use SVD or NMF to factorize the users-items
    matriz for recommendations
    '''

    # the number of ratings to read, recommendation to make and
    # latent dimensions for factorization
    n_ratings = 300000
    numberOfRecos = 10
    latent_factors = 15

    data = None
    with Timer() as t:
        # read data from file to a table
        data = pd.read_csv("ratings.csv", dtype = {'userId':np.uint32,'movieId':np.uint16,'rating':np.uint8})
        #data = data.iloc[0 : n_ratings]
        data = data[['userId', 'movieId', 'rating']]
    print("=> elapsed Dataset load: %s secs" % (t.interval))

    R_df = data.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)

    # split data in train and test (currently uses everything as train)
    n_total = R_df.shape[0]
    n_test = int(n_total*0.)

    train = R_df.iloc[:n_total-n_test]
    test = R_df.iloc[n_total-n_test:]


    # factorize matrix (saved in "model")
    # model = NMF(n_components=latent_factors, init='random', random_state=0)
    model = TruncatedSVD(n_components=latent_factors, n_iter=5, random_state=0)
    with Timer() as t:
        with parallel_backend('threading'):
            # users
            model = model.fit(train)
    print("=> elapsed algorithm fit: %s secs" % (t.interval))

    # movies
    H = model.components_

    return model, H, R_df
