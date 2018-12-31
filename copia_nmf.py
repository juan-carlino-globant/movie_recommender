

def SGD(data,R_df,info,n_factors,n_epochs):
    import numpy as np
    '''Learn the vectors p_u and q_i with SGD.
       data is a dataset containing all ratings + some useful info (e.g. number
       of items/users).
    '''

    # n_factors = 10  # number of factors
    alpha = .01  # learning rate
    # n_epochs = 50  # number of iteration of the SGD procedure

    users = [ t[0] for t in info]
    movies = [ t[1] for t in info ]
    # print (len(np.unique(users)))
    # print (len(np.unique(movies)))
    n_users = len(np.unique(users))#R_df.shape[0]
    n_items = len(np.unique(movies))#R_df.shape[1]

    # Randomly initialize the user and item factors.
    p = np.random.normal(0, .1, (n_users, n_factors))
    q = np.random.normal(0, .1, (n_items, n_factors))
    print p[0].shape
    print q[0].shape
    # Optimization procedure
    for _ in range(n_epochs):
        for u,i,r in info:
            u_idx = users.index(u)
            i_idx = movies.index(i)
            err = r - np.dot(p[u_idx], q[i_idx])
            return
            # Update vectors p_u and q_i
            p[u_idx] += alpha * err * q[i_idx]
            q[i_idx] += alpha * err * p[u_idx]

    return p,q

def get_uir(data):
    info = []
    for i in range(len(data['userId'])):
        u = data['userId']
        i = data['movieId']
        r = data['rating']
        info.append((u,i,r))
    return info


# def estimate(p,q,user,item)
#     return




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

    R_df = data.pivot(index='userId', columns ='movieId', values='rating').fillna(0) #NEW

    train_df = 0
    test_df = 0

    # factorize matrix (saved in "model")
    # model = NMF(n_components=latent_factors, init='random', random_state=0)
    model = TruncatedSVD(n_components=latent_factors, n_iter=n_iterations, random_state=0)
    with parallel_backend('threading'):
        # users
        U = model.fit_transform(R_df)
    # movies
    H = model.components_

    return data, model, H, R_df, test_df, U


def training_surpriselike(n_ratings, latent_factors, n_iterations):
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
    info = get_uir(data)
    R_df = data.pivot(index='userId', columns ='movieId', values='rating').fillna(0) #NEW
    test_df = 0

    p, q = SGD(data, R_df, info, latent_factors, n_iterations)

    train_df = 0
    model = 0
    H = 0
    U = 0
    return data, R_df, test_df, p, q
