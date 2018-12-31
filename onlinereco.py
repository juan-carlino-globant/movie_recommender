def ind2mov(ind):
    # handle usrId error as string type
    if (type(ind)==str):
        return ind

    import pandas as pd
    movlist = pd.read_csv("movies.csv")

    recommendation = []
    for index in ind[0]:
        title = movlist.iloc[index]['title'].split('(')[0]
        recommendation.append(title[:-1])
    return recommendation


def recom(numberOfRecos,model,UsrID,R_df,transformed_movies):
    '''
    takes the neares neighbors model (nbrs) and the matrix factorization
    model (model) with the user ID, data and movies (in the latent dimensions)
    '''
    import numpy as np
    import pandas as pd
    from sklearn.neighbors import NearestNeighbors

    # train a NN finder
    nbrs = NearestNeighbors(n_neighbors=numberOfRecos, algorithm='ball_tree').fit(np.transpose(transformed_movies))

    def NN(user,items):
        newUser = model.transform(user)
        distances, indices = nbrs.kneighbors(newUser)
        return newUser,distances,indices

    if (UsrID > R_df.shape[0]) or UsrID < 1:
        error = '<User> variable must be between 1 and '+str(R_df.shape[0])
        return error


    test_usr = R_df.loc[UsrID]
    test_usr = test_usr.values.reshape(-1,test_usr.shape[0])
    nu,dis,ind = NN(test_usr,transformed_movies)

    return ind




def get_RMSE(R_df, test, model, transformed_movies, transformed_users, data, latent_factors, n_iterations):
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import NMF, TruncatedSVD

    # train on complete data
    table = data.pivot(index='userId', columns ='movieId', values='rating').fillna(0)
    real_model = TruncatedSVD(n_components=latent_factors, n_iter=n_iterations, random_state=0)
    U = real_model.fit_transform(table)
    H = real_model.components_


    # using test_df as input
    Tmovies = np.transpose(transformed_movies)
    TH = np. transpose(H)
    RMSE = 0.

    for usr_idx in range(len( test['userId'] )):
        # products for test data

        movie_nmbr = test['movieId'][usr_idx]
        if movie_nmbr in R_df.columns:
            movie_idx = R_df.columns.get_loc(movie_nmbr)
        else:
            print("movie not found")

        newUser = transformed_users[usr_idx]
        newMovie = Tmovies[movie_idx] # transpose? sure???
        estim_sim = np.dot(newUser,newMovie) / np.sqrt( (np.dot(newUser,newUser)*np.dot(newMovie,newMovie)) )

        # Now, look for the real product:
        # users (almost) share indexes in test and table
        real_user = U[usr_idx]
        real_movie = TH[movie_idx]
        real_sim = np.dot(real_user,real_movie) / np.sqrt( (np.dot(real_user,real_user)*np.dot(real_movie,real_movie)) )

        RMSE += (estim_sim-real_sim)*(estim_sim-real_sim)
    RMSE = np.sqrt(RMSE/float(len(test['userId'])) )

    return RMSE
