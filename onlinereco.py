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


    movlist = pd.read_csv("movies.csv")

    recommendation = []
    for index in ind[0]:
        title = movlist.iloc[index]['title'].split('(')[0]
        recommendation.append(title[:-1])

    return recommendation
