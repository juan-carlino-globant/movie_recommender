import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import online_nmf
import time

N_R = [100000,200000]
L_F = [2]
N_I = [5]


def split_set(R_df):
    import pandas as pd
    from numpy.random import choice

    def contains_duplicates(X):
        return len(np.unique(X)) != len(X)

    train_df = R_df.copy(deep=True)
    test_movies = choice(R_df.columns, size=int(R_df.shape[1]*0.2), replace=False)

    if (contains_duplicates(test_movies)):
        print("DUPS")
        return

    # print("using",len(test_movies),"movies as test")

    test_ratings = []
    test_users = []

    skipped = 0
    for tm in test_movies:
        aux = R_df[tm].nonzero()[0]
        if aux.size == 0:
            skipped += 1
            continue
        fm = aux[0] # Movie
        test_users.append(R_df.index.get_loc(R_df.index[fm]))
        test_ratings.append( R_df[tm][R_df.index[fm]] ) # column = Movie, row = index of first nonzero rating for Movie
        train_df[tm].iloc[R_df.index[fm]] = 0

        # data_df has some erased entries
        # R_df is the same as before
    if skipped != 0:
        print("(",skipped," skipped in split_set)")
    test_movies = [ R_df.columns.get_loc(k) for k in test_movies]

    return train_df,test_movies, test_users, test_ratings



def get_RMSEII(users,movies,R_df,test_users,test_movies,test_ratings,model):
    import numpy as np
    # usual RMSE calculation

    U = np.asarray(users)
    M = np.transpose(movies)
    test_users = np.array(test_users)
    test_movies = np.array(test_movies)

    estimates = []
    reals = []

    RMSEII = 0.
    for idx in range(len(test_users)):
        tusr = U[test_users[idx]]
        tusr = tusr.reshape(-1,tusr.shape[0])

        tmov = M[test_movies[idx]]
        tmov = tmov.reshape(-1,tmov.shape[0])

        realthing = test_ratings[idx]
        rat = np.dot(tmov[0],tusr[0])
        RMSEII += (rat-realthing)*(rat-realthing)

    RMSEII = np.sqrt(RMSEII/float(len(test_users)))
    return RMSEII



FILE = open('benchmarking_SVD.csv','w')
FILE.write("n_ratings,latent_factors,n_iterations,meanRMSE,variance,time\n")

for n_ratings in N_R:
    for latent_factors in L_F:
        for n_iterations in N_I:

            training_start = time.time()
            data, model, movies, R_df, test, users = online_nmf.training(n_ratings, latent_factors, n_iterations)
            training_end = time.time()
            train_time = training_end-training_start
            print("Elapsed time for movie recommendation training: %.2f secs" % (train_time))
            print("users:",R_df.shape[0])
            print("movies:",R_df.shape[1])

            N_tests = 10
            mean = 0.
            var = 0.
            results = []

            for i in range(N_tests):
                train_df,test_movies, test_users, test_ratings = split_set(R_df)
                r = get_RMSEII(users,movies,R_df,test_users,test_movies,test_ratings,model)
                results.append(r)
                mean += r

            mean = mean/float(N_tests)

            for i in range(N_tests):
                var += (results[i]-mean)*(results[i]-mean)

            var = np.sqrt(var/float(N_tests))
            print("MEAN/VARIANCE: ",mean,'/',var)
            FILE.write(str(n_ratings)+","+str(latent_factors)+","+str(n_iterations)+","+str(mean)+","+str(var)+","+str(train_time)+'\n')
