from collections import defaultdict

from surprise import SVD
from surprise import Dataset
from surprise.model_selection import train_test_split
from surprise import accuracy

def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


# First train an SVD algorithm on the movielens dataset.
n_ratings = 1000000
data_df = pd.read_csv("ratings.csv")
data_df = data_df.iloc[:n_ratings]
reader = Reader(rating_scale=(0.5,5.0))
data = Dataset.load_from_df(data_df[['userId', 'movieId', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=.25)

#data = Dataset.load_builtin('ml-100k')
#trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)

# Than predict ratings for all pairs (u, i) that are NOT in the training set.
#testset = trainset.build_anti_testset()
predictions = algo.test(testset)

top_n = get_top_n(predictions, n=10)
# get RMSE
accuracy.rmse(predictions)

# Print the recommended items for each user
#for uid, user_ratings in top_n.items():
#    print(uid, [iid for (iid, _) in user_ratings])







from surprise import BaselineOnly
from surprise import SVD
from surprise import Dataset
from surprise import Reader
import os
import pandas as pd
from surprise.model_selection import cross_validate

n_ratings = 1000000

# path to dataset file
file_path = os.path.expanduser('./ratings_new.csv')
data_df = pd.read_csv("ratings.csv")
data_df = data_df.iloc[:n_ratings]
reader = Reader(rating_scale=(0.5,5.0))
data = Dataset.load_from_df(data_df[['userId', 'movieId', 'rating']], reader)


# We can now use this dataset as we please, e.g. calling cross_validate
cross_validate(SVD(), data, verbose=True, n_jobs=2)
