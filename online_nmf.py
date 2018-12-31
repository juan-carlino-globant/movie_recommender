
from surprise import SVD
from surprise import Dataset
from surprise import Reader

import time
import os

# def movies():
#     movlist = pd.read_csv("movies.csv")

# def users():
#     users = pd.read_csv("ratings.csv", usecols=['userId'])
#     users = users.drop_duplicates(subset=['userId'], keep=False)

def training():

    # path to dataset file
    file_path = os.path.expanduser('./ratings.csv')

    # As we're loading a custom dataset, we need to define a reader. In the
    # movielens-100k dataset, each line has the following format:
    # 'user item rating timestamp', separated by '\t' characters.
    reader = Reader(line_format='user item rating', sep=',', rating_scale=(1, 5), skip_lines=1)

    start = time.time()
    data = Dataset.load_from_file(file_path, reader=reader)
    end = time.time()
    print("=> elapsed Dataset load: %s secs" % (end - start))

    algo = SVD()
    # Retrieve the trainset.
    trainset = data.build_full_trainset()
    start = time.time()
    algo.fit(trainset)
    end = time.time()
    print("=> elapsed algorithm fit: %s secs" % (end - start))
    #cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    # Than predict ratings for all pairs (u, i) that are NOT in the training set.
    start = time.time()
    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)
    end = time.time()
    print("=> elapsed predict time: %s secs" % (end - start))

    return predictions
