import numpy as np
import pandas as pd
#from lightfm.datasets import fetch_movielens
from lightfm.data import Dataset

def load_data(n_ratings):
    # n_ratings = 500000

    # Load the file into a dataframe
    # table = pd.read_csv("ratings.csv")
    table = pd.read_csv("new_dataset")
    table = table.iloc[:n_ratings].copy()
    # table = table[ table['rating']>2.5 ]
    table = table[['userId','movieId','rating']]
    # table = table.astype(np.int32)

    # Perform the mapping between users and movies needed for creating interaction matrices
    data = Dataset()
    data.fit(table['userId'], table['movieId'], table['rating'])

    # Creating interaction matrices
    tuples = [tuple(x) for x in table[['userId','movieId','rating']].values]
    (interactions, weights) = data.build_interactions(tuples)

    return interactions



def get_metrics(interactions):
    from lightfm.cross_validation import random_train_test_split
    from lightfm import LightFM
    from lightfm.evaluation import precision_at_k
    from lightfm.evaluation import auc_score
    import time
    (train_set, test_set) = random_train_test_split(interactions, test_percentage=0.2, random_state=None)

    N_threads = 4
    model = LightFM(no_components=35, loss='warp')

    # Fit the model and measure time
    big_training_start = time.time()
    model.fit(train_set, epochs=30, num_threads=N_threads)
    big_training_end = time.time()
    print("Elapsed time for big set training: %.2f secs" % (big_training_end-big_training_start))

    biases,items_rep = model.get_item_representations()

    # get precision and AUC for train and test sets. Also measures time
    big_test_start = time.time()
    train_precision = precision_at_k(model, train_set, k=5, num_threads=N_threads).mean()
    test_precision = precision_at_k(model, test_set, k=5, num_threads=N_threads).mean()

    train_auc = auc_score(model, train_set, num_threads=N_threads).mean()
    test_auc = auc_score(model, test_set, num_threads=N_threads).mean()
    big_test_end = time.time()
    print("Elapsed time for big set testing: %.2f secs" % (big_test_end-big_test_start))

    print('*****\nRatings big dataset:\n')
    print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
    print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))
    print('*****')
    return items_rep
