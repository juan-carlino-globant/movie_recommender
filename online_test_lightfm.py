import numpy as np
import pandas as pd
#from lightfm.datasets import fetch_movielens
from lightfm.data import Dataset

def load_data(n_ratings):
    # n_ratings = 500000

    # Load the file into a dataframe
    # table = pd.read_csv("ratings.csv")
    table = pd.read_csv("new_dataset")
    #table = table.iloc[:n_ratings].copy()
    # table = table[ table['rating']>2.5 ]
    table = table[['userId','movieId']]
    # table = table.astype(np.int32)

    # Perform the mapping between users and movies needed for creating interaction matrices
    data = Dataset()
    data.fit(table['userId'], table['movieId'])

    # Creating interaction matrices
    tuples = [tuple(x) for x in table[['userId','movieId']].values]
    (interactions, weights) = data.build_interactions(tuples)

    return interactions



def get_metrics(interactions):
    from lightfm.cross_validation import random_train_test_split
    from lightfm import LightFM
    from lightfm.evaluation import precision_at_k, recall_at_k, auc_score, reciprocal_rank
    # from lightfm.evaluation import auc_score
    import time
    (train_set, test_set) = random_train_test_split(interactions, test_percentage=0.2, random_state=None)

    N_threads = 4

    FILE = open('100k_log.out','w')

    # losses = ['logistic','bpr','warp','warp-kos']
    losses = ['warp']
    print('*********\n LIGHTFM TEST RESULTS\n')
    FILE.write('*********\n LIGHTFM TEST RESULTS\n')
    for loss in losses:

        model = LightFM(no_components=40, loss=loss)

        # Fit the model and measure time
        big_training_start = time.time()
        model.fit(train_set, epochs=40, num_threads=N_threads)
        big_training_end = time.time()
        print("Elapsed time for big set training: %.2f secs" % (big_training_end-big_training_start))
        FILE.write("Elapsed time for big set training: %.2f secs" % (big_training_end-big_training_start))

        biases,items_rep = model.get_item_representations()

        # get precision and AUC for train and test sets. Also measures time
        big_test_start = time.time()
        train_precision = precision_at_k(model, train_set, k=10, num_threads=N_threads).mean()
        test_precision = precision_at_k(model, test_set, train_set, k=10, num_threads=N_threads).mean()

        train_auc = auc_score(model, train_set, num_threads=N_threads).mean()
        test_auc = auc_score(model, test_set, train_set, num_threads=N_threads).mean()

        train_recall = recall_at_k(model, train_set, k=10, num_threads=N_threads).mean()
        test_recall = recall_at_k(model, test_set, train_set, k=10, num_threads=N_threads).mean()

        train_reciprocalR = reciprocal_rank(model, train_set, num_threads=N_threads).mean()
        test_reciprocalR = reciprocal_rank(model, test_set, train_set, num_threads=N_threads).mean()

        big_test_end = time.time()


        print("Elapsed time for big set testing: %.2f secs" % (big_test_end-big_test_start))
        FILE.write("Elapsed time for big set testing: %.2f secs" % (big_test_end-big_test_start))
        # print('*********\n LIGHTFM TEST RESULTS\n')
        print('%s metric:' % loss)
        FILE.write('%s metric:' % loss)
        print('           train , test')
        FILE.write('           train , test')
        print('Precision: %.4f, %.4f.' % (train_precision, test_precision))
        FILE.write('Precision: %.4f, %.4f.' % (train_precision, test_precision))
        print('AUC      : %.4f, %.4f.' % (train_auc, test_auc))
        FILE.write('AUC      : %.4f, %.4f.' % (train_auc, test_auc))
        print('recall   : %.4f, %.4f.' % (train_recall,test_recall))
        FILE.write('recall   : %.4f, %.4f.' % (train_recall,test_recall))
        print('rec. rank: %.4f, %.4f.' % (train_reciprocalR,test_reciprocalR))
        FILE.write('rec. rank: %.4f, %.4f.' % (train_reciprocalR,test_reciprocalR))
        print('*********')
        FILE.write('*********')
    FILE.close()
    return items_rep
