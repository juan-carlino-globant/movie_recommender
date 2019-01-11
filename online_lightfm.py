from lightfm import LightFM
import pandas as pd
import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score, reciprocal_rank
# from lightfm.evaluation import auc_score

def load_data(K=10):
    # movielens = fetch_movielens()

    data = fetch_movielens(min_rating=4.)
    train = data['train']
    test = data['test']

    FILE = open('100M_log.out','w')

    # losses = ['logistic','bpr','warp','warp-kos']
    losses = ['warp']
    print('*********\n LIGHTFM TEST RESULTS\n')
    FILE.write('*********\n LIGHTFM TEST RESULTS\n')
    for loss in losses:

        model = LightFM(learning_rate=0.05, loss='warp')
        model.fit(data['train'],epochs=40,num_threads=2)

        train_precision = precision_at_k(model, train, k=K).mean()
        test_precision = precision_at_k(model, test, k=K).mean()

        train_auc = auc_score(model, train).mean()
        test_auc = auc_score(model, test).mean()

        train_recall = recall_at_k(model, train, k=K).mean()
        test_recall = recall_at_k(model, test, k=K).mean()

        train_reciprocalR = reciprocal_rank(model, train).mean()
        test_reciprocalR = reciprocal_rank(model, test).mean()

        # print("Elapsed time for big set testing: %.2f secs" % (big_test_end-big_test_start))
        # FILE.write("Elapsed time for big set testing: %.2f secs" % (big_test_end-big_test_start))
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
    return model, data


def recommendation(model, data, UsrIDs, n_recos):
    n_users, n_items = data['train'].shape

    if type(UsrIDs=='int'):
        known_positives = data['item_labels'][data['train'].tocsr()[UsrIDs].indices]
        scores = model.predict(UsrIDs, np.asarray(range(n_items)))
        top_items = data['item_labels'][np.argsort(-scores)]

        known_positives = known_positives[:n_recos]
        top_items = top_items[:n_recos]

    elif type(UsrUDs=='list'):
        for user_id in UsrIDs:

            known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
            scores = model.predict(user_id, np.asarray(range(n_items)))
            top_items = data['item_labels'][np.argsort(-scores)]

            known_positives = known_positives[:n_recos]
            top_items = top_items[:n_recos]

    return {'Recommended' : list(top_items) , 'Known' : list(known_positives) }
