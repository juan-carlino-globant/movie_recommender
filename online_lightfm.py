from lightfm import LightFM
import pandas as pd
import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score

def load_data(K=10):
    movielens = fetch_movielens()

    data = fetch_movielens(min_rating=4.)
    train = data['train']
    test = data['test']
    model = LightFM(learning_rate=0.05, loss='warp')
    model.fit(data['train'],epochs=20,num_threads=2)

    train_precision = precision_at_k(model, train, k=K).mean()
    test_precision = precision_at_k(model, test, k=K).mean()

    train_auc = auc_score(model, train).mean()
    test_auc = auc_score(model, test).mean()
    print('*********\n LIGHTFM TEST RESULTS\n')
    print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
    print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))
    print('*********')
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
