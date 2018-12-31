from collections import defaultdict
import pandas as pd

def get_top_n(predictions, user, n=5):
    '''Return the top-N recommendations for each user from a set of predictions.
    '''

    user_est = [ (int(iid), est) for uid, iid, true_r, est, _ in predictions if uid  == user ]
    user_est.sort(key=lambda x: x[1], reverse=True)
    user_est = user_est[:n]

    movlist = pd.read_csv("movies.csv")

    recommendation = []
    for index, _ in user_est:
        title = movlist.iloc[index]['title'].split('(')[0]
        recommendation.append(title[:-1])

    return recommendation
