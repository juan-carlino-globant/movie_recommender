def recommendation(users, labels_df, model, UserID, n_ratings):
    import numpy as np
    import dfFunctions
    import recommender as recomm

    predicted_ratings = np.array( model.prediction(users[UserID], labels_df['movieId'].tolist() ) )
    recoms = [ (labels_df['title'].iloc[i],predicted_ratings[i]) for i in range(len(predicted_ratings))]

    recoms.sort(key=lambda x: x[1], reverse=True)
    top_n = recoms[:n_ratings]
    labels = [x[0] for x in top_n]
    ratings = [str(x[1]) for x in top_n]

    return { "ratings" : ratings , "labels" : labels}
