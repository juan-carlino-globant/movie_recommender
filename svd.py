def train(relative_path,
        labels_path,
        batch=700,
        steps=7000,
        dimension=12,
        reg=0.0003,
        learning=0.001,
        momentum=0.926,
        info=True,
        model='svd',
        nsvd_size='mean'):

    from os import path
    import numpy as np

    import sys
    parent_path = path.abspath('')
    sys.path.insert(0, parent_path)
    import dfFunctions
    import recommender as recomm

    path = parent_path + relative_path
    labels_path  = parent_path + labels_path
    df = dfFunctions.load_dataframe(path)
    print("=> dataframe shape: %s x %s" % df.shape)

    if model == "svd":
        model = recomm.SVDmodel(df, 'user', 'item', 'rating')
    else:
        model = recomm.SVDmodel(df,
                                'user',
                                'item',
                                'rating',
                                model,
                                nsvd_size)

    regularizer_constant = reg
    learning_rate = learning
    batch_size = batch
    num_steps = steps
    momentum_factor = momentum


    model.training(dimension,
                   regularizer_constant,
                   learning_rate,
                   momentum_factor,
                   batch_size,
                   num_steps,
                   info)



    # Get items labels (movie names)
    import pandas as pd
    col_names = ["movieId", "title", "genres"]
    labels_df = pd.read_csv(labels_path, sep='::', names=col_names, engine='python')
    labels_df['movieId'] = labels_df['movieId'] - 1
    labels_df = labels_df[["movieId","title"]]


    # Return all users and movies
    print("******************************************")
    print("Training users",len(np.array(model.train['user'])))
    users = np.array(model.train['user'].append(model.test['user'].append(model.valid['user']))).reshape(1,-1).T
    movies = np.array(model.train['item'].append(model.test['item'].append(model.valid['item']))).reshape(1,-1)
    print("Total users",len(users))
    print("Movies:",len(labels_df['movieId']))
    print("******************************************\n")


    # A quick recommendation demo
    prediction = model.valid_prediction()
    print("\nThe mean square error of the whole valid dataset is ", prediction)
    user_example = np.array(model.valid['user'])[0:10]
    movies_example = np.array(model.valid['item'])[0:10]
    actual_ratings = np.array(model.valid['rating'])[0:10]
    predicted_ratings = model.prediction(user_example, movies_example)
    print("""\nUsing our model for 10 specific users and 10
    movies we predicted the following score:""")
    print(predicted_ratings)
    print("\nAnd in reality the scores are:")
    print(actual_ratings)


    # Save the list with movieIds and movie names
    ml_list = np.array(labels_df)
    
    return users, labels_df, model
