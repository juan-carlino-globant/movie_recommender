def dummy_classif():
    import numpy as np
    import pandas as pd

    data = pd.read_csv("movies.csv")
    data['genres'] = data['genres'].str.split('|')


    generos = []
    for i in range( len(data['title']) ):
        for tag in data['genres'][i]:
            generos.append(tag)

    generos = np.asarray(generos)


    categorias = {
        'Action' : [],
        'Adventure' : [],
        'Animation' : [],
        'Children' : [],
        'Comedy' : [],
        'Crime' : [],
        'Documentary' : [],
        'Drama' : [],
        'Fantasy' : [],
        'Film-Noir' : [],
        'Horror' : [],
        'IMAX' : [],
        'Musical' : [],
        'Mystery' : [],
        'Romance' : [],
        'Sci-Fi' : [],
        'Thriller' : [],
        'War' : [],
        'Western' : [],
        '(no genres listed)':[]
    }
    ranks = {
        'Action' : 0,
        'Adventure' : 0,
        'Animation' : 0,
        'Children' : 0,
        'Comedy' : 0,
        'Crime' : 0,
        'Documentary' : 0,
        'Drama' : 0,
        'Fantasy' : 0,
        'Film-Noir' : 0,
        'Horror' : 0,
        'IMAX' : 0,
        'Musical' : 0,
        'Mystery' : 0,
        'Romance' : 0,
        'Sci-Fi' : 0,
        'Thriller' : 0,
        'War' : 0,
        'Western' : 0
    }
    # now classify in 19 categories


    def clasify(tags,name):
        for tag in tags:
            for genre in categorias.keys():
                if tag == genre:
                    categorias[genre].append(name)
                    break
        return


    for i in range(len(data['title'])):
        clasify( data['genres'].iloc[i], data['title'].iloc[i] )

    return categorias,data



def dummy_reco(categorias, movies_data, labels, movies_ids, n_recos=10, word='Father'):
    word = word.lower()


    cats_nmbr = len( categorias.keys() )-1
    ranks = [0 for _ in range(cats_nmbr)]
    movies = [[] for _ in range(cats_nmbr)]

    # load all matches in <movies>
    for i in range(cats_nmbr):
        for title in categorias[list(categorias.keys())[i]]:
            short_title = title.split('(')[0][:-1]
            if (title.lower().find(word) >= 0):# and not already_in:
                movies[i].append(title)
                ranks[i] += 1


    def get_result(max_cats,ranks,result):
        # get genre with more titles matching and return those titles as result
        maxix = ranks.index(max(ranks))
        max_cats.extend([list(categorias.keys())[maxix]])
        result.extend(list(set( movies[maxix] )))

        # If there are not enough movies matching for this genre, look in the next
        if (len(result) < n_recos):
            local_r = ranks
            local_r[maxix] = 0
            if (list(set(ranks)) == [0]):
                result = list(set( result ))
                print("No more matches in the data set, "+str(len(result))+" results matching")
                return result
            return get_result(max_cats,local_r,result)

        return list(set( result ))

    def get_nearer(max_cats,ranks,movies_data,result):
        maxix = ranks.index(max(ranks))
        max_cats = list(categorias.keys())[maxix]
        result.extend(list(set( movies[maxix] )))
        needed = n_recos-len(result)
        # print("maxcat",max_cats)
        # print("ranks",ranks)
        # return 1, list([1,2])
        if (len(result) < n_recos):
            # start appending movies with the same genre, but with any title
            count = 0
            out = False
            for i in range( len(movies_data['title']) ):
                if out:
                    break
                title = movies_data['title'].iloc[i]
                movie_id = movies_data['movieId'].iloc[i]
                index, = np.where(movies_ids == movie_id)
                index = index[0]

                if (max_cats == labels[index]):
                    result.append(title)
                    count += 1
                if count == needed:
                    out = True

        return max_cats,list(set( result ))



    result = []
    max_cats = []
    # result = get_result(max_cats,ranks,result)
    max_cats,result = get_nearer(max_cats,ranks,movies_data,result)
    return max_cats,result[:n_recos]

#------------------------------------------------------------------------------#

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns


def cluster_MSD(centroids, labels, movies):
    # centroid is nparray of shape (n_features,)
    distances = [0. for _ in range(len(centroids))]
    NMembers = [0. for _ in range(len(centroids))]
    # print distances, NMembers

    for i in range(len( movies )):
        clustNmbr = labels[i]
        # add square dist of that point to its centroid
        distances[clustNmbr] += np.dot(movies[i],movies[i]) - 2.* np.dot(movies[i],centroids[clustNmbr]) + np.dot(centroids[clustNmbr],centroids[clustNmbr])
        NMembers[clustNmbr] += 1.0

    for i in range(len( centroids )):
        distances[i] /= NMembers[i]

    return distances




def clusterer(items_rep):
    # Load movies
    # movies = np.load("train_movies-2000000ratings_35factors.npy")
    movies = items_rep

    # plot_movies_multidim(movies,0,1)
    # movies = np.transpose(movies)
    X = movies[:]

    # normalising data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # perform clustering
    K = 20
    clustering = KMeans(n_clusters=K).fit(X_scaled)

    # print metrics
    MSdist = cluster_MSD(clustering.cluster_centers_, clustering.labels_, X_scaled)
    dist = 0.
    for i in range(len( clustering.cluster_centers_ )):
        dist += MSdist[i]/len( clustering.cluster_centers_ )
    print ("mean distances to centroid for clusters", dist)
    # print("Number of points",len(clustering.labels_))


    return clustering.labels_



def cluster_classif(items_rep,test_n_ratings):

    file = pd.read_csv("movies.csv")
    ratfile = pd.read_csv("ratings.csv")
    ratfile = ratfile.iloc[:test_n_ratings]
    # ratfile = ratfile[ table['rating']>2.5 ]
    movies_ids = ratfile['movieId'].unique()
    # movies_ids contains the ids of every mobie in the movies ndarray
    movies_ids = np.sort(movies_ids)
    # print(movies_ids)
    # print("peliculas en lista de ids",len(movies_ids))


    labels = clusterer(items_rep)
    # print("puntos en ndarray:",len(labels))

    generos = list( range(np.amax(labels)+1) )

    # There are no names for categories for the time being
    categorias = { x : [] for x in generos}

    for i in range(len(file['title'])):
        movie_id = file['movieId'].iloc[i]
        if movie_id in movies_ids:
            index, = np.where(movies_ids == movie_id)
            index = index[0]
            categorias[labels[index]].append(file['title'].iloc[i])
    return categorias,file,labels,movies_ids



# cats, file, lab, mids= cluster_classif()
# t1,t2  =dummy_reco(cats, file, lab, mids, n_recos=10, word='man')
# print (t1,t2)
