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



def dummy_reco(categorias, movies_data, n_recos=10, word='Father'):
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

        if (len(result) < n_recos):
            # start appending movies qith the same genre, but with any title
            count = 0
            out = False
            for i in range( len(movies_data['title']) ):
                if out:
                    break
                title = movies_data['title'].iloc[i]
                if (max_cats in movies_data['genres'].iloc[i]) and not (title in result):
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
