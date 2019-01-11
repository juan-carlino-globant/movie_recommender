import online_nmf
import online_lightfm
import online_test_lightfm
import onlinereco
import online_clustering
import time
from flask import Flask
from flask_restful import reqparse, Api, Resource
'''
Generates an API for the recommender, needs an user ID as input
'''


app = Flask(__name__)
api = Api(app)

# training parameters, shared with rmse calculator
n_ratings = 100000


# lightfm
training_start = time.time()
model, data = online_lightfm.load_data()
training_end = time.time()
print("Elapsed time for 100k set training: %.2f secs" % (training_end-training_start))


test_n_ratings = 2000000

big_loading_start = time.time()
interactions = online_test_lightfm.load_data(test_n_ratings)
big_loading_end = time.time()
print("Elapsed time for big set loading: %.2f secs" % (big_loading_end-big_loading_start))

items_rep = online_test_lightfm.get_metrics(interactions)

quit()
# categories, movies_data = online_clustering.dummy_classif()
categories, movies_data, labels, movie_ids = online_clustering.cluster_classif(items_rep,test_n_ratings)


class Recommender(Resource):
    '''
    This class has only one method. It takes an user ID and the number of recommendations requested
    for that user. Returns a list of recommended movies.
    '''
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('UsrID')
        parser.add_argument('n_recos')
        args = parser.parse_args()
        user = int(args['UsrID'])
        nr = int(args['n_recos'])
        return online_lightfm.recommendation(model, data, user, nr)

class Categories(Resource):
    '''
    Preliminar version of categories recommender. takes a single input word
    and the number of movies requested as recommendations. returns the category
    used and the recommended movies inside that category.
    '''
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('input_str')
        parser.add_argument('n_recos')
        args = parser.parse_args()
        word = str(args['input_str'])
        nr = int(args['n_recos'])
        # return list( online_clustering.dummy_reco(categorias=categories, movies_data=movies_data, n_recos=nr, word=word) )
        return list( online_clustering.dummy_reco(categories, movies_data, labels, movie_ids, n_recos=nr, word=word) )

api.add_resource(Recommender, '/reco')
api.add_resource(Categories, '/categs')


if __name__ == '__main__':
    app.run(debug=False)
