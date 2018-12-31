import online_nmf
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
n_ratings = 30000
latent_factors = 15
n_iterations = 5

# training matrix factorization
training_start = time.time()
data, model, movies, R_df, test, users = online_nmf.training(n_ratings, latent_factors, n_iterations)
training_end = time.time()
print("Elapsed time for movie recommendation training: %.2f secs" % (training_end-training_start))

# classify movies by tag
categories, movies_data = online_clustering.dummy_classif()
# print RMSE for our trained model
print(onlinereco.get_RMSE(R_df, test, model, movies, users, data, latent_factors, n_iterations))

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
        ret0 = onlinereco.recom(nr,model,user,R_df,movies)
        return onlinereco.ind2mov( ret0 )

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
        return list( online_clustering.dummy_reco(categorias=categories, movies_data=movies_data, n_recos=nr, word=word) )


api.add_resource(Recommender, '/reco')
api.add_resource(Categories, '/categs')


if __name__ == '__main__':
    app.run(debug=True)
