import svd 
import n_reco
import time

from flask import Flask
from flask_restful import reqparse, Api, Resource

'''
Generates an API for the recommender, needs an user ID as input
'''

app = Flask(__name__)
api = Api(app)

UserID = 5
n_ratings = 5
Users, labels_df, model = svd.train(relative_path='/movielens/ml-1m/ratings.dat',labels_path='/movielens/ml-1m/movies.dat')

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
        rec = "no reco function yet"
        start = time.time()
        rec = n_reco.recommendation(Users, labels_df, model, user, nr)
        end = time.time()
        print("=> elapsed recommendation time: %s secs" % (end - start))
        return rec

api.add_resource(Recommender, '/reco')

if __name__ == '__main__':
    app.run(debug=False)
