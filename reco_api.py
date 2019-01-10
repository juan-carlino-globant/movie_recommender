import online_nmf
import onlinereco
import time

from flask import Flask
from flask_restful import reqparse, Api, Resource

'''
Generates an API for the recommender, needs an user ID as input
'''

app = Flask(__name__)
api = Api(app)
predictions = online_nmf.training()



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
        user = args['UsrID']
        nr = int(args['n_recos'])
        start = time.time()
        rec = onlinereco.get_top_n(predictions,user=user,n=nr)
        end = time.time()
        print("=> elapsed recommendation time: %s secs" % (end - start))
        return rec

api.add_resource(Recommender, '/reco')

if __name__ == '__main__':
    app.run(debug=False)
