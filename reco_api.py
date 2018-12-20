import online_nmf
import onlinereco
import online_clustering
from flask import Flask
from flask_restful import reqparse, Api, Resource
'''
Generates an API for the recommender, needs an user ID as input
'''


app = Flask(__name__)
api = Api(app)
model, movies, R_df= online_nmf.training()
categories, movies_data = online_clustering.dummy_classif()


class Recommender(Resource):

    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('UsrID')
        parser.add_argument('n_recos')
        args = parser.parse_args()
        user = int(args['UsrID'])
        nr = int(args['n_recos'])
        return onlinereco.recom(nr,model,user,R_df,movies)

class Categories(Resource):

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
