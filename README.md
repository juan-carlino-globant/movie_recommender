# movie_recommender
Recommendation Engine (PoC)

## Files usage

#### Model trainer for the movie recommendator
In order to run the recommendation engine in a micro-service, the following files are needed:
* online_clustering.py (categories recommendator)
* online_nmf.py (training for the user's content recommendator)
* onlinereco.py (user's content recommendator)
* reco_api.py (API)
Along with the following files from the movielens dataset:
* ratings.csv
* movies.csv
These csv files are not contained in this repository.

#### Seting up the virtual environment
Running the command `conda env create -f environment.yaml` will create the virtual env needed for starting the API.
The command `conda activate recom_env` will activate the environment.


#### Usage
Within the virtual environment *recom_env* run `python reco_api.py`. It will take some time to start the service as the recommendation algorithms are trained when the program starts.

To ask for recommendations for a **single user** use the endpoint **reco**. Example: `curl localhost:5000/reco -d "UsrID=10" -d "n_recos=5" -X GET`. Here *UsrID* is the user identifier, an integer number and *n_recos* is the amount of recommendations we want to get for that user, another integer number.

To ask for movies recommendation matching a word** use the endpoint **categs**. Example: ``curl localhost:5000/categs -d "input_str=dumb" -d "n_recos=30" -X GET``. Here *input_str* is the word we use to describe a movie and *n_recos* is the same as in the *reco* endpoint.

#### New dataset generation
The Movielens dataset is somewhat different from the one we want to use. While Movielens has data from about 140.000 users and 27.000 movies, including 20.000.000 ratings, we will test our work in a dataset of 1.000.000 users and 1.000 movies, making 30.000.000 ratings. Note the different sparicty of the user-item matrices here, the Movielens dataset is almost six times "sparser":

`20.000.000 / (140.000*27.000) = 0.005291
 30.000.000 / (1000*1.000.000) = 0.03
 0.03 / 0.005291 = 5.67`

With that in mind, we use the file
* **random_generation.c**

It generates random movies Ids between 1 and 1000 for 1.000.000 users and 30 ratings per user using the function *rand()* from the C standard library.It is able to generate ratings for each movie, although that may depend on the seed (currently initialized with the pc time).

#### Usage
In linux:
* compile the code with gcc: `gcc random_generation.c -o random_generation`. It works fine with gcc version 7.3.0.
* run the executable file `./random_generation`. It will generate a file named *new_dataset* with columns with the format `userId,moviesId,ratings` and 30.000.000 rows.
