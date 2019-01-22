# movie_recommender
Recommendation Engine (PoC)

Implementation based on: https://github.com/felipessalvatore/Recommender

## Setup

Main requirements are:

* Tensorflow
* Numpy
* Pandas

### How to setup the project with virtualenv

```
$ virtualenv /usr/bin/python3 venv
$ venv/bin/activate
$ pip install -r requirements.txt
```

### Download MovieLens data

```
$ ./download_data.sh
```

### New dataset generation
The Movielens dataset is somewhat different from the one we want to use. While Movielens has data from about 140.000 users and 27.000 movies, including 20.000.000 ratings, we will test our work in a dataset of 1.000.000 users and 1.000 movies, making 30.000.000 ratings. Note the different sparicty of the user-item matrices here, the Movielens dataset is almost six times "sparser":

`20.000.000 / (140.000*27.000) = 0.005291
 30.000.000 / (1000*1.000.000) = 0.03
 0.03 / 0.005291 = 5.67`

With that in mind, we use the file
* **random_generation.c**

It generates random movies Ids between 1 and 1000 for 1.000.000 users and 30 ratings per user using the function *rand()* from the C standard library.It is able to generate ratings for each movie, although that may depend on the seed (currently initialized with the pc time).

## Usage

1. Activate your virtualenv if you didn't it already

```
$ venv/bin/activate
```

2. Run the SVD example

```

$ python svd.py --help
usage: svd.py [-h] [-p PATH] [-e EXAMPLE] [-b BATCH] [-s STEPS] [-d DIMENSION]
              [-r REG] [-l LEARNING] [-m MOMENTUM] [-i INFO] [-M MODEL]
              [-S NSVD_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  ratings path (default=pwd/movielens/ml-1m/ratings.dat)
  -e EXAMPLE, --example EXAMPLE
                        movielens dataset examples (only 1, 10 or 20)
                        (default=1)
  -b BATCH, --batch BATCH
                        batch size (default=700)
  -s STEPS, --steps STEPS
                        number of training steps (default=7000)
  -d DIMENSION, --dimension DIMENSION
                        embedding vector size (default=12)
  -r REG, --reg REG     regularizer constant for the loss function
                        (default=0.0003)
  -l LEARNING, --learning LEARNING
                        learning rate (default=0.001)
  -m MOMENTUM, --momentum MOMENTUM
                        momentum factor (default=0.926)
  -i INFO, --info INFO  Training information. Only True or False
                        (default=True)
  -M MODEL, --model MODEL
                        models: either svd or nsvd (default=svd)
  -S NSVD_SIZE, --nsvd_size NSVD_SIZE
                        size of the vectors of the nsvd model: either max,
                        mean or min (default=mean)



```

## Example

```
$ bash download_data.sh
$ cd examples/
$ python svd.py -s 20000

>> step batch_error test_error elapsed_time
  0 3.930429 3.988358* 0.243376(s)
1000 0.943535 0.934758* 1.532505(s)
2000 0.921224 0.933712* 1.571072(s)
3000 0.943956 0.927437* 1.534095(s)
4000 0.913235 0.840039* 1.525031(s)
5000 0.897798 0.901872 1.281967(s)
6000 0.978220 0.896336 1.277157(s)
7000 0.899796 0.903618 1.292524(s)
8000 0.925525 0.944306 1.279324(s)
9000 0.894377 0.883023 1.285019(s)
10000 0.924365 0.941058 1.279905(s)
11000 0.921969 0.897630 1.267302(s)
12000 0.917880 0.899381 1.274572(s)
13000 0.922738 0.933798 1.285953(s)
14000 0.876588 0.946282 1.285653(s)
15000 0.904958 0.891187 1.278772(s)
16000 0.954195 0.907019 1.293461(s)
17000 0.900970 0.903008 1.294990(s)
18000 0.902404 0.879164 1.277366(s)
19000 0.875246 0.957183 1.292368(s)

>> The duration of the whole training with 20000 steps is 26.93 seconds,
which is equal to:  0:0:0:26 (DAYS:HOURS:MIN:SEC)

>> The mean square error of the whole valid dataset is  0.915779

>> Using our model for 10 specific users and 10 movies we predicted the following score:
[ 4.11244917  4.38496399  3.26372051  3.59210873  1.446275    3.33612514
  3.27328825  4.65662336  2.41137171  3.19429493]

>> And in reality the scores are:
[ 5.  5.  1.  1.  1.  5.  5.  5.  1.  2.]

```

## API
The new API contains the recommender algorithm. The training is automatically run when the API starts, after that, there is only one endpoint (at least for now) which can be used to request recommended movies for a certain user.

### Usage
In order to use th API the virtual env must be active. Once the environment is activated run the program by using:

```
python reco_api.py
```
A sample output is
```
$ python reco_api.py

=> dataframe shape: 1000209 x 3
(Some deprecation warnings, to be treated properly).
********** TRAINING IN PROGRESS **********
 step  batch_error test_error elapsed_time
    0  4.485020    2.091566*  0.081800(s)
 1000  0.932317    0.913201*  0.852809(s)
 2000  0.905894    0.943206   0.829196(s)
 3000  0.935509    0.926992   0.839957(s)
 4000  0.891815    0.895175*  0.859693(s)
 5000  0.900175    0.830581*  0.831637(s)
 6000  0.865771    0.885482   0.826419(s)

The duration of the whole training with 7000 steps is 5.94 seconds,
which is equal to:  0:0:0:5 (DAYS:HOURS:MIN:SEC)
******************************************
Training users 800167
Total users 1000209
Movies: 3883
******************************************


The mean square error of the whole valid dataset is  0.9008459

Using our model for 10 specific users and 10
    movies we predicted the following score:
[3.3169467 4.3824377 3.4254353 2.1723516 3.4878612 4.095423  4.670119
 2.9581535 4.39942   3.0512538]

And in reality the scores are:
[4. 3. 3. 3. 4. 4. 5. 2. 5. 5.]
 * Serving Flask app "reco_api" (lazy loading)
 * Environment: production
   WARNING: Do not use the development server in a production environment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

```
Once the API is running we can access its endpoint as `localhots:5000/reco` using
a GET method and specifying `UsrID` (the ID of the user we are requesting recommendations for)
and `n_recos`(the number of recommendations we want to get for that user).


example for linux `curl localhost:5000/reco -d "UsrID=40" -d "n_recos=5" -X GET`
