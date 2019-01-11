#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#define MOVIE_QTY           3000
#define USER_QTY            100000
#define RATINGS_PER_USER    30


// Gaussian-distributed pseudo-random number generator
double randn (double mu, double sigma)
{
  double U1, U2, W, mult;
  static double X1, X2;
  static int call = 0;

  if (call == 1)
    {
      call = !call;
      return (mu + sigma * (double) X2);
    }

  do
    {
      U1 = -1 + ((double) rand () / RAND_MAX) * 2;
      U2 = -1 + ((double) rand () / RAND_MAX) * 2;
      W = pow(U1, 2) + pow(U2, 2);
    }
  while (W >= 1 || W == 0);

  mult = sqrt((-2 * log(W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;

  call = !call;

  return (mu + sigma * (double) X1);
}

// array initializer
char* init_assign_map()
{
    static char movie_assignation[MOVIE_QTY];
    for (int i=0; i<MOVIE_QTY; i++)
    {
        movie_assignation[i] = 0;
    }
    return movie_assignation;
}

// int movie_idx(int random)
// {
//    return (random % MOVIE_QTY) + 1;
// }

int* get_movies(int llimit)
{
    static int movies[RATINGS_PER_USER];
    char* movie_assignation = init_assign_map();

    int nmbr, idx;

    for (int i=0; i<RATINGS_PER_USER; i++)
    {
        do
        {
            nmbr = rand();
            // idx = movie_idx(nmbr);
            idx = (nmbr % MOVIE_QTY/3) + llimit + 1;
            if (idx>MOVIE_QTY)
            {
                printf("MOVIE OUT, USING LAST MOVIE");
                idx = MOVIE_QTY-1;
            }
        } while (movie_assignation[idx] != 0);

        movies[i] = idx;
        movie_assignation[idx] = 1;
    }
    return movies;
}

int main()
{
    // seed for random numbers
    srand(time(0));
    // open and write header of .csv file
    FILE *outfile;
    outfile = fopen("new_dataset", "w");
    fprintf(outfile,"userId,movieId,rating\n");

    // declaration of some vars
    int* movies;
    // float possible_ratings[10] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0};
    float possible_ratings[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float usr_rats[RATINGS_PER_USER];

    // creating some users groups
    int ulimit0, ulimit1, ulimit2, mlimit0, mlimit1, mlimit2, llimit, rnmbr;
    ulimit0 = 0;
    ulimit1 = USER_QTY/3;
    ulimit2 = 2*USER_QTY/3;
    mlimit0 = 1;
    mlimit1 = MOVIE_QTY/3;
    mlimit2 = 2*MOVIE_QTY/3;

    // Random generator parameters
    double mu,sigma;
    mu = 2.5;
    sigma = 1.0;

    for (int usrId=1; usrId<=USER_QTY; usrId++)
    {
        if (usrId > ulimit0)
        {
            llimit = mlimit0;
        }
        if (usrId > ulimit1)
        {
            llimit = mlimit1;
        }
        if (usrId > ulimit2)
        {
            llimit = mlimit2;
        }

        // generate list of movies numbers <= MOVIE_QTY
        movies = get_movies(llimit);
        // generate same amount of random samplings from the possible ratings list
        for (int i=0; i<RATINGS_PER_USER; i++)
        {

            rnmbr = randn (mu, sigma);
            for (int j=0; j<5; j++)
            {
                if (rnmbr < possible_ratings[j])
                {
                    usr_rats[i] = possible_ratings[j];
                    break;
                }
            }
        }
        // print to csv
        for (int i=0; i<RATINGS_PER_USER; i++)
        {
            fprintf(outfile,"%d,%d,%f\n",usrId,*(movies+i),usr_rats[0+i]);
        }

    }


    return 0;
}
