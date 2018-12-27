#include <stdlib.h>
#include <stdio.h>
#include <time.h>


#define MOVIE_QTY           3000
#define USER_QTY            2000000
#define RATINGS_PER_USER    30

char* init_assign_map()
{
    static char movie_assignation[MOVIE_QTY];
    for (int i=0; i<MOVIE_QTY; i++)
    {
        movie_assignation[i] = 0;
    }
    return movie_assignation;
}

int movie_idx(int random)
{
   return (random % MOVIE_QTY) + 1;
}

int* get_movies()
{
    static int movies[RATINGS_PER_USER];
    char* movie_assignation = init_assign_map();

    int nmbr, idx;

    for (int i=0; i<RATINGS_PER_USER; i++)
    {
        do
        {
            nmbr = rand();
            idx = movie_idx(nmbr);
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
    float possible_ratings[10] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0};
    float usr_rats[RATINGS_PER_USER];

    for (int usrId=1; usrId<=USER_QTY; usrId++)
    {
        // generate list of movies numbers <= MOVIE_QTY
        movies = get_movies();
        // generate same amount of random samplings from the possible ratings list
        for (int i=0; i<RATINGS_PER_USER; i++)
        {
            usr_rats[i] = possible_ratings[rand()%10];
        }
        // print to csv
        for (int i=0; i<RATINGS_PER_USER; i++)
        {
            fprintf(outfile,"%d,%d,%f\n",usrId,*(movies+i),usr_rats[0+i]);
        }

    }


    return 0;
}
