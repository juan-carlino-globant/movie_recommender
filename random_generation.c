#include <stdlib.h>
#include <stdio.h>
#include <time.h>


int cong_generator()
{
    static int nmbr;
    int a,c,m;
    a = 7;
    c = 2;
    m = 1000;
    nmbr = (a*nmbr + c) % m;
    return nmbr;
}


int* get_movies()
{
    static int movies[30] ;
    int a,c,m,nmbr;
// ver parametros de esta cosa!!!
    nmbr = 13;
    a = 87;
    c = 2;
    m = 1000;
    for (int i=0; i<30; i++)
        {
            nmbr = (a*nmbr + c) % m;
            movies[i] = nmbr + 1;
        }
    return movies;
}





int main()
{
    // open and write header of .csv file
    FILE *outfile;
    outfile = fopen("new_dataset", "w");
    fprintf(outfile,"usrId,movieId,rating\n");

    // declaration of some vars
    int* peliculas;
    float possible_ratings[10] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0};
    float usr_rats[30];

    for (int usrId=1; usrId<1000001; usrId++)
    {
        // generate list of movies numbers <= 1000
        peliculas = get_movies();
        // generate same amount of random samplings from the possible ratings list
        for (int i=0; i< 30; i++)
        {
            usr_rats[i] = possible_ratings[rand()%10];
        }
        // print to csv
        for (int i=0; i<30; i++)
        {
            fprintf(outfile,"%d,%d,%f\n",usrId,*(peliculas+i),usr_rats[0+i]);
        }

    }


    return 0;
}
