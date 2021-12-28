# include <stdio.h>
# include <stdlib.h>
# include <math.h>

# define INPUTSHAPE 3 * 299 * 299
# define OUTPUTSHAPE 1000
# define TESTNUM 10
# define ITERNUM 1
double inputArr[TESTNUM][INPUTSHAPE];
double benchOutArr[TESTNUM][OUTPUTSHAPE];

void formatInput(char *filename, char *o_filename)
{
    FILE *fp = NULL;
    fp = fopen(filename, "r");
    for (int i = 0; i < TESTNUM; i++)
        for (int j = 0; j < INPUTSHAPE; j++) 
            fscanf(fp, "%lf", &inputArr[i][j]);
    fclose(fp);
    fp = fopen(o_filename, "w");
    fprintf(fp, "{\n");
    for (int i = 0; i < TESTNUM; ++ i) {
        fprintf(fp, "\"test%d\":[", i);
        for (int j = 0; j < INPUTSHAPE; ++ j)
            fprintf(fp, "%.10lf%c", inputArr[i][j], ((j != INPUTSHAPE - 1) ? ',' : ']'));
        if (i != TESTNUM - 1) fprintf(fp, ",");
        fprintf(fp, "\n");
    }
    fprintf(fp, "}\n");
}

void formatOutput(char *filename, char *o_filename)
{
    FILE *fp = NULL;
    fp = fopen(filename, "r");
    for (int i = 0; i < TESTNUM; i++)
        for (int j = 0; j < OUTPUTSHAPE; j++)
            fscanf(fp, "%lf", &benchOutArr[i][j]);
    fclose(fp);    
    fp = fopen(o_filename, "w");
    fprintf(fp, "{\n");
    for (int i = 0; i < TESTNUM; ++ i) {
        fprintf(fp, "\"test%d\":[", i);
        for (int j = 0; j < OUTPUTSHAPE; ++ j)
            fprintf(fp, "%.10lf%c", benchOutArr[i][j], ((j != OUTPUTSHAPE - 1) ? ',' : ']'));
        if (i != TESTNUM - 1) fprintf(fp, ",");
        fprintf(fp, "\n");
    }
    fprintf(fp, "}\n");
}

int main()
{   
    formatInput("../data/inceptionInput.txt", "../data/inceptionInput.json"); 
    formatOutput("../data/inceptionOutput.txt", "../data/inceptionOutput.json");
}