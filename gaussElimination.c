#include<stdio.h>
#include <stdlib.h>
#include<math.h>
#include <time.h>
#include <omp.h>
/*******
 Function that performs Gauss-Elimination and returns the Upper triangular matrix and solution of equations:
There are two options to do this in C.
1. Pass the augmented matrix (a) as the parameter, and calculate and store the upperTriangular(Gauss-Eliminated Matrix) in it.
2. Use malloc and make the function of pointer type and return the pointer.
This program uses the first option.
********/
void gaussEliminationLS(int m, int n, double** a, double* x, int threadCount);
void readMatrix(int m, int n, double matrix[m][n]);
void printMatrix(int m, int n, double matrix[m][n]);
void copyMatrix(int m, int n, double** matrix1, double** matrix2);
double** allocateMatrix(int rows, int cols);
 
int main(int argc, char* argv[]){
    int m,n,threadCount=strtol(argv[1],NULL,10);;
   
    printf("Enter the size of the augmeted matrix:\nNo. of rows (m)\n");
    scanf("%d",&m);
    printf("No.of columns (n)\n");
    scanf("%d",&n);

    // if (m%threadCount!=0 && threadCount!=1){
    //     printf("This application is meant to be run with an even number of processes.\n");
    //     return 0;
    // }

    double** a=allocateMatrix(m,n);
    double* x=malloc(m*sizeof(double));
    double** U=allocateMatrix(m,n);

    for (int i=0; i<m;i++)
        for (int j=0;j<n;j++)
            a[i][j]=rand()%1000;
    

    //printf("The original matrix is:\n");
    //printMatrix(m,n,a);
    //copyMatrix(m,n,a,U);
    
    //Perform Gauss Elimination
    time_t begin,end;
    begin = time(NULL);
    gaussEliminationLS(m,n,a,x, threadCount);
    end = time(NULL);
    double elapsedTime = end-begin;
    printf("Elapsed time (seconds): %.2f\n",elapsedTime);
    // printf("\nThe Upper Triangular matrix after Gauss Eliminiation is:\n\n");
    // printMatrix(m,n,U);
    // printf("\nThe solution of linear equations is:\n\n");
    
    // for(int i=0;i<n-1;i++)
    //      printf("x[%d]=\t%lf\n",i+1,x[i]);
    free(x);
    free(a);
    free(U);
}


void gaussEliminationLS(int m, int n, double** a, double* x, int threadCount){
    int i,j,k;
    for(i=0;i<m-1;i++){
        if (i%500==0)
            printf("Solving ...\n");
        //Partial Pivoting
        for(k=i+1;k<m;k++){
            //If diagonal element(absolute vallue) is smaller than any of the terms below it
            if(fabs(a[i][i])<fabs(a[k][i])){
                //Swap the rows
                for(j=0;j<n;j++){                
                    double temp;
                    temp=a[i][j];
                    a[i][j]=a[k][j];
                    a[k][j]=temp;
                }
            }
        }

        double term;

        //Begin Gauss Elimination
        # pragma omp parallel for num_threads(threadCount) private(k,j,term)
        for(k=i+1;k<m;k++){
            term=a[k][i]/ a[i][i];
            for(j=0;j<n;j++){
                a[k][j]=a[k][j]-term*a[i][j];
            }
        }
         
    }
    //Begin Back-substitution
    for(i=m-1;i>=0;i--){
        x[i]=a[i][n-1];
        for(j=i+1;j<n-1;j++){
            x[i]=x[i]-a[i][j]*x[j];
        }
        x[i]=x[i]/a[i][i];
    }
             
}
/*******
Function that reads the elements of a matrix row-wise
Parameters: rows(m),columns(n),matrix[m][n] 
*******/
void readMatrix(int m, int n, double matrix[m][n]){
    int i,j;
    for(i=0;i<m;i++){
        for(j=0;j<n;j++){
            scanf("%lf",&matrix[i][j]);
        }
    } 
}
/*******
Function that prints the elements of a matrix row-wise
Parameters: rows(m),columns(n),matrix[m][n] 
*******/
void printMatrix(int m, int n, double matrix[m][n]){
    int i,j;
    for(i=0;i<m;i++){
        for(j=0;j<n;j++){
            printf("%lf\t",matrix[i][j]);
        }
        printf("\n");
    } 
}
/*******
Function that copies the elements of a matrix to another matrix
Parameters: rows(m),columns(n),matrix1[m][n] , matrix2[m][n]
*******/
void copyMatrix(int m, int n, double** matrix1, double** matrix2){
    int i,j;
    for(i=0;i<m;i++){
        for(j=0;j<n;j++){
            matrix2[i][j]=matrix1[i][j];
        }
    } 
}

double** allocateMatrix(int rows, int cols)
{
        double* arr = malloc(rows*cols*sizeof(double));
        double** matrix = malloc(rows*sizeof(double*));
        int i;
        for(i=0; i<rows; i++)
        {
                matrix[i] = &(arr[i*cols]);
        }
        return matrix;
}