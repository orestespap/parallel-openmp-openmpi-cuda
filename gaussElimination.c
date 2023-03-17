#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <unistd.h>
#include <sched.h>
#include <sys/sysinfo.h>
#include <math.h>
#include <time.h>
#include <ctype.h>


void printMatrix(int m, int n, double* matrix);
void gaussElimination(double* b, int size, int elementsPerRow, int pivotColumnIndex, double pivotRow[elementsPerRow]);

int main()
{   

    //cpus: 1, 3000x3000, -> 128 seconds
    //cpus: 4 3000x3000, -> 62 seconds
      
    int m,n,modulo, div, rowsLeft, totalPs, myID, print;
    double temp, begin,end, term;
    double* b;
    double* a;
    double* pivotRow;
    double* x;

    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &totalPs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myID); //increments rank and stores it in myID
    int counts[totalPs], displacements[totalPs];

    
    if (myID==0){
        printf("Enter the size of the augmeted matrix:\nNo. of rows (m)\n");
        scanf("%d",&m);
        printf("No. of columns: \n");
        scanf("%d",&n);
        printf("Print results? (No: 0, Yes:1 )\n");
        scanf("%d",&print);

        a=malloc(m*n*sizeof(double));
        x=malloc(m*sizeof(double));
        
        for (int i=0; i<m;i++)
            for (int j=0;j<n;j++)
                a[i*n+j]=rand()%1000;

        
        if (print){
            printf("The original matrix is:\n");
            printMatrix(m,n,a);
        }
    }

    begin = MPI_Wtime();
   
    MPI_Bcast(&m,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
    if (myID>0)
        pivotRow=malloc(n*sizeof(double));

    for(int i=0;i<m-1;i++){
        
        if (myID==0)
            //Partial Pivoting
            for(int k=i+1;k<m;k++){
                //If diagonal element(absolute vallue) is smaller than any of the terms below it
                if(fabs(a[i*n+i])<fabs(a[k*n+i])){
                    //Swap the rows
                    for(int j=0;j<n;j++){ 
                        temp=a[i*n+j];
                        a[i*n+j]=a[k*n+j];
                        a[k*n+j]=temp;
                    }
                }
            }

        //Begin Gauss Elimination

        rowsLeft= m-1-i;

        if (rowsLeft>=totalPs){

            if (myID==0){
                if (i%500==0)
                    printf("Solving ...\n");
               
                pivotRow=&a[i*n]; 
                //instead of copying the Pivot Row to a new vector, point Pivot Row to the first element of the Pivot Row, and send
                //the next n*sizeof(double) bytes to the other n-1 processes

                
                modulo=rowsLeft%totalPs;
                div=rowsLeft/totalPs;
                displacements[0]=(i+1)*n;
                counts[0]=div*n;
                if(modulo==0)
                    for(int q=1;q<totalPs;q++){
                        displacements[q]= displacements[0]+q*counts[q-1];
                        counts[q]=counts[0];
                    }
                else{
                    for(int q=1;q<totalPs-1;q++){
                        displacements[q]= displacements[0]+q*counts[q-1];
                        counts[q]=counts[0];
                    }
                    int q=totalPs-1;
                    displacements[q]= displacements[0]+q*counts[q-1];
                    counts[q]=(div+modulo)*n;

                }
                
            }

            
            MPI_Bcast(pivotRow,n,MPI_DOUBLE,0,MPI_COMM_WORLD);
            MPI_Bcast(counts,totalPs,MPI_INT,0,MPI_COMM_WORLD);
            
            b=malloc(counts[myID]*sizeof(double));
            

            MPI_Scatterv(a,counts,displacements,MPI_DOUBLE,b,counts[myID],MPI_DOUBLE,0,MPI_COMM_WORLD);
            gaussElimination(b,counts[myID],n,i,pivotRow);
            

            MPI_Gatherv(b,counts[myID],MPI_DOUBLE,a,counts,displacements,MPI_DOUBLE,0,MPI_COMM_WORLD);
            
            free(b);
            
        }
        else{

            if (myID==0){
                for(int k=i+1;k<m;k++){
                    term=a[k*n+i]/ a[i*n+i];
                    for(int j=0;j<n;j++)
                        a[k*n+j]=a[k*n+j]-term*a[i*n+j];
                }
            }
            else{
                printf("EXITING... Proccess id: %d from core: %d\n", myID, sched_getcpu());
                MPI_Finalize();
                return 0;
            }
        }
         
    }
 
    
    if (myID==0){
        //Begin Back-substitution
        for(int i=m-1;i>=0;i--){
            x[i]=a[i*n+n-1];
            for(int j=i+1;j<n-1;j++)
                x[i]=x[i]-a[i*n+j]*x[j];
            
            x[i]=x[i]/a[i*n+i];
        }

        end = MPI_Wtime();
        double elapsedTime = end-begin;
        sleep(2);
        if (print){
            printf("\nThe Upper Triangular matrix after Gauss Eliminiation is:\n\n");
            printMatrix(m,n,a);
            printf("\nThe solution of linear equations is:\n\n");
            
            for(int i=0;i<n-1;i++)
                printf("x[%d]=\t%lf\n",i+1,x[i]);
        }
        printf("Problem size: %dx%d\nElapsed time (seconds): %.2f\n",m,n,elapsedTime);
        free(a);
    }
    else
        free(pivotRow);
    
    printf("EXITING... Proccess id: %d from core: %d\n", myID, sched_getcpu());
    MPI_Finalize();
    return 0;
}

void gaussElimination(double* b, int size, int elementsPerRow, int pivotColumnIndex, double pivotRow[elementsPerRow]){
    double term;
    int k;
    for(int j=0;j<size;j++){
        if (j%elementsPerRow==0){
            term=b[pivotColumnIndex+j]/pivotRow[pivotColumnIndex];
            k=0;
        }
        b[j]=b[j]-term*pivotRow[k];
        k+=1; 
    }        
}


void printMatrix(int m, int n, double* matrix){
    int i,j;
    for(i=0;i<m;i++){
        for(j=0;j<n;j++){
            printf("%lf\t",matrix[i*n+j]);
        }
        printf("\n");
    } 
}
