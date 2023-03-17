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
#include <stdbool.h>

int matchString(char* text, char* string, int stringLength, int chunkSize, int myID);

int main()
{
    char* buffer; 
    char* recvBuffer;
    char* keyString="she was";
    int stringLength =strlen(keyString);
    char fileName[40];
    long length;

    int totalPs, myID, myCount, totalCount;
    
    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &totalPs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myID); //increments rank and stores it in myID

    
    if (myID==0){
        FILE *fp;
        
        printf("Type in file name: \n");
        scanf("%s",fileName);
        fp = fopen ( fileName, "rb" );
        if( !fp ) perror(fileName),exit(1);

        fseek( fp , 0L , SEEK_END);
        length = ftell( fp );
        rewind( fp );

        fp = fopen ( fileName, "rb" );
        if( !fp ) perror(fileName),exit(1);

        /* allocate memory for entire content */
        buffer = calloc( 1, length+1 );
        if( !buffer ) fclose(fp),fputs("memory alloc fails",stderr),exit(1);

        /* copy the file into the buffer */
        if( 1!=fread( buffer , length, 1 , fp) )
          fclose(fp),free(buffer),fputs("entire read fails",stderr),exit(1);
    }
    
    time_t begin,end;
    begin = time(NULL);

    MPI_Bcast(&length,1,MPI_LONG,0,MPI_COMM_WORLD);
    
    int chunkSize=length/totalPs+(length%totalPs!=0);
    
    recvBuffer= calloc( 1, chunkSize+1);
    
    MPI_Scatter(buffer,chunkSize,MPI_CHAR,recvBuffer,chunkSize,MPI_CHAR,0,MPI_COMM_WORLD);
    if (myID==0)
        free(buffer);
    
    myCount=matchString(recvBuffer,keyString,stringLength,chunkSize,myID);
    free(recvBuffer);
    
    MPI_Reduce(&myCount, &totalCount, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    end = time(NULL);
    
    if (myID==0){
        sleep(1);
        printf("File: %s\nText length: %ld\nString pattern \"%s\" matched %d times in text.\n",fileName, length,keyString, totalCount);
        double elapsedTime = end-begin;
        printf("Elapsed time (seconds): %.2f\n",elapsedTime);
    }
    
    printf("EXITING... Proccess id: %d from core: %d\n", myID, sched_getcpu());
    MPI_Finalize();
    return 0;
}


int matchString(char* text, char* string, int stringLength, int chunkSize, int myID){
    int k=0, count=0, start=0; 

    while (start<chunkSize)
        if (tolower(text[start])==tolower(string[k])){
            if(k==stringLength-1){
                count++;
                k=0;
            }
            else
                k++;
            start++;
        }
        else{
            k=0;
            start++;
        }
    return count;
        
}