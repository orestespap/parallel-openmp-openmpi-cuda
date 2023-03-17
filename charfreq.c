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

int charFrequency(char* buffer, char key, int length);

int main()
{
    char* buffer; 
    char* recvBuffer;
    char keyChar='a';
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
    int chunkSize=length/totalPs + (length%totalPs!=0);
    
    
    recvBuffer= calloc( 1, chunkSize+1);
    
    MPI_Scatter(buffer,chunkSize,MPI_CHAR,recvBuffer,chunkSize,MPI_CHAR,0,MPI_COMM_WORLD);
    if (myID==0)
        free(buffer);
    
    myCount=charFrequency(recvBuffer,keyChar,chunkSize);
    free(recvBuffer);
    
    MPI_Reduce(&myCount, &totalCount, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    end = time(NULL);
    
    if (myID==0){
        sleep(1);
        printf("Text length: %ld\nChar '%c' frequency in %s: %d\n",length,keyChar, fileName, totalCount);
        double elapsedTime = end-begin;
        printf("Elapsed time (seconds): %.2f\n",elapsedTime);
    }
    
    printf("EXITING... Proccess id: %d from core: %d\n", myID, sched_getcpu());
    MPI_Finalize();
    return 0;
}


int charFrequency(char* buffer, char key, int length){

    int count=0;
    for (int i=0;i<length;i++)
        if (buffer[i]==key || tolower(buffer[i])==key)
            count++;
    return count;

}