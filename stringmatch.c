#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <unistd.h>
#include <sched.h>
#include <sys/sysinfo.h>
#include <math.h>
#include <time.h>
#include <ctype.h>


int matchString(char* text, char* string, int stringLength, int start, int end);

int main(int argc, char* argv[])
{
    char* buffer; 
    char* keyString="she was";
    char fileName[40];
    long length;
    int threadCount=strtol(argv[1],NULL,10), totalCount=0;

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
    
    int stringLength =strlen(keyString);
    time_t begin,end;
    begin = time(NULL);

    if (threadCount==1)
        totalCount=matchString(buffer,keyString,stringLength,0,length);
    else{
        int chunkSize=length/threadCount;
        #pragma omp parallel num_threads(threadCount)\
            reduction(+: totalCount)
        {
            int myID=omp_get_thread_num();
            int start=myID*chunkSize; 
            int end=start+chunkSize;
            if (myID==threadCount-1)
                end+=length%threadCount;
            
            totalCount=matchString(buffer,keyString,stringLength,start,end);
        }
    }

    free(buffer);    
   
    end = time(NULL);
    
    printf("File: %s\nText length: %ld\nString pattern \"%s\" matched %d times in text.\n",fileName, length,keyString, totalCount);
    double elapsedTime = end-begin;
    printf("Elapsed time (seconds): %.2f\n",elapsedTime);
    return 0;
}


int matchString(char* text, char* string, int stringLength, int start, int end){
    int k=0, count=0, l=start;

    while (l<end)
        if (tolower(text[l])==tolower(string[k])){
            if(k==stringLength-1){
                count++;
                k=0;
            }
            else
                k++;
            l++;
        }
        else{
            k=0;
            l++;
        }

    return count;
        
}