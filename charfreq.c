#include<stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <ctype.h>

int main(int argc, char* argv[])
{   

    char* buffer; 
    char keyChar='a';
    char fileName[40];
    long length;
    int threadCount=strtol(argv[1],NULL,10),count=0; 
    
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
    
  
    time_t begin,end;
    begin = time(NULL);
    

   # pragma omp parallel for num_threads(threadCount)\
        reduction(+: count)
        for (int i=0;i<length;i++)
            if (buffer[i]==keyChar || tolower(buffer[i])==keyChar)
                count++;
    free(buffer);
    
    end = time(NULL);
    
    printf("Text length: %ld\nChar '%c' frequency in %s: %d\n",length,keyChar, fileName, count);
    double elapsedTime = end-begin;
    printf("Elapsed time (seconds): %.2f\n",elapsedTime);
    
    return 0;
}