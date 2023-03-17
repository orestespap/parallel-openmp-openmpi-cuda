#include<stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctype.h>

__global__ void charFreq(char* buffer,int* countVector, char lowerChar, char upperChar, long length){
    
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    if( id<length){
         countVector[id]= buffer[id]==lowerChar || buffer[id]==upperChar;
    }
}

int main(int argc, char* argv[])
{   

    char* h_buffer;
    int* h_countVector; 
    char keyChar='A';
    char upperChar = tolower(keyChar);
    char fileName[40];
    long length;
    int blockSize,gridSize;
     
    FILE *fp;
    
    printf("Type in file name: \n");
    scanf("%s",fileName);
    fp = fopen ( fileName, "rb" );
    if( !fp ) perror(fileName),exit(1);

    // Number of threads in each thread block
    printf("Type in block size: \n");
    scanf("%d",&blockSize);
    if (blockSize>1024){
        printf("Number of threads/block must be less than 1024!\nExiting ...\n");
        return 0;
    }

    fseek( fp , 0L , SEEK_END);
    length = ftell( fp );
    rewind( fp );

    fp = fopen ( fileName, "rb" );
    if( !fp ) perror(fileName),exit(1);

    /* allocate memory for entire content */
    h_buffer =(char*) calloc( 1, length+1 );
    if( !h_buffer ) fclose(fp),fputs("memory alloc fails",stderr),exit(1);

    /* copy the file into the buffer */
    if( 1!=fread( h_buffer , length, 1 , fp) )
      fclose(fp),free(h_buffer),fputs("entire read fails",stderr),exit(1);
    
    
    gridSize = (int)ceil((float)length/blockSize);
    time_t begin,end;
    begin = time(NULL);
    h_countVector=(int*)malloc(length*sizeof(int));

    char *d_buffer;
    int *d_countVector;
    cudaMalloc(&d_buffer, length*sizeof(char));
    cudaMalloc(&d_countVector, length*sizeof(int));
    

    cudaMemcpy(d_buffer, h_buffer, length*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_countVector, h_countVector, length*sizeof(int), cudaMemcpyHostToDevice);

    charFreq<<<gridSize, blockSize>>>(d_buffer, d_countVector, keyChar, upperChar, length);

    cudaMemcpy(h_countVector, d_countVector, length*sizeof(int), cudaMemcpyDeviceToHost);

    int count=0;
    for (int i=0;i<length;i++)
        count+=h_countVector[i];

    free(h_buffer);
    free(h_countVector);
    cudaFree(d_buffer);
    cudaFree(d_countVector);
    
    end = time(NULL);
    
    printf("Text length: %ld\nChar '%c' frequency in %s: %d\n",length,keyChar, fileName, count);
    double elapsedTime = end-begin;
    printf("Elapsed time (seconds): %.2f\n",elapsedTime);
    
    return 0;
}