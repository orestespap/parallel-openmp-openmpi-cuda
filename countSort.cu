#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
 
// CUDA kernel. Each thread takes care of one element of c
__global__ void countSort(int* inputVector, int* countVector, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    // Make sure we do not go out of bounds
    if (id < n){
        int count=0;
        for (int j=0;j<n;j++)
            count+= inputVector[j]<inputVector[id] || (inputVector[j]==inputVector[id] && j<id);
        countVector[count]=inputVector[id];
        //printf("My block id: %d, My thread id: %d, My block's x dimension: %d My warp: %d\n",blockIdx.x,threadIdx.x,blockDim.x,threadIdx.x/32);
    }
}

__global__ void updateInputVector(int* inputVector, int* countVector, int n)
{
    
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    if (id<n)
        inputVector[id]=countVector[id];
}
 
int main( int argc, char* argv[] )
{
    // Size of vectors
    int n, blockSize, gridSize;
    printf("Type in array size: \n");
    scanf("%d",&n);
    printf("Problem size: %d\n",n);
    
    // Number of threads in each thread block
    printf("Type in block size: \n");
    scanf("%d",&blockSize);
    if (blockSize>1024){
        printf("Number of threads/block must be less than 1024!\nExiting ...\n");
        return 0;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device name: %s\n", prop.name);
 
    // Host input vectors
    int *h_inputVector;
    int *h_countVector;
    time_t begin,end;
 
    // Device input vectors
    int *d_inputVector;
    int *d_countVector;
 
    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(int);
 
    // Allocate memory for each vector on host
    h_inputVector = (int*)malloc(bytes);
    h_countVector = (int*)malloc(bytes);
    
    for(int i = 0; i < n; i++ ){
        h_inputVector[i] = rand()%100000;
        h_countVector[i] =-1;
    }  

    // Number of thread blocks in grid
    gridSize = (int)ceil((float)n/blockSize);

    printf("Random vectors created. Beginning calcualtions!\nThreads per block: %d\nTotal blocks in grid: %d\n",blockSize,gridSize);
   
    begin=time(NULL);
    
    // Allocate memory for each vector on GPU
    cudaMalloc(&d_inputVector, bytes);
    cudaMalloc(&d_countVector, bytes);
 
    // Copy host vectors to device
    cudaMemcpy(d_inputVector, h_inputVector, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_countVector, h_countVector, bytes, cudaMemcpyHostToDevice);
 
    // Execute the kernel
    countSort<<<gridSize, blockSize>>>(d_inputVector, d_countVector, n);
    updateInputVector<<<gridSize, blockSize>>>(d_inputVector, d_countVector, n);


    // Copy array back to host
    cudaMemcpy(h_inputVector, d_inputVector, bytes, cudaMemcpyDeviceToHost );

    end = time(NULL);
    double elapsedTime = end - begin;
    printf("Elapsed time (seconds): %.2f\n",elapsedTime);
 
    //for (int i=n-500;i<n;i++)
        //printf("%d.) %d\n",i, h_inputVector[i]);
    
    // Release device memory
    cudaFree(d_inputVector);
    cudaFree(d_countVector);
 
    // Release host memory
    free(h_inputVector);
    free(h_countVector);
 
    return 0;
}