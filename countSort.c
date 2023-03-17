#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <unistd.h>
#include <sched.h>
#include <sys/sysinfo.h>
#include <math.h>
#include <time.h>




void parallelCountSort(int* inputVector, int* countVector, int size, int chunkSize);


int main(int argc, char* argv[])
{

	int threadCount, size;
	int* inputVector;
	int* countVector;
	time_t begin,end;
	threadCount=strtol(argv[1],NULL,10);
	
	printf("Type in array size: \n");
	scanf("%d",&size);


	if (size%threadCount!=0 && threadCount>1){
		printf("Array size MOD number of processes must be zero or number of processes equal to one.\n");
		return 0;	
	}

	
	inputVector= malloc(size*sizeof(int));
	countVector = malloc(size*sizeof(int));
	
	int chunkSize=size/threadCount;
	
	
	for (int i=0; i<size;i++){
		inputVector[i]=rand()%1000;
		countVector[i]=-1;
		//printf("%d.) %d\n",i+1,inputVector[i]); //for testing purposes
	}

	begin=time(NULL);
	
	# pragma omp parallel num_threads(threadCount)	
	parallelCountSort(inputVector,countVector,size,chunkSize);

	end = time(NULL);
	double elapsedTime = end-begin;
	printf("Elapsed time (seconds): %.2f\n",elapsedTime);
	// printf("Vector sorted\n");
	// for(int i=0;i<size;i++)
	// 	printf("%d.) %d\n",i+1,inputVector[i]);


	free(inputVector);
	free(countVector);
	return 0;
}

void parallelCountSort(int* inputVector, int* countVector, int size, int chunkSize){
	
	
	int myID=omp_get_thread_num(), count;
	int start=myID*chunkSize;;
	int end=start+chunkSize;

	for (int i=start;i<end;i++){
		count=0;
		for (int j=0;j<size;j++)
			count+=inputVector[j]<inputVector[i] || (inputVector[j]==inputVector[i] && j<i);
		countVector[count]=inputVector[i];
	}

	# pragma omp barrier //enter section once all threads have finished
	for(int i=0;i<size;i++)
	 	if (countVector[i]!=-1)
	 		inputVector[i]=countVector[i];
}