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


void parallelCountSort(int inputVector[], int size, int myID, int chunkSize);


int main()
{

	int totalPs, myID, size;
	int* inputVector;
	
	MPI_Init(NULL,NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &totalPs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myID); //increments rank and stores it in myID
	
	
	if (myID==0){
		printf("Type in array size: \n");
		scanf("%d",&size);

		if (size%totalPs!=0 && totalPs>1){
			printf("Array size MOD number of processes must be zero or number of processes equal to one.\n");
			MPI_Abort(MPI_COMM_WORLD,MPI_ERR_COUNT);
		}
	}

	


	double begin,end;
	begin = MPI_Wtime();


	MPI_Bcast(&size,1,MPI_INT,0,MPI_COMM_WORLD);
	inputVector= (int*)malloc(size*sizeof(int));
	int chunkSize=size/totalPs;
	
	if (myID==0){
		for (int i=0; i<size;i++){
			inputVector[i]=rand()%100;
			//printf("%d.) %d\n",i+1,inputVector[i]); //for testing purposes
		}
	}

	
	MPI_Bcast(inputVector,size,MPI_INT,0,MPI_COMM_WORLD);
	parallelCountSort(inputVector,size,myID,chunkSize);

	if (myID>0){
		MPI_Send(inputVector,size,MPI_INT,0,0,MPI_COMM_WORLD);
	}
	else{
		int* tempVector;
		tempVector= (int*)malloc(size*sizeof(int));
		

		for (int q=1;q<totalPs;q++){
			MPI_Recv(tempVector, size, MPI_INT, MPI_ANY_SOURCE , 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			for(int i=0;i<size;i++)
				if (tempVector[i]!=-1)
					inputVector[i]=tempVector[i];
		}
		free(tempVector);

		end = MPI_Wtime();
		double elapsedTime = end-begin;
		printf("Elapsed time (seconds): %.2f\n",elapsedTime);
		// for(int i=0;i<size;i++)
		// 	printf("%d.) %d\n",i+1,inputVector[i]);

	}

	printf("EXITING... Proccess id: %d from core: %d\n", myID, sched_getcpu());
	MPI_Finalize();
	return 0;
}

void parallelCountSort(int inputVector[], int size, int myID, int chunkSize){

	int* temp = malloc(size*sizeof(int));
	memset(temp,-1,sizeof(int)*size);
	int count;
	int start=myID*chunkSize;
	int end=start+chunkSize;


	for (int i=start;i<end;i++){
		count=0;
		for (int j=0;j<size;j++)
			count+=inputVector[j]<inputVector[i] || (inputVector[j]==inputVector[i] && j<i);
		temp[count]=inputVector[i];

	}
	memcpy(inputVector,temp,size*sizeof(int));
	free(temp);
}
