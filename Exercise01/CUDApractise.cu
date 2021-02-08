#include<cuda_runtime.h>
#include<cstdef>
#include<sys/time.h>

#include<iostream>
#include<vector>

double getSeconds()
{
	struct timeval to;
	gettimeofday(&tp, NULL);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1e-6);
}

void checkError(cudaError_t err)
{
	if(err != cudaSuccess)
	{
		std::cout<<cudaGetErrorString(err)<<std::endl;
		exit(-1);
	}
}

// kernel

__global__ void sum(int *A, int *B, int *C, long long N)
{
	const long long idx = blockId.x * blockDim.x + threadIdx.x ;
	c[idx] = A[idx];
}
	
int main()
{

	const long long nElem = 1 << 30 ;   // should be less than total mem of grapics card 4gb for current
	std::vector<int> A(nElem,1);
	std::vector<int> B(nElem,1);
	std::vector<int> C(nElem,0);

	const long long nBytes = nElem * sizeof(int);
	std::cout<< nBytes* 1e-6 << std::endl;

	int* d_A;
	int* d_B;
	int* d_C;

	checkError(cudaMalloc(&d_A,nBytes));
        checkError(cudaMalloc(&d_B,nBytes));
        checkError(cudaMalloc(&d_C,nBytes));

	checkError(cudaMemcpy(d_A,&A[0],nBytes,cudaMemcpyHostToDevice));
        checkError(cudaMemcpy(d_B,&B[0],nBytes,cudaMemcpyHostToDevice));
 
	const double start = getSeconds();

       
	sum<<< (1<<14 ), (1<<7) >>>(d_A, d_B,d_C.nElem);    // the no should be less than no of thread 1024
	checkError(cudaPeekAtLastError() );
	checkError(cudaDeviceSynchronize() );               // wait untill all process is finished

	const double stop = getSeconds();


	std::cout<<"run time"<< (start-stop) * 1e3<<"ms"<<std::endl ; 


	checkError(cudaMemcpy(&C[0],d_A,nBytes,cudaMemcpyDeviceToHost));

	for(long long i=0;i<nElem;i++)
	{
		if(C.at(i) != 1)
		{
			std::cout<<"error:"<<i<<" "<<C.at(i)<<std::endl;
			exit(-1);
		}
	}

	checkError(cudaFree(d_A));
        checkError(cudaFree(d_B));
        checkError(cudaFree(d_C));

	return EXIT_SUCESS;
}



 
