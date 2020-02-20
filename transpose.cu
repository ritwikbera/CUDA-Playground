
#include <iostream>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

#define TILE_WIDTH 100 // for shared kernel

// CPU computation

__host__ void cpuTranspose(float* M, float* R, int dim1, int dim2){

	for (int i = 0; i < dim1; i++){
		for (int j = 0; j < dim2; j++){
			R[j*dim2 + i] = M[i*dim2 + j];
		}
	}
}

// NAIVE APPROACH - global memory access only
__global__ void transpose(float* M, float* R, int dim1, int dim2){

	int column = blockDim.x* blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;

	if(column < dim2 && row < dim1){
		R[column*dim2 + row] = M[column + row*dim2];
	}

	// __syncthreads() not needed above as only non-conflicting read/write operations occuring
}

// SHARED MEM APROACH - use shared memory
__global__ void sharedMem_transpose(float* M, float* R, int dim1, int dim2){

	// fill data into shared memory
	__shared__ float M_Shared[TILE_WIDTH][TILE_WIDTH];

	int column = TILE_WIDTH * blockIdx.x + threadIdx.x;
	int row = TILE_WIDTH * blockIdx.y + threadIdx.y;
	
	int index_in = row*dim2 + column;
	int index_out = column*dim2 + row;


	if(row < dim1 && column < dim2 && index_in < dim1*dim2){
		M_Shared[threadIdx.y][threadIdx.x] = M[index_in];
	}
	__syncthreads(); // transfer to global mem after all threads finish computation

	if(row < dim1 && column < dim2 && index_out < dim1*dim2){
		R[index_out] = M_Shared[threadIdx.y][threadIdx.x];
	}
}

int main(void){

	int const dim1 = 3000;
	int const dim2 = 3000;

	float *M_h;
	float *R_h;
	float *M_d;
	float *R_d;

	size_t size = dim1*dim2*sizeof(float);

	cudaMallocHost((float**)&M_h,size); //page locked host mem allocation
	R_h = (float*)malloc(size);
	cudaMalloc((float **)&M_d, size);


	// init matrix
	for (int i = 0; i < dim1*dim2; ++i) {
		M_h[i]=i;
	}

	// asynchronous mem copies can only happen from pinned memory (direct RAM transfer)
	// CPU cannot be held up in mem copies for async copy
	cudaMemcpyAsync(M_d,M_h,size,cudaMemcpyHostToDevice);
	cudaMalloc((float**)&R_d,size);
	cudaMemset(R_d,0,size);


	// init kernel

	// TILE_WIDTH is chosen as per shared memory availaibility in a block
	// TILE_WIDTH doesn't have much use for naive access and we could lump computatiom into fewer blocks.
	int threadNumX = TILE_WIDTH;
	int threadNumY = TILE_WIDTH;
	int blockNumX = dim1 / TILE_WIDTH + (dim1%TILE_WIDTH == 0 ? 0 : 1 );
	int blockNumY = dim2 / TILE_WIDTH + (dim2%TILE_WIDTH == 0 ? 0 : 1 );

	dim3 blockSize(threadNumX,threadNumY);
	dim3 gridSize(blockNumX, blockNumY);

	// CUDA TIMER to Measure the performance
	cudaEvent_t start_naive, start_shared, stop_shared, stop_naive;
	
	float elapsedTime1, elapsedTime2;
	cudaEventCreate(&start_naive);
	cudaEventCreate(&stop_naive);
	cudaEventCreate(&start_shared);
	cudaEventCreate(&stop_shared);

	cudaEventRecord(start_naive, 0);

	transpose<<<gridSize,blockSize>>>(M_d,R_d,dim1,dim2);

	cudaEventRecord(stop_naive, 0);
	cudaEventSynchronize(stop_naive);
	cudaEventElapsedTime(&elapsedTime1, start_naive, stop_naive);

	cudaEventRecord(start_shared,0);

	sharedMem_transpose<<<gridSize,blockSize>>>(M_d,R_d,dim1,dim2);

	cudaEventRecord(stop_shared, 0);
	cudaEventSynchronize(stop_shared);
	cudaEventElapsedTime(&elapsedTime2, start_shared, stop_shared);

	clock_t begin = clock();

    cpuTranspose(M_h, R_h, dim1, dim2); //matrix multiplication on cpu
    
    clock_t end = clock();
    double elapsedTime3 = (double)1000*(end - begin) / CLOCKS_PER_SEC;

	cout <<"Time for the NAIVE kernel: "<<elapsedTime1<<" ms"<<endl;
	cout <<"Time for the SHARED kernel: "<<elapsedTime2<<" ms"<<endl;
	cout <<"Time for the CPU code "<<elapsedTime3<<" ms"<<endl;


	cudaFreeHost(M_h);
	free(R_h);
	cudaFree(R_d);
	cudaFree(M_d);
	return 0;
}
