/*
*****************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 *****************************************************************************
 */

#define BLOCK_SIZE 512


//sums is the auxilary array to store last element of each block scan
__global__ void scanBlock(float *sums,float *out, float *in, unsigned int in_size){
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(BLOCK_SIZE*2) + tid;
    unsigned int last;
    if (blockIdx.x == gridDim.x-1){
        last = in_size - blockIdx.x*BLOCK_SIZE*2 - 1;
    }else{
        last = 2*BLOCK_SIZE - 1;
    }

    //load from in to Shared mem
//----------------------------------------------
    if(i<in_size){
           sdata[tid] = in[i]; 
    }
    if(i+BLOCK_SIZE<in_size){
           sdata[tid+BLOCK_SIZE] = in[i+BLOCK_SIZE]; 
    }
    
    __syncthreads();
//-----------------------------------------------


     //Reduction
 //----------------------------------------------------
    int stride = 1;
    int index = 0;
    while(stride < 2*BLOCK_SIZE)
    {
    index = (threadIdx.x+1)*stride*2 - 1;
        if(index < 2*BLOCK_SIZE && (index-stride) >=0)
        {
            sdata[index] += sdata[index-stride];
        }
    stride = stride*2;
    __syncthreads();
    }
//-------------------------------------------------------


    // Post Scan
//------------------------------------------------
    stride = BLOCK_SIZE/2;
    while(stride > 0)
    {
    int index = (threadIdx.x+1)*stride*2 - 1;
    if((index+stride) < 2*BLOCK_SIZE)
    {
    sdata[index+stride] += sdata[index];
    }
    stride = stride/2;
    __syncthreads();
    }
//--------------------------------------------------

    //copy shared memory to out
    if(i<in_size){
    out[i] = sdata[tid];
        if(i+BLOCK_SIZE<in_size){
            out[i+BLOCK_SIZE] = sdata[tid+BLOCK_SIZE];
        }
    }

    //copy shared mem to sums
    if (tid == last){
        sums[blockIdx.x] = sdata[tid];
    }else if(tid+BLOCK_SIZE == last){
       sums[blockIdx.x] = sdata[tid+BLOCK_SIZE]; 
    }

}






//Scans the auxilary array of block sums
__global__ void scanSums(float *sums, unsigned int sums_size){
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(BLOCK_SIZE*2) + tid;


    //load from in to Shared mem
//----------------------------------------------
    if(i<sums_size){
        sdata[tid] = sums[i]; 
    }
    if(i+BLOCK_SIZE<sums_size){
        sdata[tid+BLOCK_SIZE] = sums[i+BLOCK_SIZE]; 
    }
    
    __syncthreads();
//-----------------------------------------------


     //Reduction
 //----------------------------------------------------
    int stride = 1;
    int index = 0;
    while(stride < 2*BLOCK_SIZE)
    {
    index = (threadIdx.x+1)*stride*2 - 1;
        if(index < 2*BLOCK_SIZE && (index-stride) >=0)
        {
            sdata[index] += sdata[index-stride];
        }
    stride = stride*2;
    __syncthreads();
    }
//-------------------------------------------------------

///////////////////////////////

    // Post Scan
//------------------------------------------------
    stride = BLOCK_SIZE/2;
    while(stride > 0)
    {
    int index = (threadIdx.x+1)*stride*2 - 1;
    if((index+stride) < 2*BLOCK_SIZE)
    {
    sdata[index+stride] += sdata[index];
    }
    stride = stride/2;
    __syncthreads();
    }
//--------------------------------------------------

    //copy shared to in

    if(i<sums_size){
    sums[i] = sdata[tid];
        if(i+BLOCK_SIZE<sums_size){
            sums[i+BLOCK_SIZE] = sdata[tid+BLOCK_SIZE];
        }
    }

}




//combines sums and output arrays, also adjusts the output to (exclusive|inclusive)
__global__ void adjustScan(bool exclusive,float *sums, float *out, unsigned int sums_size, unsigned int in_size){
    extern __shared__ float sdata[];
    float blockSum = 0;
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(BLOCK_SIZE*2) + tid;

 
    if(i<in_size) sdata[tid] = out[i];
    if(i+BLOCK_SIZE<in_size){sdata[tid+BLOCK_SIZE] = out[i+BLOCK_SIZE];}

    if(blockIdx.x>0){
        blockSum = sums[blockIdx.x-1];
    }
    __syncthreads();
//-----------------------------------------------
    
    if (blockIdx.x > 0){
        
        if(i<in_size) {
            sdata[tid] += blockSum;
            }
        if(i+BLOCK_SIZE<in_size) sdata[tid+BLOCK_SIZE] += blockSum;
    }
    __syncthreads();

 //----------------------------------------------------   
    //copy shared mem to out
    if (exclusive){
        if(i<in_size){
            out[i+1] = sdata[tid];
            if(i+BLOCK_SIZE+1<in_size){
                out[i+BLOCK_SIZE+1] = sdata[tid+BLOCK_SIZE];
            }
            if(i==0) out[0]=0;
        }
    }else{
       if(i<in_size){
            out[i] = sdata[tid];
            if(i+BLOCK_SIZE+1<in_size){
                out[i+BLOCK_SIZE] = sdata[tid+BLOCK_SIZE];
            }
        } 
    }

}





/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void preScan(float *out, float *in, float in_size)
{
    float g_size = ceil(in_size/(2*BLOCK_SIZE));
    int sharedMemSize =  BLOCK_SIZE*2*sizeof(float);
    dim3 dim_grid = dim3(g_size,1,1);
    dim3 dim_block = dim3(BLOCK_SIZE,1,1);
    float *sums;
    cudaError_t cuda_ret;
    cuda_ret = cudaMalloc((void**)&sums, g_size*sizeof(float));
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory sums");

    printf("\n grid dimention: %d,%d,%d \n block dimention %d,%d,%d \n",dim_grid.x,dim_grid.y,dim_grid.z,dim_block.x,dim_block.y,dim_block.z);
    
    //Scans block elements independently and puts the last element in sums array
    scanBlock<<<dim_grid,dim_block,sharedMemSize>>>(sums,out,in,in_size);
    
    //Scan sums (inclusive)
    //Also checkes if we need a nested scan to calculate sums scan for larger input length
    //Nested auxilary array for sums = sums_1, nested sums_1 = sums_2, ... 
    //--------------------
    float sums_1_size = ceil(g_size/(2*BLOCK_SIZE));
    float sums_2_size = ceil(sums_1_size/(2*BLOCK_SIZE));
    printf("sums_1=%f,sums_2=%f",sums_1_size,sums_2_size);

    cudaDeviceSynchronize();

    if(sums_1_size == 1){
        scanSums<<<sums_1_size,BLOCK_SIZE,sharedMemSize>>>(sums,g_size);
    }else if(sums_2_size == 1){
        float *sums_1;
        cudaMalloc((void**)&sums_1, (sums_1_size)*sizeof(float));
        scanBlock<<<sums_1_size,BLOCK_SIZE,sharedMemSize>>>(sums_1,sums,sums,g_size);
        scanSums<<<sums_2_size,BLOCK_SIZE,sharedMemSize>>>(sums_1,sums_1_size);
        adjustScan<<<sums_1_size,BLOCK_SIZE,sharedMemSize+sizeof(float)>>>(false,sums_1,sums,sums_1_size,g_size); 
    }
    //---------------------
    
    cudaDeviceSynchronize();

    //adjust output with sums values
    //---------------------------
    adjustScan<<<dim_grid,dim_block,sharedMemSize+sizeof(float)>>>(true,sums,out,g_size, in_size);

}
