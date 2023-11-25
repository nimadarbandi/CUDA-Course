/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

// Feel free to use other numbers for best performance
#define TILE_WIDTH 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float *C)
{
    
    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE
    __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];

 
    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    


    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    float Pvalue = 0;

    ///////////////////////////////////////////////////////////////////////////
    
    for (int i = 0; i < (k-1)/TILE_WIDTH + 1; ++i) {
        // Collaborative loading of M and N tiles into shared memory

        if (Row < m && i * TILE_WIDTH + tx < k){
            //int val = Row*k + i*TILE_WIDTH+tx;
            subTileM[ty][tx] = A[Row*k + i*TILE_WIDTH+tx];
        }else{
            subTileM[ty][tx] = 0;
            
        }
        
        if (Col < n && i* TILE_WIDTH + ty < k){
            subTileN[ty][tx] = B[(i*TILE_WIDTH+ty)*n+Col];    
        }else{
           subTileN[ty][tx] = 0;
        }

        __syncthreads();

        if (Row < m && Col < n){
            for (int j = 0; j < TILE_WIDTH; ++j)
                Pvalue += subTileM[ty][j] * subTileN[j][tx];
        }

    ///////////////////////////////////////////////////////////////////////////////////

    __syncthreads();
    }    

    if (Row < m && Col < n){
        C[Row*n+Col] = Pvalue;
    }


}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc, int testRound)
{
    if ((transa != 'N') && (transa != 'n'))
    {
        printf("unsupported value of 'transa'\n");
        return;
    }
    
    if ((transb != 'N') && (transb != 'n'))
    {
        printf("unsupported value of 'transb'\n");
        return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10))
    {
        printf("unsupported value of alpha\n");
        return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10))
    {
        printf("unsupported value of beta\n");
        return;
    }

    // Initialize thread block and kernel grid dimensions ----------------------
    // INSERT CODE HERE
    

        float BlkSize = 16.0;
        
        dim3 dimGrid(ceil(n/BlkSize),ceil(m/BlkSize),1);
        dim3 dimBlk(BlkSize,BlkSize,1);
        //printf("%f",ceil(n/BlkSize));
    for (int i = 0; i < testRound; i++) {
        // Invoke CUDA kernel --------------------------------------------------
        // INSERT CODE HERE
        mysgemm<<<dimGrid,dimBlk>>>(m,n,k,A,B,C);
        cudaDeviceSynchronize();
    }
}
