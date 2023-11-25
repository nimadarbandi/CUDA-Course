/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#define BlkSize 16.0

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float *C)
{

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE

    int row = threadIdx.y+blockIdx.y*blockDim.y;
    int col = threadIdx.x+blockIdx.x*blockDim.x;
 
    if(row < m && col < n){
        float p_value = 0;
        //printf("%f",p_value);
        for (int i=0; i<k ; i++){
            //printf("A: %f,B: %f",A[row*k+i],B[col+i*n]);
            p_value += A[row*k+i]*B[col+i*n];
        }

        C[row*n+col] = p_value;
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
