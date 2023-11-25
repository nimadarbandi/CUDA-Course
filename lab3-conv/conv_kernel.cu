/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

__constant__ float M_c[FILTER_SIZE][FILTER_SIZE];

__global__ void convolution(Matrix N, Matrix P)
{
    /********************************************************************
    Determine input and output indexes of each thread
    Load a tile of the input image to shared memory
    Apply the filter on the input image tile
    Write the compute values to the output image at the correct indexes
    ********************************************************************/

    // INSERT KERNEL CODE HERE

    // shared tile with space for both halos
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * TILE_WIDTH + ty;
    int col_o = blockIdx.x * TILE_WIDTH + tx;
    int row_i = row_o - (FILTER_SIZE/2); // MASK_WIDTH / 2
    int col_i = col_o - (FILTER_SIZE/2); // (radius in prev. code)
    float Pvalue = 0.0f;
    if((row_i >= 0) && (row_i < P.height) && (col_i >= 0) && (col_i < P.width)) {
        tile[ty][tx] = N.elements[row_i*N.width + col_i];
    }
    else {
        tile[ty][tx] = 0.0f;
    }
    __syncthreads(); // wait for tile

    if(ty <TILE_WIDTH && tx <TILE_WIDTH){
        for(int i = 0; i < FILTER_SIZE; i++) {
            for(int j = 0; j < FILTER_SIZE; j++) {
                Pvalue += M_c[i][j] * tile[i+ty][j+tx];
            }
        }
        if(row_o < P.height && col_o < P.width){
            P.elements[row_o * P.width + col_o] = Pvalue;
        }
    }    
}
