#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

#define N 20
#define rows 20
#define cols 20
#define input_n 20

void addMatrix(float *H_x, float *H_y, float *H_z) {

    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            // H_z[i][j] = H_x[i][j] + H_y[i][j];
            int idx = row * cols + col;
            H_z[idx] = H_x[idx] + H_y[idx];
        }
    }
}

__global__ void addMatrix2D(float *D_x, float *D_y, float *D_z) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if((Row < N) && (Col < N)) {
        D_z[Row * N + Col] = D_x[Row * N + Col] + D_y[Row * N + Col];
    }
}

int main() {
    float *H_x, *H_y, *H_z;
    float *D_x, *D_y, *D_z;

    H_x = (float*)malloc(N * N * sizeof(float));
    H_y = (float*)malloc(N * N * sizeof(float));
    H_z = (float*)malloc(N * N * sizeof(float));
    

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // H_x[i][j] = 1;
            // H_y[i][j] = -1;
            H_x[i * N + j] = i*N+j;
            H_y[i * N + j] = 1;
       }
    }

    H_z = (float*)malloc(N * N * sizeof(float));

    cudaMalloc((void**) &D_x, N * N * sizeof(float));
    cudaMalloc((void**) &D_y, N * N * sizeof(float));
    cudaMalloc((void**) &D_z, N * N * sizeof(float));

    cudaMemcpy(D_x, H_x,  N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(D_y, H_y, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(D_z, H_z,  N * N * sizeof(float), cudaMemcpyHostToDevice);



    addMatrix(H_x, H_y, H_z);

    //  for (int i = 0; i < N*N; i++) {
    //     if (i % 20 == 0) {
    //         printf("\n");
    //     }
    //     printf("%f ", D_z[i]);
    // }

    // printf("\n");

    int thread_per_block_x = 16;
    int thread_per_block_y = 16;
    
    int num_block_x = ceil((float)input_n / thread_per_block_x);
    int num_block_y = ceil((float)input_n / thread_per_block_y);

    dim3 dimGrid(num_block_x, num_block_y, 1);
    dim3 dimBlock(thread_per_block_x,thread_per_block_y, 1);
    
    addMatrix2D<<<dimGrid, dimBlock>>>(D_x, D_y, D_z);
    

    cudaMemcpy(H_z, D_z, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < N*N; i++) {
        if (i % 20 == 0 && (i != 0)) {
            printf("\n");
        }
        printf("%f ", H_z[i]);
    }

    // for (int i = 0; i < N*N; i++) {
    //     if (i % 20 == 0) {
    //         printf("\n");
    //     }
    //     printf("%f ", H_z[i]);
    // }

    free(H_x);
    free(H_y);
    free(H_z);

    cudaFree(D_x);
    cudaFree(D_y);
    cudaFree(D_z);
    return 0;
}