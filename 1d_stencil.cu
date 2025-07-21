#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define N 1000000
#define R 3
#define BLOCK_SIZE 512

void stencil(int *in, int *out, int n, int r)
{
    for (int i = R; i < (N - R); i++)
    {   
        int result = 0;
        for (int j = -R; j <= R; j++)
        {
            result += in[i+j];
        }
        out[i] = result;

        // printf("%d ", answer[i]);
    }
}

__global__ void CUDAstencil(int *d_in, int *d_out, int n, int r)
{
    int idx = (blockDim.x * blockIdx.x) + threadIdx.x;

    if (idx < n)
    {
        int result = 0;

        if (idx < r || idx >= n - r)
        {
            d_out[idx] = d_in[idx];
        }
        else
        {
            for (int i = -r; i <= r; i++)
            {
                result += d_in[idx + i];
            }

            d_out[idx] = result;
        }
    }
}

__global__ void fast_stencil(int *d_in, int *d_out, int n)
{

    __shared__ int temp[BLOCK_SIZE + (2 * R)];

    int g_idx = (blockDim.x * blockIdx.x) + threadIdx.x;
    int l_idx = threadIdx.x + R;

    if (g_idx < n) {
        temp[l_idx] = d_in[g_idx];
    }
    
    if (g_idx >= R && g_idx + blockDim.x < N) {

        if (threadIdx.x < R)
        {
            temp[l_idx - R] = d_in[g_idx - R];
            temp[l_idx + blockDim.x] = d_in[g_idx + blockDim.x];
            
        }
    }

    __syncthreads();

    if (g_idx < n)
    {
        int result = 0;
        for (int i = -R; i <= R; i++)
        {
            result += temp[l_idx + i];
        }
        d_out[g_idx] = result;
    }

    // if (threadIdx.x == 0) {
    //     printf("thead 0 print temp in fast_stencil: [");
    //     for (int i = 0; i < N + (2 * R); i++)
    //     {
    //         printf("%d ", temp[i]);
    //     }

    //     printf("]\n");
    // }
}

int main()
{

    clock_t start_time, end_time;
    double cpu_time_used;

    cudaEvent_t start, stop;
    

    int *h_in, *h_out;

    // Declare variables in vanilla C
    h_in = (int *)malloc(sizeof(int) * N);
    h_out = (int *)malloc(sizeof(int) * N);

    for (int i = 0; i < N; i++)
    {
        h_in[i] = i + 1;
    }
    int *d_in, *d_out;

    // Declare variable in CUDA C
    cudaMalloc((void **)&d_in, sizeof(int) * N);
    cudaMalloc((void **)&d_out, sizeof(int) * N);

    cudaMemcpy(d_in, h_in, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_out, sizeof(int) * N, cudaMemcpyHostToDevice);

    // printf("Array: [ ");
    // for (int i = 0; i < N; i++)
    // {
    //     printf("%d ", h_in[i]);
    // }
    // printf("]\n");

    start_time = clock();
    stencil(h_in, h_out, N, R);
    end_time = clock();

    cpu_time_used = ( (double) (end_time - start_time) ) / CLOCKS_PER_SEC;
    printf("Execution time for CPU stencil: %lf second\n", cpu_time_used);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    CUDAstencil<<<(N+ BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_in, d_out, N, R);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("Execution time for CUDAstencil: %f ms\n", elapsed);


    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    fast_stencil<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_in, d_out, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop); 
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("Execution time for fast_stencil (shared memory): %f ms\n", elapsed);

    // printf("h_out: [");
    // for (int i = 0; i < N; i++)
    // {
    //     printf("%d ", h_out[i]);
    // }
    // printf("]\n");

    // for (int i = 0; i < N; i++) {
    //     printf("%d ", h_out[i]);
    // }

    cudaFree(d_in);
    cudaFree(d_out);

    free(h_in);
    free(h_out);

    printf("\n");

    return 0;
}