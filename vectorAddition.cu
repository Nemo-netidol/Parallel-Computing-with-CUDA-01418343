#include <stdio.h>
#include <cuda.h>

void addVector(int *a, int*b, int* c, int n) {

    for (int i = 0; i < n; i++) {
        a[i] = i+1;
        b[i] = i+1;
    }

    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vectorAdd(int *d_a, int*d_b, int* d_c, int n) {
    int idx = threadIdx.x;
    printf("%d", idx);
    d_c[idx] = d_a[idx] + d_b[idx];

    
}

int main() {
    int n = 512;
    int *h_a = (int*)malloc(sizeof(int) * n);
    int *h_b = (int*)malloc(sizeof(int) * n);
    int *h_c = (int*)malloc(sizeof(int) * n);

    for (int i = 0; i < n; i++) {
        h_a[i] = i+1;
        h_b[i] = i+1;
    }

    // allocate object in device global memory
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**) &d_a, sizeof(int) * n);
    cudaMalloc((void**) &d_b, sizeof(int) * n);
    cudaMalloc((void**) &d_c, sizeof(int) * n);

    cudaMemcpy(d_a, h_a, sizeof(int) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int) * n, cudaMemcpyHostToDevice);

    // calling kernel (device funtion)
    vectorAdd<<<1, n>>>(d_a, d_b, d_c, n);
    // copy from device memory to host memory
    cudaMemcpy(h_c, d_c, sizeof(int) * n, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);


    for (int i = 0; i < n; i++) {
        printf("%d\n", h_c[i]);
    }
    return 0;
}