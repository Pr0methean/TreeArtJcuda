extern "C"
__global__ void multiply(long n, float *a, float *b, float *output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n) {
        output[i] = a[i] + b[i];
    }

}