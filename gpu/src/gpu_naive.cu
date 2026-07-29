#include"gpu_naive.cuh"


__global__ void add(const float* a ,const float* b ,float* c,const int N){

    int row = blockIdx.y * blockDim.y + threadIdx.y;  //行方向
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row<N && col<N){
        float sum=0.0f;

        for (int i = 0; i < N; i++)
          sum += a[row*N+i]*b[i*N+col];
        
        c[row*N+col]=sum;
    }

}


void run_naive_gemm(const float* d_a ,const float* d_b ,float* d_c ,const int N){

    dim3 threads(16,16);
    dim3 blocks( (N+threads.x-1)/threads.x ,(N+threads.y-1)/threads.y );

    add<<<blocks,threads>>>(d_a,d_b,d_c,N);

}
