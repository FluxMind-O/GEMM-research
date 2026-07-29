#include"shared.cuh"

#define TILE_SIZE 16


__global__ void shared_gemm(const float* a, const float* b, float* c, const int N){

    __shared__ float as[TILE_SIZE][TILE_SIZE];
    __shared__ float bs[TILE_SIZE][TILE_SIZE];
 
    const int row = blockIdx.y*TILE_SIZE + threadIdx.y; //行方向(向量个数)
    const int col = blockIdx.x*TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    const int numTiles = (N+TILE_SIZE-1) / TILE_SIZE;

    for(int i=0; i<numTiles; i++){
        
        // 阶段 1：协作加载 tile 到 shared memory 
        int a_col = i*TILE_SIZE + threadIdx.x;
        int b_row = i*TILE_SIZE + threadIdx.y;

        as[threadIdx.y][threadIdx.x] = (row<N && a_col<N)? a[row*N + a_col] : 0.0f;
        bs[threadIdx.y][threadIdx.x] = (b_row<N && col<N)? b[b_row*N + col] :0.0f;

        __syncthreads();
        
        //阶段 2：从 shared memory 读取并累加 
        #pragma unroll   //循环展开优化
        for(int j=0; j<numTiles; j++)
         sum += as[threadIdx.y][j] * bs[j][threadIdx.x];

         __syncthreads();

    }
    
    //写回 global memory
    if(row<N && col<N)
        c[row*N + col] = sum;

}


void run_shared_gemm(const float* d_a ,const float* d_b ,float* d_c ,const int N){

    dim3 threads(TILE_SIZE,TILE_SIZE);
    dim3 blocks( (N+TILE_SIZE-1)/TILE_SIZE ,(N+TILE_SIZE-1)/TILE_SIZE);

    shared_gemm<<<blocks,threads>>>(d_a ,d_b ,d_c ,N);
}