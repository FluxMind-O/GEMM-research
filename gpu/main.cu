#include<iostream>
#include<vector>

#include "common/utils.h"
#include"naive_gpu.cuh"
#include"shared.cuh"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

int main(){
      
    const int N=1024;
    std::vector<float> h_a(N*N), h_b(N*N), h_c(N*N);
    const float expected=2.0f*N;  
    init_matrices(h_a,h_b,N);
    const size_t M=static_cast<size_t>(N)*N*sizeof(float);

    float *d_a ,*d_b ,*d_c;
    CUDA_CHECK(cudaMalloc(&d_a,M));
    CUDA_CHECK(cudaMalloc(&d_b,M));
    CUDA_CHECK(cudaMalloc(&d_c,M));

    CUDA_CHECK(cudaMemcpy(d_a,h_a.data(),M,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b,h_b.data(),M,cudaMemcpyHostToDevice));


    // ===================== Naive GPU =====================
    double t=benchmark([&](){
        run_naive_gemm(d_a ,d_b ,d_c ,N);
        CUDA_CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
    });
    CUDA_CHECK(cudaMemcpy(h_c.data(),d_c,M,cudaMemcpyDeviceToHost));
    if(verify(h_c,N,expected)){   
        std::cout<<"Time: "<<t<<" S"<<std::endl;
        std::cout<<"GPU_naive的GFLOPS："<<(2.0* N*N*N/t/1e9)<<std::endl<<std::endl;
    }else{
        std::cout<<"验证失败！请检查 Kernel 索引映射或内存越界"<<std::endl;
    }

    // ===================== Shared Memory Tiling=====================
    t=benchmark([&](){
        run_shared_gemm(d_a ,d_b ,d_c ,N);
        CUDA_CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
    });
    CUDA_CHECK(cudaMemcpy(h_c.data(),d_c,M,cudaMemcpyDeviceToHost));
    if(verify(h_c,N,expected)){   
        std::cout<<"Time: "<<t<<" S"<<std::endl;
        std::cout<<"Shared Memory Tiling的GFLOPS："<<(2.0* N*N*N/t/1e9)<<std::endl<<std::endl;
    }else{
        std::cout<<"验证失败！请检查 Kernel 索引映射或内存越界"<<std::endl;
    }


    
    return 0;

}

