#pragma once

#include<iostream>
#include<cuda_runtime.h>
#include<cstdlib>

void run_naiveg_gemm(const float* d_a ,const float* d_b ,float* d_c ,const int N);

