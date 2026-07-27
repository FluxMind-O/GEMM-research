#pragma once
#include <vector>
#include <chrono>
#include <cmath>
#include <iostream>
#include <algorithm>

//inlinne关键字：utils.h会被多个.cpp文件引用，加上这个不会报错
//矩阵初始化赋值
inline void init_matrices(std::vector<float>& a, std::vector<float>& b, int n, float av = 1.0f, float bv = 2.0f) {     
    for (int i = 0; i < n * n; i++) {
        a[i] = av;
        b[i] = bv;
    }
}

//最后计算结果验证
inline bool verify(const std::vector<float>& c, int n, float expected) {
    for (int i = 0; i < n * n; i++) {
        if (std::fabs(c[i] - expected) > 1e-4f) {
            std::cerr << "Mismatch at index " << i << ": got " << c[i] << ", expected " << expected << std::endl;
            return false;
        }
    }
    return true;
}

//性能测试：跑20次取最低值
template<typename Func>  //函数模板复用
inline double benchmark(Func&& func, int runs = 20) {
    double min_seconds = 1e9;
    for (int r = 0; r < runs; r++) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        double seconds = std::chrono::duration<double>(end - start).count();
        if (seconds < min_seconds) min_seconds = seconds;
    }
    return min_seconds;
}

//计算浮点数
inline double compute_gflops(int n, double seconds) {
    return (2.0 * n * n * n / seconds / 1e9);
}
