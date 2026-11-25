// DRAM 實際可達頻寬量測：用最簡單的串流 kernel 量出這張卡的 DRAM 上限，
// 作為 phase3 DRAM Throughput（約 87%）的對照組。
// 用法：bw_test [MiB per array]（預設 256）
// 量測方式：cudaEvent 算有效頻寬；ncu 看 DRAM Throughput %（佔理論峰值）。
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s:%d CUDA 錯誤：%s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err_));                           \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

// 1 讀 : 1 寫
__global__ void copy_kernel(const int4* __restrict__ a, int4* __restrict__ c, size_t n4) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4;
         i += (size_t)gridDim.x * blockDim.x)
        c[i] = a[i];
}

// 2 讀 : 1 寫，與 phase3 的 DRAM 讀寫比（70 : 35 MB）相同
__global__ void min2_kernel(const int4* __restrict__ a, const int4* __restrict__ b,
                            int4* __restrict__ c, size_t n4) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4;
         i += (size_t)gridDim.x * blockDim.x) {
        int4 x = a[i], y = b[i];
        c[i] = make_int4(min(x.x, y.x), min(x.y, y.y), min(x.z, y.z), min(x.w, y.w));
    }
}

// 只讀：每個 thread 累加後寫一個值，寫入量可忽略
__global__ void read_kernel(const int4* __restrict__ a, int* __restrict__ out, size_t n4) {
    int s = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4;
         i += (size_t)gridDim.x * blockDim.x) {
        int4 x = a[i];
        s += x.x ^ x.y ^ x.z ^ x.w;
    }
    if (s == 0x7fffffff) *out = s;  // 幾乎不會成立，只為了不讓編譯器刪掉讀取
}

int main(int argc, char** argv) {
    size_t mib = argc > 1 ? std::atoll(argv[1]) : 256;
    size_t bytes = mib << 20;
    size_t n4 = bytes / sizeof(int4);

    int4 *a, *b, *c;
    int* out;
    CUDA_CHECK(cudaMalloc(&a, bytes));
    CUDA_CHECK(cudaMalloc(&b, bytes));
    CUDA_CHECK(cudaMalloc(&c, bytes));
    CUDA_CHECK(cudaMalloc(&out, sizeof(int)));
    CUDA_CHECK(cudaMemset(a, 1, bytes));
    CUDA_CHECK(cudaMemset(b, 2, bytes));

    int sms;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const int threads = 256;
    const int blocks = sms * 32;  // 每個 SM 足夠多的 block，讓 grid-stride 迴圈填滿

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    const int reps = 10;

    // name、每次搬動的總 bytes、啟動函式
    auto run = [&](const char* name, double moved, auto launch) {
        launch();  // 暖身
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(t0));
        for (int r = 0; r < reps; ++r) launch();
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaGetLastError());
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        double sec = ms / 1e3 / reps;
        std::printf("%-12s %8.3f ms  %7.2f GB/s\n", name, sec * 1e3, moved / sec / 1e9);
    };

    std::printf("array = %zu MiB, grid = %d x %d\n", mib, blocks, threads);
    run("read", 1.0 * bytes, [&] { read_kernel<<<blocks, threads>>>(a, out, n4); });
    run("copy", 2.0 * bytes, [&] { copy_kernel<<<blocks, threads>>>(a, c, n4); });
    run("min2", 3.0 * bytes, [&] { min2_kernel<<<blocks, threads>>>(a, b, c, n4); });
    run("memcpyD2D", 2.0 * bytes,
        [&] { CUDA_CHECK(cudaMemcpyAsync(c, a, bytes, cudaMemcpyDeviceToDevice)); });

    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
    CUDA_CHECK(cudaFree(out));
    return 0;
}
