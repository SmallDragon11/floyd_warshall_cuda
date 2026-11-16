// Blocked Floyd-Warshall（單 GPU 版）
// 用法：apsp <input> <output>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include "graph_io.h"

// 檢查 CUDA API 的回傳值，失敗時印出位置與錯誤訊息並結束程式
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s:%d CUDA 錯誤：%s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err_));                           \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

// blocking factor：可用 .\build.ps1 -Define BF=32 覆寫
#ifndef BF
#define BF 64
#endif

// 每個 block 的 thread 形狀：BF × BLOCK_Y（baseline 為 64×4）
constexpr int BLOCK_Y = 4;

// 第 r 輪的 phase 1：pivot block (r, r)
__global__ void phase1(int* d, int n, int r) {
    int x = r * BF + threadIdx.x;
    int y_start = r * BF + threadIdx.y;
    int y_end = (r + 1) * BF;
    int k_start = r * BF;
    int k_end = (r + 1) * BF;

    for (int k = k_start; k < k_end; k++) {
        const int &d_kx = d[k * n + x];
        for (int y = y_start; y < y_end; y += BLOCK_Y) {
            int * const d_y = d + y * n;
            int &d_yx = d_y[x];
            d_yx = min(d_yx, d_y[k] + d_kx);
        }
        __syncthreads();
    }
}

// 第 r 輪的 phase 2：pivot row 與 pivot column 上的 block
__global__ void phase2(int* d, int n, int r) {
    // TODO
}

// 第 r 輪的 phase 3：其餘 block
__global__ void phase3(int* d, int n, int r) {
    // TODO
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "用法：%s <input> <output>\n", argv[0]);
        return 1;
    }

    int V;
    std::vector<Edge> edges;
    read_graph(argv[1], V, edges);

    // 補齊到 BF 的倍數：虛擬點與真實點之間沒有邊，全部填 INF
    const int n = (V + BF - 1) / BF * BF;
    const int rounds = n / BF;
    const size_t bytes = (size_t)n * n * sizeof(int);

    // host 端距離矩陣：n×n、每列長度 n
    std::vector<int> h_dist = make_dist_matrix(V, edges, n);

    int* d_dist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dist, bytes));
    CUDA_CHECK(cudaMemcpy(d_dist, h_dist.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(BF, BLOCK_Y);
    for (int r = 0; r < rounds; r++) {
        phase1<<<1, block>>>(d_dist, n, r);
        // TODO: phase2
        // TODO: phase3
    }

    CUDA_CHECK(cudaGetLastError());  // 檢查 kernel launch 是否失敗（例如 blockDim 超過上限）
    CUDA_CHECK(cudaMemcpy(h_dist.data(), d_dist, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_dist));

    write_dist(argv[2], h_dist.data(), V, n);  // 只寫出前 V 列、每列前 V 欄
    return 0;
}
