// Blocked Floyd-Warshall（單 GPU 版）
// 用法：apsp <input> <output>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>  // NVTX：在 Nsight Systems 時間線上標出自訂區段

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
        int d_kx = d[k * n + x];
        for (int y = y_start; y < y_end; y += BLOCK_Y) {
            int * const d_y = d + y * n;
            int &d_yx = d_y[x];
            d_yx = min(d_yx, d_y[k] + d_kx);
        }
        __syncthreads();
    }
}

// 第 r 輪的 phase 2：pivot row 與 pivot column 上的 block
__global__ void phase2(int* d, int n, int r, int rounds) {
    int tileNo = blockIdx.x % (rounds - 1);
    if (tileNo >= r) {
        tileNo += 1;
    }
    
    int x, y_start, y_end;
    if (blockIdx.x < rounds - 1) { // pivot row
        x = tileNo * BF + threadIdx.x;
        y_start = r * BF + threadIdx.y;
        y_end = (r + 1) * BF;
    } else { // pivot column
        x = r * BF + threadIdx.x;
        y_start = tileNo * BF + threadIdx.y;
        y_end = (tileNo + 1) * BF;
    }

    int k_start = r * BF;
    int k_end = (r + 1) * BF;

    for (int k = k_start; k < k_end; k++) {
        int d_kx = d[k * n + x];
        for (int y = y_start; y < y_end; y += BLOCK_Y) {
            int * const d_y = d + y * n;
            int &d_yx = d_y[x];
            d_yx = min(d_yx, d_y[k] + d_kx);
        }
        __syncthreads();
    }
}

// 第 r 輪的 phase 3：其餘 block
__global__ void phase3(int* d, int n, int r) {
    int tileNoX = blockIdx.x;
    int tileNoY = blockIdx.y;
    if (tileNoX >= r) {
        tileNoX += 1;
    }
    if (tileNoY >= r) {
        tileNoY += 1;
    }
    
    int x = tileNoX * BF + threadIdx.x;
    int y_start = tileNoY * BF + threadIdx.y;

    int k_start = r * BF;
    int k_end = (r + 1) * BF;

    int d_yx[BF/BLOCK_Y];

    __shared__ int pivotRow[BF][BF];
    __shared__ __align__(16) int pivotCol[BF][BF];

    // d[y][k] tile, aka pivotCol
    for (int i=0; i<BF/BLOCK_Y; i++) {
        int offset = i * BLOCK_Y;
        int y = y_start + offset;
        int y_shared = threadIdx.y + offset;
        int k = k_start + threadIdx.x;
        pivotCol[y_shared][threadIdx.x] = d[y * n + k];
    }

    // d[k][x] tile, aka pivotRow
    for (int i=0; i<BF/BLOCK_Y; i++) {
        int offset = threadIdx.y + i * BLOCK_Y;
        int k = k_start + offset;
        pivotRow[offset][threadIdx.x] = d[k * n + x];
    }

    __syncthreads();
    
    #pragma unroll
    for (int i=0; i<BF/BLOCK_Y; i++) {
        int y = y_start + i * BLOCK_Y;
        d_yx[i] = d[y * n + x];
    }

    for (int k = k_start; k < k_end; k+=4) {
        int k_shared = k - k_start;
        int d_kx0 = pivotRow[k_shared][threadIdx.x];
        int d_kx1 = pivotRow[k_shared + 1][threadIdx.x];
        int d_kx2 = pivotRow[k_shared + 2][threadIdx.x];
        int d_kx3 = pivotRow[k_shared + 3][threadIdx.x];
        #pragma unroll
        for (int i=0; i<BF/BLOCK_Y; i++) {
            int offset = threadIdx.y + i * BLOCK_Y;
            const int4 d_yk = *reinterpret_cast<const int4*>(&pivotCol[offset][k_shared]);
            d_yx[i] = min(d_yx[i], d_yk.x + d_kx0);
            d_yx[i] = min(d_yx[i], d_yk.y + d_kx1);
            d_yx[i] = min(d_yx[i], d_yk.z + d_kx2);
            d_yx[i] = min(d_yx[i], d_yk.w + d_kx3);
        }
    }

    #pragma unroll
    for (int i=0; i<BF/BLOCK_Y; i++) {
        int y = y_start + i * BLOCK_Y;
        d[y * n + x] = d_yx[i];
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "用法：%s <input> <output>\n", argv[0]);
        return 1;
    }

    // NVTX 區段：nvtxRangePushA 開始、nvtxRangePop 結束，對應報告的 time distribution
    int V;
    std::vector<Edge> edges;
    nvtxRangePushA("read input");
    read_graph(argv[1], V, edges);
    nvtxRangePop();

    // 補齊到 BF 的倍數：虛擬點與真實點之間沒有邊，全部填 INF
    const int n = (V + BF - 1) / BF * BF;
    const int rounds = n / BF;
    const size_t bytes = (size_t)n * n * sizeof(int);

    // host 端距離矩陣：n×n、每列長度 n
    nvtxRangePushA("build matrix");
    std::vector<int> h_dist = make_dist_matrix(V, edges, n);
    nvtxRangePop();

    int* d_dist = nullptr;
    nvtxRangePushA("cudaMalloc + CUDA init");  // 第一個 CUDA API 呼叫會連帶做 context 初始化
    CUDA_CHECK(cudaMalloc(&d_dist, bytes));
    nvtxRangePop();

    nvtxRangePushA("H2D copy");
    CUDA_CHECK(cudaMemcpy(d_dist, h_dist.data(), bytes, cudaMemcpyHostToDevice));
    nvtxRangePop();

    nvtxRangePushA("compute");

    dim3 block(BF, BLOCK_Y);
    dim3 grid2(2*rounds-2), grid3(rounds-1, rounds-1);
    if (rounds > 1) {
        for (int r = 0; r < rounds; r++) {
            phase1<<<1, block>>>(d_dist, n, r);
            phase2<<<grid2, block>>>(d_dist, n, r, rounds);
            phase3<<<grid3, block>>>(d_dist, n, r);
        }
    } else {
        phase1<<<1, block>>>(d_dist, n, 0);
    }

    CUDA_CHECK(cudaGetLastError());  // 檢查 kernel launch 是否失敗（例如 blockDim 超過上限）
    // 量測用：launch 是非同步的，這裡等所有 kernel 跑完，"compute" 區段才會涵蓋實際計算時間；
    // 少了這行，等待時間會被算進後面的 D2H cudaMemcpy
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    nvtxRangePushA("D2H copy");
    CUDA_CHECK(cudaMemcpy(h_dist.data(), d_dist, bytes, cudaMemcpyDeviceToHost));
    nvtxRangePop();
    CUDA_CHECK(cudaFree(d_dist));

    nvtxRangePushA("write output");
    write_dist(argv[2], h_dist.data(), V, n);  // 只寫出前 V 列、每列前 V 欄
    nvtxRangePop();
    return 0;
}
