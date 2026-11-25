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

// phase1、phase2 每個 block 的 thread 形狀：BF × BLOCK_Y（64×4）
constexpr int BLOCK_Y = 4;

// phase3 每個 block 的 thread 形狀：BLOCK_X_P3 × BLOCK_Y_P3（32×8）
// 每個 thread 負責 BF/BLOCK_Y_P3 個 y、BF/BLOCK_X_P3 個 x（x、x+BLOCK_X_P3、…）
#ifndef BLOCK_X_P3
#define BLOCK_X_P3 32
#endif
constexpr int BLOCK_Y_P3 = 8;

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

    // 前 rounds-1 個 block 負責 pivot row（tile (r, tileNo)），其餘負責 pivot column（tile (tileNo, r)）
    const bool isRow = blockIdx.x < rounds - 1;
    const int tileY = isRow ? r : tileNo;
    const int tileX = isRow ? tileNo : r;

    // self：自己的 tile，k 迴圈中原地更新；pivot：tile (r, r)，phase1 已算完、只讀
    __shared__ int self[BF][BF];
    __shared__ int pivot[BF][BF];

    const int tx = threadIdx.x;
    int* const selfBase = d + tileY * BF * n + tileX * BF;
    int* const pivotBase = d + r * BF * n + r * BF;

    // 從 global 各讀一次：一個 warp 讀同一列連續 32 個 int，合併存取
    for (int y = threadIdx.y; y < BF; y += BLOCK_Y) {
        self[y][tx] = selfBase[y * n + tx];
        pivot[y][tx] = pivotBase[y * n + tx];
    }
    __syncthreads();

    // k 必須在外：第 k 步要用到自己 tile 第 k 列（pivot row）或第 k 行（pivot column）
    // 在第 k-1 步之後的值，而那是其他 thread 算的
    for (int k = 0; k < BF; k++) {
        for (int y = threadIdx.y; y < BF; y += BLOCK_Y) {
            // isRow 對整個 block 相同，不會造成 warp 分歧
            const int via = isRow ? pivot[y][k] + self[k][tx]   // d[y][k] 在 pivot、d[k][x] 在自己
                                  : self[y][k] + pivot[k][tx];  // d[y][k] 在自己、d[k][x] 在 pivot
            // 只在變小時寫入：第 k 列（或行）的 via = d[k][k] + 自己，而 d[k][k] >= 0（權重非負），
            // 所以第 k 步絕不寫第 k 列（或行），其他 thread 同時讀它不會有 data race，
            // 同一步內也不需要第二個 barrier
            if (via < self[y][tx]) {
                self[y][tx] = via;
            }
        }
        // 第 k 步的寫入要在第 k+1 步讀取前完成
        __syncthreads();
    }

    for (int y = threadIdx.y; y < BF; y += BLOCK_Y) {
        selfBase[y * n + tx] = self[y][tx];
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
    
    int x_base = tileNoX * BF;
    int x_start = x_base + threadIdx.x;
    int y_start = tileNoY * BF + threadIdx.y;

    int k_start = r * BF;

    int d_yx[BF/BLOCK_Y_P3][BF/BLOCK_X_P3];

    __shared__ int pivotRow[BF][BF];
    __shared__ __align__(16) int pivotCol[BF][BF];

    // d_yx 只放在自己的 register、不經過 shared memory，所以不必等 __syncthreads()；
    // 放在載入 tile 之前，讓這批 global load 最先發出，延遲和 tile 的載入重疊
    #pragma unroll
    for (int i = 0; i < BF/BLOCK_Y_P3; i++) {
        int y = y_start + i * BLOCK_Y_P3;
        #pragma unroll
        for (int j = 0; j < BF/BLOCK_X_P3; j++) {
            d_yx[i][j] = d[y * n + x_start + j * BLOCK_X_P3];
        }
    }

    // d[y][k] tile, aka pivotCol
    for (int i = 0; i < BF/BLOCK_Y_P3; i++) {
        int offset_y = i * BLOCK_Y_P3;
        int y = y_start + offset_y;
        int y_shared = threadIdx.y + offset_y;
        for (int j = 0; j < BF/BLOCK_X_P3; j++) {
            int k_shared = threadIdx.x + j * BLOCK_X_P3;
            pivotCol[y_shared][k_shared] = d[y * n + k_start + k_shared];
        }
    }

    // d[k][x] tile, aka pivotRow
    for (int i = 0; i < BF/BLOCK_Y_P3; i++) {
        int offset = threadIdx.y + i * BLOCK_Y_P3;
        int k = k_start + offset;
        for (int j = 0; j < BF/BLOCK_X_P3; j++) {
            int x_shared = threadIdx.x + j * BLOCK_X_P3;
            pivotRow[offset][x_shared] = d[k * n + x_base + x_shared];
        }
    }

    __syncthreads();

    for (int k_shared = 0; k_shared < BF; k_shared += 4) {
        int d_kx[4][BF/BLOCK_X_P3];

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < BF/BLOCK_X_P3; j++) {
                int x_shared = threadIdx.x + j * BLOCK_X_P3;
                d_kx[i][j] = pivotRow[k_shared + i][x_shared];
            }
        }

        #pragma unroll
        for (int i = 0; i < BF/BLOCK_Y_P3; i++) {
            int offset = threadIdx.y + i * BLOCK_Y_P3;
            const int4 d_yk = *reinterpret_cast<const int4*>(&pivotCol[offset][k_shared]);
            #pragma unroll
            for (int j = 0; j < BF/BLOCK_X_P3; j++) {
                d_yx[i][j] = min(d_yx[i][j], d_yk.x + d_kx[0][j]);
                d_yx[i][j] = min(d_yx[i][j], d_yk.y + d_kx[1][j]);
                d_yx[i][j] = min(d_yx[i][j], d_yk.z + d_kx[2][j]);
                d_yx[i][j] = min(d_yx[i][j], d_yk.w + d_kx[3][j]);
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < BF/BLOCK_Y_P3; i++) {
        int y = y_start + i * BLOCK_Y_P3;
        #pragma unroll
        for (int j = 0; j < BF/BLOCK_X_P3; j++) {
            d[y * n + x_start + j * BLOCK_X_P3] = d_yx[i][j];
        }
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

    dim3 block(BF, BLOCK_Y), block3(BLOCK_X_P3, BLOCK_Y_P3);
    dim3 grid2(2*rounds-2), grid3(rounds-1, rounds-1);
    if (rounds > 1) {
        for (int r = 0; r < rounds; r++) {
            phase1<<<1, block>>>(d_dist, n, r);
            phase2<<<grid2, block>>>(d_dist, n, r, rounds);
            phase3<<<grid3, block3>>>(d_dist, n, r);
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
