// CPU 參考實作：最單純的三層迴圈 Floyd-Warshall，用來產生標準答案
// 用法：ref <input> <output>
// 複雜度 O(V^3)，V=2000 約數秒；V 上萬時太慢，大測資只拿來量 GPU 效能、不比對
#include <chrono>
#include <cstdio>

#include "../src/graph_io.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "用法：%s <input> <output>\n", argv[0]);
        return 1;
    }
    int V;
    std::vector<Edge> edges;
    read_graph(argv[1], V, edges);
    std::vector<int> dist = make_dist_matrix(V, edges);

    auto t0 = std::chrono::steady_clock::now();
    int* d = dist.data();
    for (int k = 0; k < V; ++k) {
        const int* dk = d + (size_t)k * V;
        // 第 k 輪內各列互不相依（第 k 列本身在第 k 輪不會變），可平行處理
#pragma omp parallel for schedule(static)
        for (int i = 0; i < V; ++i) {
            int* di = d + (size_t)i * V;
            int dik = di[k];
            if (dik == INF) continue;
            for (int j = 0; j < V; ++j) {
                int cand = dik + dk[j];  // 最大 INF + INF = 2e9，不會超出 int32
                if (cand < di[j]) di[j] = cand;
            }
        }
    }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    write_dist(argv[2], d, V, V);
    std::printf("%s：V=%d E=%zu，CPU 計算 %.1f ms\n", argv[2], V, edges.size(), ms);
    return 0;
}
