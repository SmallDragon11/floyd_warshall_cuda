// 量測 make_dist_matrix（src/graph_io.h）各步驟的 CPU 時間與 page fault 次數
// 用法：bench_build.exe <input> [BF=64] [reps=3]
// 每一輪：
//   1. alloc+fill   ：std::vector<int>(n*n, INF)，記憶體剛配置（第一次寫入）
//   2. diag         ：對角線填 0
//   3. scatter      ：依輸入檔順序寫入邊（隨機位置）
//   4. refill       ：同一個 vector 再填一次 INF（已寫過的記憶體，無 first-touch page fault）
//   5. scatter2     ：refill 後再 scatter 一次（同樣工作量，記憶體已寫過）
//   6. scatter_sorted：邊依 (src, dst) 排序後 scatter（排序不計時），比較隨機與循序位置
// 以 -HostFlags /O2 與不加分別編譯，比較 host 編譯最佳化的影響
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../src/graph_io.h"

static unsigned long long page_faults() {
    PROCESS_MEMORY_COUNTERS c{};
    GetProcessMemoryInfo(GetCurrentProcess(), &c, sizeof(c));
    return c.PageFaultCount;
}

struct Timer {
    std::chrono::steady_clock::time_point t0;
    unsigned long long pf0;
    void start() { pf0 = page_faults(); t0 = std::chrono::steady_clock::now(); }
    void stop(const char* name) {
        auto t1 = std::chrono::steady_clock::now();
        unsigned long long pf1 = page_faults();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("  %-15s %9.1f ms  %9llu page faults\n", name, ms, pf1 - pf0);
    }
};

static void scatter(std::vector<int>& dist, const std::vector<Edge>& edges, int n) {
    for (const Edge& e : edges) {
        int& d = dist[(size_t)e.src * n + e.dst];
        if (e.w < d) d = e.w;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "用法：%s <input> [BF] [reps]\n", argv[0]); return 1; }
    const int BF = argc > 2 ? std::atoi(argv[2]) : 64;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 3;

    int V;
    std::vector<Edge> edges;
    read_graph(argv[1], V, edges);
    const int n = (V + BF - 1) / BF * BF;
    const size_t nn = (size_t)n * n;
    std::printf("V=%d E=%zu n=%d matrix=%.1f MB\n", V, edges.size(), n, nn * 4 / 1e6);

    std::vector<Edge> sorted = edges;
    std::sort(sorted.begin(), sorted.end(), [](const Edge& a, const Edge& b) {
        return a.src != b.src ? a.src < b.src : a.dst < b.dst;
    });

    long long checksum = 0;  // 避免編譯器把結果整個消掉
    for (int r = 0; r < reps; ++r) {
        std::printf("rep %d\n", r);
        Timer t;
        t.start();
        std::vector<int> dist(nn, INF);
        t.stop("alloc+fill");

        t.start();
        for (int i = 0; i < V; ++i) dist[(size_t)i * n + i] = 0;
        t.stop("diag");

        t.start();
        scatter(dist, edges, n);
        t.stop("scatter");
        checksum += dist[nn / 2];

        t.start();
        std::fill(dist.begin(), dist.end(), INF);
        t.stop("refill");

        t.start();
        scatter(dist, edges, n);
        t.stop("scatter2");
        checksum += dist[nn / 3];

        std::fill(dist.begin(), dist.end(), INF);
        t.start();
        scatter(dist, sorted, n);
        t.stop("scatter_sorted");
        checksum += dist[nn / 4];
    }
    std::printf("checksum %lld\n", checksum);
    return 0;
}
