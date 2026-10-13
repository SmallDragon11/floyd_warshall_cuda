// 規格（PP_2018_HW4_rev3.pdf §3）的二進位 I/O，apsp.cu 與 tools/ 共用
//   輸入：int32 V, int32 E, 接著 E 組 (src, dst, weight)，皆為 int32
//   輸出：V*V 個 int32，依 (i, j) 列優先排列；i == j 為 0，無路徑為 INF
#pragma once

#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int INF = 1000000000;

struct Edge {
    int src, dst, w;
};

// 讀入規格格式的圖；檔案不存在或長度不符時印出錯誤並結束程式
inline void read_graph(const char* path, int& V, std::vector<Edge>& edges) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "無法開啟輸入檔 %s\n", path); std::exit(1); }
    int header[2];
    if (std::fread(header, sizeof(int), 2, f) != 2) {
        std::fprintf(stderr, "%s：讀不到 V、E\n", path); std::exit(1);
    }
    V = header[0];
    int E = header[1];
    edges.resize(E);
    if (E > 0 && std::fread(edges.data(), sizeof(Edge), E, f) != (size_t)E) {
        std::fprintf(stderr, "%s：邊的數量少於 E=%d\n", path, E); std::exit(1);
    }
    std::fclose(f);
}

// 把邊轉成 V*V 的距離矩陣（列優先，dist[i*V + j]）：對角線 0、無邊 INF
// 重複的邊取最小權重
inline std::vector<int> make_dist_matrix(int V, const std::vector<Edge>& edges) {
    std::vector<int> dist((size_t)V * V, INF);
    for (int i = 0; i < V; ++i) dist[(size_t)i * V + i] = 0;
    for (const Edge& e : edges) {
        int& d = dist[(size_t)e.src * V + e.dst];
        if (e.w < d) d = e.w;
    }
    return dist;
}

// 寫出 V*V 個 int32；dist 的每列長度為 ld（ld >= V，多出的欄位不會寫出）
inline void write_dist(const char* path, const int* dist, int V, size_t ld) {
    FILE* f = std::fopen(path, "wb");
    if (!f) { std::fprintf(stderr, "無法開啟輸出檔 %s\n", path); std::exit(1); }
    if (ld == (size_t)V) {
        std::fwrite(dist, sizeof(int), (size_t)V * V, f);
    } else {
        for (int i = 0; i < V; ++i) std::fwrite(dist + (size_t)i * ld, sizeof(int), V, f);
    }
    std::fclose(f);
}
