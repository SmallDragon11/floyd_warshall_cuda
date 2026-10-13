// 隨機測資產生器：產生 V 個頂點、E 條不重複且無自環的有向邊，權重均勻分布於 [0, 100]
// 用法：gen <V> <E> <output> [seed]
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <unordered_set>
#include <vector>

#include "../src/graph_io.h"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "用法：%s <V> <E> <output> [seed]\n", argv[0]);
        return 1;
    }
    int V = std::atoi(argv[1]);
    long long E = std::atoll(argv[2]);
    uint64_t seed = argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 1;

    // 所有可能的邊 (src, dst), src != dst，共 V*(V-1) 條，以 0..M-1 編號
    long long M = (long long)V * (V - 1);
    if (V < 1 || E < 0 || E > M || E > INT32_MAX) {
        std::fprintf(stderr, "參數不合法：需 V >= 1 且 0 <= E <= V*(V-1) = %lld\n", M);
        return 1;
    }

    // Robert Floyd 抽樣法：從 [0, M) 抽出 E 個不重複的編號，只需 O(E) 記憶體
    std::mt19937_64 rng(seed);
    std::unordered_set<long long> picked;
    picked.reserve((size_t)E * 2);
    std::vector<long long> ids;
    ids.reserve((size_t)E);
    for (long long j = M - E; j < M; ++j) {
        long long t = std::uniform_int_distribution<long long>(0, j)(rng);
        long long id = picked.insert(t).second ? t : j;
        if (id == j) picked.insert(j);
        ids.push_back(id);
    }

    std::uniform_int_distribution<int> weight(0, 100);
    std::vector<Edge> edges((size_t)E);
    for (size_t k = 0; k < edges.size(); ++k) {
        long long id = ids[k];
        int src = (int)(id / (V - 1));
        int r = (int)(id % (V - 1));
        edges[k] = {src, r < src ? r : r + 1, weight(rng)};  // 跳過 dst == src
    }

    FILE* f = std::fopen(argv[3], "wb");
    if (!f) { std::fprintf(stderr, "無法開啟輸出檔 %s\n", argv[3]); return 1; }
    int header[2] = {V, (int)E};
    std::fwrite(header, sizeof(int), 2, f);
    std::fwrite(edges.data(), sizeof(Edge), edges.size(), f);
    std::fclose(f);
    std::printf("%s：V=%d E=%lld seed=%llu\n", argv[3], V, E, (unsigned long long)seed);
    return 0;
}
