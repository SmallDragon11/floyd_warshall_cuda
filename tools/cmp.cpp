// 比對兩個輸出檔（例如 apsp.exe 的結果與 ref.exe 的標準答案）
// 用法：cmp <expected> <actual> [max_report]
// 完全相同時 exit 0，否則印出前幾個不一致的位置並 exit 1
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static std::vector<int> read_all(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "無法開啟 %s\n", path); std::exit(2); }
    std::vector<int> data;
    int buf[1 << 16];
    size_t n;
    while ((n = std::fread(buf, sizeof(int), 1 << 16, f)) > 0) data.insert(data.end(), buf, buf + n);
    std::fclose(f);
    return data;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "用法：%s <expected> <actual> [max_report]\n", argv[0]);
        return 2;
    }
    int max_report = argc > 3 ? std::atoi(argv[3]) : 10;
    std::vector<int> a = read_all(argv[1]), b = read_all(argv[2]);

    if (a.size() != b.size()) {
        std::printf("FAIL：長度不同，expected %zu 個 int，actual %zu 個 int\n", a.size(), b.size());
        return 1;
    }
    long long V = std::llround(std::sqrt((double)a.size()));
    if (V * V != (long long)a.size()) {
        std::printf("FAIL：%zu 個 int 不是 V*V\n", a.size());
        return 1;
    }

    size_t bad = 0;
    for (size_t idx = 0; idx < a.size(); ++idx) {
        if (a[idx] == b[idx]) continue;
        if ((int)bad < max_report)
            std::printf("  dist(%lld, %lld)：expected %d，actual %d\n",
                        (long long)idx / V, (long long)idx % V, a[idx], b[idx]);
        ++bad;
    }
    if (bad) {
        std::printf("FAIL：V=%lld，%zu / %zu 個值不一致\n", V, bad, a.size());
        return 1;
    }
    std::printf("PASS：V=%lld，%zu 個值完全一致\n", V, a.size());
    return 0;
}
