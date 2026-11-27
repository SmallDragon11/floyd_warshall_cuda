# Blocked Floyd-Warshall（CUDA 單 GPU）

用 CUDA 實作 Blocked Floyd-Warshall 求全點對最短路徑（APSP），並用 Nsight Systems／Nsight Compute 分析與優化效能的練習專案。題目為 [`PP_2018_HW4_rev3.pdf`](PP_2018_HW4_rev3.pdf)（CS542200 Parallel Programming 作業 4），規格摘要見 [`docs/spec.md`](docs/spec.md)。

- 只實作單 GPU 版 `src/apsp.cu`；多 GPU、多節點版只做概念討論，未經實測。
- 規格指定的 nvprof／nvvp 改用 `nsys` 與 `ncu`。
- 效能分析報告見 [`reports/report.md`](reports/report.md)（完整推導的封存版：[`reports/report_full.md`](reports/report_full.md)）。

## 目前成果

p01（V = 10000）上的量測，數據出處見報告第 5、10 節：

| 版本 | p01 `compute` | Integer GOPS |
|---|---:|---:|
| baseline（global memory） | 41,025.6 ms | 48.8 |
| 目前（shared memory + register 重用 + block swizzling） | 1,519.8 ms | 1,315.9 |

整個程式（含讀檔、建矩陣、傳輸、寫檔）的 span 為 2,337.1 ms，`compute` 佔 69.70%。

## 目錄

| 路徑 | 內容 |
|---|---|
| `src/apsp.cu` | 單 GPU 版主程式（phase1／phase2／phase3 三個 kernel） |
| `src/graph_io.h` | 規格格式的讀寫，`apsp.cu` 與 `tools/` 共用 |
| `tools/gen.cpp` | 隨機測資產生器 |
| `tools/ref.cpp` | CPU 參考實作（OpenMP 三層迴圈），產生標準答案 |
| `tools/cmp.cpp` | 比對兩個輸出檔 |
| `tools/bw_test.cu` | 量這張卡 DRAM 實際可達頻寬，作為 phase3 的對照 |
| `tools/bench_build.cpp` | 量 host 端建矩陣各步驟的時間與 page fault |
| `tools/make_testcases.ps1` | 編譯上面的 CPU 工具並產生測資 |
| `build.ps1` | 編譯腳本 |
| `testcases/` | 測資（不進 git，用腳本重建） |
| `reports/` | 報告、截圖與匯出 CSV（`.nsys-rep`、`.ncu-rep(z)` 不進 git） |
| `build/` | 編譯產物 |

## 環境

- Windows 11、NVIDIA GeForce GTX 1650（sm_75）
- CUDA Toolkit 13.4、Visual Studio（含「使用 C++ 的桌面開發」，提供 `cl.exe`）
- Nsight Systems 2026.3.2、Nsight Compute 2026.3.0

`build.ps1` 找不到 `cl.exe` 時會自動載入 Visual Studio 的開發環境。

## 使用方式

以下指令都在 PowerShell、專案根目錄執行。

### 1. 產生測資

```powershell
.\tools\make_testcases.ps1            # 編譯 gen/ref/cmp，產生 c01–c06 與 p01–p02
.\tools\make_testcases.ps1 -SkipPerf  # 只產生正確性測資
.\tools\make_testcases.ps1 -Rebuild   # 強制重編工具
```

- `cXX.in`／`cXX.out`：正確性測資與 `ref.exe` 的標準答案，V 混合 BF 的倍數與非倍數以檢查邊界處理。
- `pXX.in`：效能測資（p01 V = 10000、p02 V = 20000），CPU 參考實作跑不完，沒有標準答案。p02 的輸出檔約 1.6 GB。

### 2. 編譯

```powershell
.\build.ps1                    # src/apsp.cu → build/apsp.exe，BF 預設 64
.\build.ps1 -Define BF=32      # 改 blocking factor（等同 #define BF 32）
.\build.ps1 -Source tools/bw_test.cu -Output build/bw_test.exe
```

`build.ps1` 固定傳給 `nvcc` 的參數：

- `-arch=sm_75`：GTX 1650 的 compute capability。
- `-lineinfo`：保留 device 端行號，讓 Nsight Compute 能對應到原始碼。
- `-res-usage`：編譯時印出每個 kernel 的 register／shared memory 用量。
- `-Xcompiler /utf-8`：`cl.exe` 以 UTF-8 讀原始碼，避免中文註解造成 C4819。
- `-Xcompiler /O2`：開 host 端最佳化；`nvcc` 預設不會傳 `/O` 給 `cl.exe`（報告 4.7）。

### 3. 執行與驗證

```powershell
.\build\apsp.exe testcases\c06.in build\c06.out
.\build\cmp.exe testcases\c06.out build\c06.out   # 相同時 exit 0
```

### 4. Profiling

```powershell
# Nsight Systems：整體時間線、NVTX 區段、kernel 佔比、記憶體傳輸
& "C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.3.2\target-windows-x64\nsys.exe" `
    profile --trace=cuda,nvtx --output=p01 --force-overwrite=true --stats=true apsp.exe p01.in p01.out

# Nsight Compute：只量 c06 第 24 輪的一次 phase3
ncu --kernel-name phase3 --launch-skip 23 --launch-count 1 --set full --import-source yes `
    -o phase3 -f apsp.exe c06.in c06.out
```

- `nsys` 不在 PATH，用完整路徑呼叫（版本資料夾依實際安裝調整）。
- **路徑不能含非 ASCII 字元**：在本專案的中文路徑下直接跑 `nsys profile` 會失敗（`working_directory contains invalid UTF-8 data`）。做法是把執行檔與輸入檔複製到純 ASCII 目錄再執行，報告檔再複製回 `reports/`。
- `ncu --import-source yes` 要匯入原始碼時，執行檔也要從 ASCII 路徑的原始碼編譯：把 `src/apsp.cu`、`src/graph_io.h` 複製過去，再用 `.\build.ps1 -Source <該目錄>\apsp.cu -Output build\<名稱>.exe`。
- `ncu` 的輸出副檔名是 `.ncu-repz`。

各參數的意義與報告中使用的匯出指令見 [`reports/report.md`](reports/report.md) 第 3 節。

## 實作重點

- 矩陣補齊到 BF 的倍數（補的值填 INF），kernel 不需邊界判斷。
- 每輪在 default stream 依序 launch phase1（pivot block）、phase2（pivot row／column）、phase3（其餘 block），迴圈內沒有 host 同步。
- phase2、phase3 把需要的 tile 放進 shared memory；phase3 每個 thread 負責 8 × 2 格，累積值留在 register，並以 block swizzling 提高 L2 命中。
- host 端 CUDA 初始化在另一條 thread 執行，與讀檔、建矩陣重疊。

各項優化的動機、量測與分析見報告第 5–9 節。
