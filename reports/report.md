# Blocked Floyd-Warshall（單 GPU）效能分析報告

> **草稿狀態說明**
> - 數據、表格、量測方法由 agent 彙整，數字皆可追溯到下列 log 或報告檔。
> - 「分析」段落分四種標記：
>   - **【使用者】**：使用者在討論中自己寫出的結論（照原意整理）。
>   - **【參考答案，待改寫】**：使用者請 agent 直接回答的部分，交報告前要用自己的話重寫。
>   - **【待寫】**：尚未作答，由使用者撰寫。
>   - **【待驗證假設】**：附驗證方式，尚未量測。
> - 本檔為 2026-11-25 精簡版。完整推導、PC sampling 樣本表、Source 頁逐條數據與被推翻推論的原文見 [`report_full.md`](report_full.md)。
>   節號對照（精簡版 ← 完整版）：4 ← 4、10.1–10.2；6 ← 5–9；7 ← 10.3–10.5；8 ← 11.1–11.4；9 ← 11.5；10 ← 12；11 ← 13；附錄 A 為新增。

## 1. 實驗環境

| 項目 | 內容 |
|---|---|
| GPU | NVIDIA GeForce GTX 1650（sm_75），16 SM，L2 1,048,576 bytes，每個 SM 64 K registers、64 KB shared memory、最多 1024 threads（deviceQuery） |
| CUDA Toolkit | 13.4 |
| Nsight Systems | 2026.3.2 |
| Nsight Compute | 2026.3.0 |
| 編譯 | `nvcc -arch=sm_75 -lineinfo -res-usage -Xcompiler /utf-8`（`build.ps1`） |
| BF | 64（第 8 節另測 32） |

測資：

| 名稱 | V | 用途 |
|---|---|---|
| c06 | 3001（補齊至 n = 3008，47 輪） | 正確性 + `ncu` |
| p01 | 10000（補齊至 n = 10048，157 輪），矩陣 403.849 MB | 只量效能，**輸出未驗證正確性** |

## 2. 實作摘要（目前版本，commit `0b5d541`）

- 矩陣補齊到 BF 的倍數，補的值填 INF（虛擬點的對角線也是 INF），kernel 不需邊界判斷。
- 每輪在 default stream 依序 launch `phase1`（1 個 block）、`phase2`（`2*rounds-2` 個 block，前半 pivot row、後半 pivot column）、`phase3`（`(rounds-1)²` 個 block）。迴圈內沒有 host 同步；迴圈結束後的 `cudaDeviceSynchronize()` 只為了讓 NVTX `compute` 區段涵蓋實際的計算時間。
- 各 phase 的設計：

| kernel | block 形狀 | 記憶體用法 |
|---|---|---|
| phase1 | 64 × 4 | 直接讀寫 global（未優化，佔比低，見 10 節） |
| phase2 | 64 × 4 | 自己的 tile（`self`）與 pivot tile（`pivot`）放 shared memory，k 在外、每步一個 `__syncthreads()`，只在變小時寫入（第 7 節） |
| phase3 | 32 × 8（`BLOCK_X_P3` × `BLOCK_Y_P3`） | `pivotCol` = `d[y][k]`、`pivotRow` = `d[k][x]` 放 shared memory；每個 thread 負責 8 個 y、2 個 x（x 與 x+32），累積值留在 register；`pivotCol` 以 `int4` 讀取；`d_yx` 在 barrier 前讀取；1D grid + block swizzling（`SWIZZLE_G` = 8）（第 6、9 節） |

## 3. 量測方法

**Nsight Systems（整體時間線）**：host 程式用 NVTX（`nvtxRangePushA`／`nvtxRangePop`）標出 7 個區段：`read input`、`build matrix`、`cudaMalloc + CUDA init`、`H2D copy`、`compute`、`D2H copy`、`write output`。

```
nsys profile --trace=cuda,nvtx --output=<名稱> --force-overwrite=true --stats=true apsp.exe <input> <output>
```

`--stats=true` 產生 `nvtx_sum`、`cuda_api_sum`、`cuda_gpu_kern_sum` 等摘要表（log 存於 `build\nsys_*.log`）。

**Nsight Compute（kernel 內部）**：

```
ncu --kernel-name phase3 --launch-skip 23 --launch-count 1 --set full --import-source yes -o <名稱> -f apsp.exe c06.in <output>
```

只量 c06 第 24 輪（共 47 輪）的一次 launch；每輪的 grid 相同，各輪工作量相同。`ncu` 預設 `--clock-control boost`、`--cache-control all`，所以 Duration 和 Nsight Systems 不完全相同（見 4.4）。

**Integer GOPS**：每次 relaxation 算 2 個整數運算（加法與 min），共 V³ 次，GOPS = 2 × V³ ÷ `compute` 時間（只計 V³、不含補齊）。`ncu --set full` 沒有整數指令計數器，所以用定義計算。

## 4. Time distribution 與整體 profiling

### 4.1 Time distribution（Nsight Systems，`nvtx_sum`）

來源：`build\nsys_{p01,c06}_baseline.log`（global memory baseline）、`build\nsys_{p01,c06}_swz.log`（目前版本）。百分比的分母是 7 個區段的加總。

| 區段 | p01 baseline (ms) | p01 目前 (ms) | p01 目前 % | c06 baseline (ms) | c06 目前 (ms) | c06 目前 % |
|---|---:|---:|---:|---:|---:|---:|
| read input | 120.781 | 85.527 | 2.63 | 36.028 | 27.742 | 5.12 |
| build matrix | 1,103.314 | 980.387 | 30.19 | 131.637 | 116.049 | 21.42 |
| cudaMalloc + CUDA init | 340.875 | 252.872 | 7.79 | 349.874 | 279.193 | 51.53 |
| H2D copy | 63.140 | 56.215 | 1.73 | 6.194 | 5.396 | 1.00 |
| compute | 41,025.623 | **1,519.842** | **46.81** | 1,143.877 | **55.223** | **10.19** |
| D2H copy | 51.272 | 49.230 | 1.52 | 5.710 | 4.476 | 0.83 |
| write output | 266.417 | 302.911 | 9.33 | 63.854 | 53.718 | 9.91 |
| **合計** | 42,971.422 | **3,246.984** | 100 | 1,737.174 | **541.797** | 100 |

依題目分類：

| 類別 | 組成 | p01 baseline % | p01 目前 (ms) | p01 目前 % | c06 目前 (ms) | c06 目前 % |
|---|---|---:|---:|---:|---:|---:|
| computing | compute | 95.47 | 1,519.842 | 46.81 | 55.223 | 10.19 |
| memory copy | H2D + D2H | 0.27 | 105.445 | 3.25 | 9.872 | 1.82 |
| I/O | read input + write output | 0.90 | 388.438 | 11.96 | 81.460 | 15.04 |
| other | build matrix + cudaMalloc + CUDA init | 3.36 | 1,233.259 | 37.98 | 395.242 | 72.95 |
| communication | — | 0 | 0 | 0 | 0 | 0 |

量測事實：p01 的 `compute` 從 41,025.6 ms 降到 1,519.8 ms（27.0 倍），整個程式從 42,971.4 ms 降到 3,247.0 ms（13.2 倍）。`compute` 以外的區段變化不大（`write output` 在 234.5–302.9 ms 之間浮動），所以現在 p01 有一半以上的時間不在 GPU 計算。

**分析**
- 【使用者】`build matrix` 和 `cudaMalloc + CUDA init` 不屬於題目的三類，另列 other。
- 【待寫】communication 為何為 0（單 GPU 沒有 GPU 間傳輸）。
- 【待寫】other 的組成說明：`build matrix`（CPU 端）在 p01 為 980 ms，佔 30%，約是 GPU 計算時間的 65%；是否值得優化，以及它能否與 GPU 計算重疊。

### 4.2 Kernel 佔比（Nsight Systems，`cuda_gpu_kern_sum`）

| kernel | p01 baseline 總時間 | % | p01 目前 總時間 | % | p01 目前 平均 | c06 目前 總時間 | % | c06 目前 平均 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| phase3 | 40,420.375 ms | 98.5 | 1,383.125 ms | 91.1 | 8.810 ms | 38.117 ms | 69.9 | 811.0 µs |
| phase2 | 596.077 ms | 1.5 | 125.942 ms | 8.3 | 802.2 µs | 13.165 ms | 24.1 | 280.1 µs |
| phase1 | 8.204 ms | < 0.1 | 9.508 ms | 0.6 | 60.6 µs | 3.259 ms | 6.0 | 69.3 µs |

- 【使用者】baseline 最值得優化的是 phase 3。
- `compute` 區段與 kernel 總時間幾乎相同（pre 版 p01：2,315.8 與 2,314.7 ms），launch 之間的空檔可忽略（量測事實，見 4.3）。

### 4.3 時間線與 launch 成本（baseline，p01）

- 同一輪的 phase1、phase2、phase3 **沒有重疊**，依 launch 順序執行（量測事實，驗證「同一 stream 依序執行」）。
- kernel 之間的空檔：`cuda_gpu_trace` 470 個空檔中位數 1.024 µs、最大 9.95 µs、合計 0.53 ms，約佔 compute 的 0.001%。【使用者】空檔和 phase3 平均 257 ms 相比影響不大。（註：和 phase1 平均 52.254 µs 相比約 2%。）
- `cudaLaunchKernel` 471 次，平均 12.3 µs，合計 5.806 ms；`cudaDeviceSynchronize` 41.019 s（`cuda_api_sum`）。
- 【使用者】`cudaLaunchKernel` 回傳時 kernel 還沒執行完。launch 和 kernel 執行是非同步的，kernel 執行的同時就可以 launch 下一個，所以 host 在約 6 ms 內排完 471 個 launch，剩下的時間都在 `cudaDeviceSynchronize` 等待。
- 【使用者】`cudaDeviceSynchronize` 的作用：避免 kernel 的計算時間被算進 D2H 區段。
- 【參考答案，待改寫】launch 成本在兩種情況下會拖慢總時間：kernel 比 launch 還短（phase1 只有 launch 的約 4–5 倍長，若再更短，GPU 會等 host），或迴圈內每輪都做 host 同步。

### 4.4 同一量測在兩個工具中的呈現

- **NVTX 投影**：CUDA HW 列下的 NVTX 是區段投影到 GPU 的結果，從區段內第一個 GPU 操作開始，到最後一個結束（官方文件：nsys 內附 `nvtx_gpu_proj_sum.py` 說明）。baseline p01：

  | 區段 | CPU 端 | GPU 投影 |
  |---|---:|---:|
  | compute | 41,025.623 ms | 41,025.189 ms |
  | H2D copy | 63.140 ms | 51.787 ms |
  | D2H copy | 51.272 ms | 51.055 ms |

  H2D 多出的 11.4 ms 全部在 GPU 開始搬資料之前，來源是 pageable memory；成因見第 11 節。
- **kernel Duration**（c06 單次 phase3，`ncu` vs nsys 47 次平均）：

  | 版本 | `ncu` | nsys 平均 | 差異 |
  |---|---:|---:|---:|
  | baseline | 23.94 ms | 23.29 ms | `ncu` 長 2.8% |
  | pre | 945.41 µs | 947.5 µs | 幾乎相同 |
  | swizzle（G = 8） | 750.7 µs | 811.0 µs | **nsys 長 8.0%** |

  - 【使用者】數字不一致，原因是 profiling overhead 不同。
  - 【待寫】具體原因：單次 vs 平均、`ncu` 鎖時脈、清 cache。swizzle 版差異方向相反且變大，見第 11 節的待驗證問題。

### 4.5 小測資的提醒

【使用者】小測資時，bottleneck 很容易變成非 compute 的區段，因此用小測資量效能時，很難看清楚 compute 的改善。

（補充數據：`cudaMalloc + CUDA init` 的絕對時間兩者相近，c06 目前佔 51.53%、p01 只佔 7.79%。）

## 5. 優化總覽

c06 單次 phase3（`ncu`）與 p01 的 `compute`（nsys）。「主要限制」以 SOL 最高項與利用率反推的下限判斷（推論）。

| 版本 | commit | 改動 | c06 phase3 | 主要限制 | p01 `compute` | Integer GOPS |
|---|---|---|---:|---|---:|---:|
| baseline | — | global memory，k 在外 | 23.94 ms | DRAM（每格寫約 54 次） | 41,025.6 ms | 48.8 |
| y 在外 | `aac6b22` | 累積值放區域變數 | 3.38 ms | LSU 指令數 | — | — |
| register 重用 | `fd323ea` | 16 個累積值放 register | 2.00 ms | LSU | — | — |
| shared memory | `9888208` | pivot tile 放 shared | 989.86 µs | LSU（92.57%） | — | — |
| int4 | `5d56c03` | `pivotCol` 128 bit 讀取 | 964.51 µs | shared wavefront + DRAM | — | — |
| x2 | `564cc36` | 32 × 8，一個 thread 2 個 x | 946.91 µs | DRAM | — | — |
| pre | `25fa74b` | `d_yx` 在 barrier 前讀 | 945.41 µs | DRAM（實際上限的 99%） | 2,315.8 ms | 863.6 |
| phase2 shared | `572a9d4` | phase2 改用 shared（phase2 970.53 → 251.71 µs） | （不變） | — | 1,840.5 ms | 1,086.7 |
| swizzle | `0b5d541` | 1D grid、G = 8 | 750.7 µs | L1/TEX + DRAM 同時接近滿載 | **1,519.8 ms** | **1,315.9** |

GOPS 為 2 × 10000³ ÷ `compute`（計算值）；中間版本沒有量 p01 的 nsys。

## 6. phase3 的優化

### 6.1 y 在外、k 在內（commit `aac6b22`）

累積值放在區域變數，k 迴圈結束才寫回。報告：`reports\c06_phase3_{base,y}.ncu-repz`；`build\nsys_p01_youter.log`。

| 指標 | 改前（k 在外） | 改後（y 在外） |
|---|---:|---:|
| Duration（c06 `ncu`） | 23.94 ms | 3.38 ms |
| p01 phase3 平均（nsys） | 257.455 ms | 36.783 ms（7.00×） |
| global load／store requests | 35,751,936／17,334,272 | 34,939,392／270,848（store −98.44%） |
| load Sectors/Req | 2.545 | 2.511 |
| L1 hit rate（global loads） | 39.58% | 89.85% |
| DRAM 讀／寫 | 487.338／1,867.557 MB | 56.989／34.653 MB |
| SOL Memory／Compute | 77.89%／17.86% | 84.97%／84.97% |
| DRAM Throughput | 77.89% | 21.27% |

**分析**
- 【使用者】改前改後 Sectors/Req 的預測值分別為 2.545、2.511（`d[y][x]`、`d[k][x]` 由連續 32 個 thread 讀取為 4 sector，`d[y][k]` 在同一 warp 內同一位址為 1 sector），與實測相同。
- 【使用者】L1 hit rate（global loads）上升，多出來的 `d[k][x]` 讀取被 cache 接住了。
- 【使用者】改前同一格被寫了將近 64 次（程式碼每個 k 寫一次）。
- 【使用者】phase3 時間變短，其他 phase 大致不變，所以 phase3 的佔比下降（p01：98.5% → 90.5%）。
- 【參考答案，待改寫】DRAM 寫入 1,867.6 ÷ 34.65 ≈ 53.9 次；34.65 MB 正好是 46 × 46 個 tile × 16 KB，改後每格只寫回一次。L1 不擋寫入（L1 store sectors = 送到 L2 的寫入 sectors），L2 只擋下約 15.8%：同時駐留約 64 個 block × 16 KB ≈ 1 MB，等於整個 L2（推論）。
- 【參考答案，待改寫】瓶頸：改前卡在 DRAM 頻寬（98.36 GB/s）；改後卡在 LSU（`SM: Inst Executed Pipe Lsu` 84.97%，DRAM 只剩 21.27%），被 load 指令數限制（每次 `min` 要 2 條 global load），所以 store 少了 64 倍，speedup 只有 7 倍。

### 6.2 register 重用（commit `fd323ea`、`a26ef39`）

每個 k 只讀一次 `d[k][x]`，16 個累積值留在 register。報告：`reports\c06_phase3_{reg_lmem,reg}.ncu-repz`。

**反例（未 commit）**：第一版陣列 index 寫成 `(y - y_start) / BLOCK_Y`，`y_start` 含 `threadIdx.y`，迴圈無法完全展開，陣列被放進 local memory（64 bytes stack frame）。Duration 19.52 ms、DRAM 寫入 1,221.7 MB，又回到每格寫多次（官方文件：CUDA Programming Guide 2.3.3.4，無法確定以常數 index 存取的陣列會放到 local memory）。修正為常數範圍的迴圈 index 後，0 bytes stack frame、72 registers。

| 指標 | y 在外 | register 重用 |
|---|---:|---:|
| Duration | 3.38 ms | **2.00 ms**（1.69×） |
| global load requests（每個 warp） | 2064 | **1104** |
| local load requests | 0 | 0 |
| Block Limit Registers | 4 | 3 |
| 理論／Achieved Occupancy | 100%／98.23% | 75%／74.06% |
| Stall Long Scoreboard（cycles／指令） | 4.03 | 7.55（+87.20%） |
| SOL Memory／Compute | 84.97%／84.97% | 76.17%／76.17% |
| 總指令數（Executed） | 109,405,664 | 80,391,072（−26.52%） |

**分析**
- 【使用者】speedup 1.69×。每個 warp 的 global load requests 從 2064 降到 1104，local load 為 0，表示累積值確實留在 register。
- 【使用者】一個 thread 72 registers × 256 threads/block = 18,432 registers/block。sm_75 每個 SM 有 64 K registers（CUDA Programming Guide 的 compute capability 表），65,536 / 18,432 = 3.56，所以每個 SM 只放得下 3 個 block，和 `Block Limit Registers` = 3 一致。
- 【參考答案，待改寫】occupancy 從 98% 降到 74%，速度卻快了 1.69×：occupancy 只是隱藏延遲的手段，本身不是目標。Achieved 74.06% 幾乎等於理論 75%，限制來自 register。
- 【參考答案，待改寫】LSU 工作量（每個 warp 的 global load requests）少 1.87 倍，速度只快 1.69 倍，差額來自 LSU 利用率從 84.97% 降到 76.17%：1.87 × (76.17 / 84.97) ≈ 1.68（推論）。利用率下降是因為駐留的 warp 從約 31 降到約 24 個，且 L2 load hit rate 從 81.30% 降到 20.55%，等待變長時能補位的 warp 變少。
- 【使用者】驗證：Warp State Statistics 的 `Stall Long Scoreboard` 從 4.03 升到 7.55 cycles／指令（+87.20%），和上面的假設一致。
- 限制（推論）：stall 上升的兩個來源（warp 變少、L2 hit rate 下降）這個量測分不開，見第 11 節。

### 6.3 pivot tile 放進 shared memory（commit `9888208`）

每個 block 先以合併存取把 `pivotCol`、`pivotRow` 兩個 64×64 tile 載入 shared memory，`__syncthreads()` 一次後 k 迴圈改從 shared 讀取。tile 在 k 迴圈中只讀不寫，不需要第二個 barrier。ptxas：77 registers、32768 bytes smem。報告：`reports\c06_phase3_smem.ncu-repz`。

| 指標 | register 重用 | 預測 | shared memory |
|---|---:|---|---:|
| Duration | 2.00 ms | — | **989.86 µs（2.02×）** |
| global load requests（每個 warp） | 1104 | 48 | **48** |
| Block Limit Registers／Shared Mem | 3／— | ?／2 | 3／**2** |
| 理論／Achieved Occupancy | 75%／74.06% | 50%／≈ 50% | 50%／48.72% |
| Stall Long／Short Scoreboard（cycles／指令） | 7.55／0.03 | 下降／上升 | 0.61／0.97 |
| SOL（`Pipe Lsu`） | 76.17% | — | 92.57% |
| k 迴圈 shared load 指令（每個 warp） | — | 1088 | **576** |
| shared load bank conflicts | — | 0 | 52,106（逐條指令 `Excessive`：0） |

**分析**
- 【使用者】預測的計算：每個 warp 的 global load 為 `d[y][x]` 16 + `d[y][k]` tile（64×64 / 32 / 8 = 16）+ `d[k][x]` tile 16 = 48 條，實測相符。一個 block 用 2 × 64 × 64 × 4 B = 32 KB shared memory，sm_75 每個 SM 最多 64 KB（CUDA Programming Guide 的 compute capability 表），所以每個 SM 只放得下 2 個 block，理論 occupancy 2 × 8 / 32 = 50%，實測 48.72%。
- 【使用者】shared memory 也走 LSU（Nsight Compute Profiling Guide：「The LSU pipeline issues load, store, atomic, and reduction instructions to the L1TEX unit for global, local, and shared memory」），所以 k 迴圈的 load 指令數不會因為改用 shared memory 而減少，改善來自等待時間。
- 【使用者】瓶頸仍是 LSU，利用率從 76.17% 升到 92.57%。
- 【參考答案，待改寫】occupancy 降到 50% 仍快 2.02×：等待從 Long Scoreboard（L1TEX：global）換成 Short Scoreboard（MIO：shared）（官方文件：Profiling Guide 的 stall 定義），每條指令的 Long Scoreboard 少 6.94 cycles，Short Scoreboard 只多 0.94 cycles；等待變短，較少的 warp 也足以讓 LSU 保持忙碌。
- 【參考答案，待改寫】指令數比預測少：編譯器把 k 迴圈展開成每次 4 個 k，`pivotCol[offset][k]` 與 `[k+1]` 合併成一條 `LDS.U.64`，每個 warp 共 576 條（Source 頁）。
- 【參考答案，待改寫】bank conflict：【使用者】52,106 只佔 shared load wavefronts（9,801,470）的 0.53%。Source 頁逐條指令的 `L1 Wavefronts Shared` = `Ideal`、`Excessive` = 0，存取模式本身沒有 bank conflict（`pivotRow` 為同一列連續 32 個 int，`pivotCol` 為 broadcast）。總計數器每次重跑都不同（52,664／50,876／50,365），推論來自執行時的競爭（見第 11 節）。結論：判斷 bank conflict 以逐條指令的 `Excessive` 為準。
- 【待寫】下一步的方向。（提示：6.4 顯示減少 LSU 指令數本身不夠，要看與時間成正比的是哪個量。）

### 6.4 `pivotCol` 改用 `int4` 讀取（commit `5d56c03`）

k 迴圈每次前進 4，以 `int4` 一次讀 4 個 `pivotCol`；`pivotRow` 照舊。先做的對照實驗：只加 `__align__(16)` 時 SASS 完全不變，編譯器仍只用 `LDS.U.64`，所以「無法證明 16 bytes 對齊」不是原因（量測事實，`cuobjdump -sass`），128 bit 讀取必須在程式中明確寫出。ptxas：80 registers；memcheck 0 errors。報告：`reports\c06_phase3_int4.ncu-repz`。

| 指標 | shared memory | 預測 | int4 |
|---|---:|---|---:|
| Duration | 989.86 µs | 509–550 µs | **964.51 µs（1.03×）** |
| shared load 指令（每個 warp） | 576 | 320 | **320** |
| 每條 `LDS.U.128` 的 wavefront | — | 1 或 4 | **2** |
| shared load wavefront（每個 warp） | 576 | — | **576** |
| `Pipe Lsu` | 92.71% | — | 58.98% |
| L1/TEX Cache Throughput | 93.17% | — | 95.34% |
| DRAM Throughput | 82.93% | — | 84.84% |
| Stall Long Scoreboard（cycles／指令） | 0.61 | — | 1.53 |

**分析**
- 【使用者】預測：LSU 利用率 92.57%，若時間由 LSU 指令數決定，利用率維持不變時為 989.86 × 320 / 576 ≈ 550 µs；利用率拉到 100% 的下限為 989.86 × 0.9257 × 320 / 576 ≈ 509 µs。
- 【使用者】occupancy：shared memory 仍是 32 KB，每個 SM 只放得下 2 個 block；register 增為 80 後 65536 / (80 × 256) = 3.2，仍可放 3 個 block，所以 occupancy 不受影響。
- 【使用者】wavefront：`LDS.U.128` 每條 `L1 Wavefronts Shared` = `Ideal`，每條 2 個 wavefront、沒有 bank conflict。原本的兩個假設（只看需要的位元組數 → 1 個；固定以 1/4 warp 為單位 → 4 個）都不成立。
- 【使用者】每個 warp 的 shared load wavefront：shared memory 版 64 + 512 × 1 = 576，int4 版 64 + 256 × 2 = 576；576 × 16,928 = 9,750,528，與 Raw 頁的 9,801,470、9,790,266 相近。
- 【參考答案，待改寫】幾乎沒變快：一個 wavefront 在 L1TEX 佔一個處理 cycle（官方文件：Profiling Guide 2.3.7），而每個 warp 的 wavefront 數兩版都是 576；指令數減少只降低了發出端的負擔（`Pipe Lsu` 92% → 59%）。預測失準是因為假設「時間正比於指令數」。同時 DRAM 已到 84.84%，用利用率反推，L1/TEX 下限約 920 µs、DRAM 下限約 818 µs，兩者都接近 Duration（見 6.5），所以這一版是 shared wavefront 與 DRAM 兩個上限同時接近滿載，而不是只有 wavefront。
- 【參考答案，待改寫】Long Scoreboard 增加：PC sampling 中 MIO Throttle 減少的樣本（−2,069）與 Long Scoreboard 增加的（+2,044）幾乎相同，總樣本數不變。等待原因從 shared memory 轉成 global load，總時間與 DRAM 讀寫量幾乎不變（量測事實）。
- 【待寫】下一步的方向。

### 6.5 一個 thread 負責 2 個 x（commit `564cc36`）

phase3 改用 32 × 8 的 block（phase1、phase2 維持 64 × 4）。每個 thread 負責 8 個 y、2 個 x（x 與 x+32），每列 y 讀一次 `int4` 的 `pivotCol`，供 2 個 x 共用。ptxas：64 registers。報告：`reports\c06_phase3_x2.ncu-repz`。

原理：`pivotCol[y][k]` 是 broadcast，一個 wavefront 服務整個 warp 涵蓋的 x。一個 warp 涵蓋整個 tile 寬度（64）後，每列 y 只讀一次；`pivotRow[k][x]` 每個 x 都不同，總量省不了。

- 【使用者】2 個 x 取 x 與 x+32：一條 `LDS` 讀同一列連續 32 個 int，落在 32 個不同 bank，1 個 wavefront。寫回 global memory 也維持合併存取（推論）。
- 【使用者】兩種 block 形狀的比較（`pivotRow` 與 occupancy 兩格經 agent 提示後更正）：

| | 現在（64 × 4） | 32 × 4 | 32 × 8 |
|---|---:|---:|---:|
| 每個 block 的 warp 數 | 8 | 4 | 8 |
| 每個 thread 負責 | 16 y、1 x | 16 y、2 x | 8 y、2 x |
| 累積值（`d_yx`）個數 | 16 | 32 | 16 |
| block 的 `pivotCol` wavefront | 16 × 16 × 2 × 8 = 4096 | 16 × 16 × 2 × 4 = 2048 | 8 × 16 × 2 × 8 = 2048 |
| block 的 `pivotRow` wavefront | 64 × 8 = 512 | 64 × 2 × 4 = 512 | 64 × 2 × 8 = 1024 |
| block 的 shared load wavefront 合計 | 4608 | 2560（−44%） | 3072（−33%） |
| 每個 SM 的 block 數（shared memory 限制） | 2 | 2 | 2 |
| 理論 occupancy | 16 / 32 = 50% | 8 / 32 = 25% | 16 / 32 = 50% |

- 【使用者】先做 32 × 8：thread 數與累積值個數都和現在相同，只改變 wavefront 數，較容易判斷因果；32 × 4 同時改變 occupancy 與 register 兩個因素。
- 【使用者】兩個上限（推論，前提：Duration 與 wavefront 數成正比；`DRAM: Cycles Active` 近似 DRAM 頻寬利用率；DRAM 讀寫量不變）：shared memory wavefront 964.51 × 3072 / 4608 ≈ **643 µs**；DRAM 忙到 100%：964.51 × 0.8484 ≈ **818 µs**。Duration 由較慢者決定，所以預測約 818 µs 以上。（DRAM 的 100% 是理論峰值，實際達不到，見 6.6。）

| 指標 | int4 | 預測 | x2 |
|---|---:|---|---:|
| Duration | 964.51 µs | ≥ 818 µs | **946.91 µs（1.02×）** |
| block 的 shared load wavefront | 4608 | 3072 | **3072** |
| L1/TEX Cache Throughput | 95.34% | — | 70.15% |
| DRAM Throughput | 84.84% | 接近 100% | **86.76%** |
| DRAM 讀／寫 | 70.09／34.63 MB | 不變 | 70.10／34.61 MB |
| Theoretical／Achieved Occupancy | 50%／48.78% | 50% | 50%／48.92% |
| Stall MIO Throttle／Long Scoreboard（cycles／指令） | 2.40／1.53 | — | 1.34／2.77 |

**分析**
- 【使用者】用利用率反推 100% 時的時間下限：

  | | Duration | L1/TEX 下限 | DRAM 下限（理論峰值） |
  |---|---:|---:|---:|
  | int4 | 964.51 µs | 964.51 × 0.9534 ≈ 919.6 µs | 964.51 × 0.8484 ≈ 818.3 µs |
  | x2 | 946.91 µs | 946.91 × 0.7015 ≈ 664.3 µs | 946.91 × 0.8676 ≈ 821.5 µs |

- 【使用者】int4 版的 L1 下限（919.6 µs）和 DRAM 下限（818.3 µs）都接近實際 Duration，是 2 個資源同時接近滿載。x2 版把 wavefront 減少約 1/3，L1 下限降到 664.3 µs，但 Duration 只降 1.8%，因為被 DRAM 卡住。所以「瓶頸是 wavefront 數」要修正為：瓶頸是 DRAM。
- 【參考答案，待改寫】Long Scoreboard 增加與 6.4 相同：MIO Throttle 減少的樣本（−2,349）與 Long Scoreboard 增加的（+2,468）幾乎相同，Long Scoreboard 樣本幾乎全部落在 global load 的 consumer（第一次使用 `d_yx` 的 `IMNMX`、載入 tile 的 `STS`）。k 迴圈變快後，warp 更早走到需要 global 資料的指令。
- 【參考答案，待改寫】當時的【待驗證假設】：DRAM 只到 87%，Duration 比 DRAM 下限多約 125 µs，是因為讀 `d_yx` 排在 barrier 之後、延遲沒被遮住，以及兩個 block 同時在 k 迴圈時 DRAM 閒置。驗證方式是 6.6 的 pre 版。**結果：不成立**，87% 已是實際上限（見 6.6）。

### 6.6 `d_yx` 讀取移到 barrier 之前（pre 版，commit `25fa74b`），以及 DRAM 實際可達頻寬

讀 `d_yx` 的 48 條 `LDG` 全部移到 `BAR.SYNC` 之前，讓它的延遲和載入 tile、等 barrier 重疊。ptxas 64 registers 不變。報告：`reports\c06_phase3_pre.ncu-repz`。

| 指標 | x2 | pre | 6.5 的預期 | 結果 |
|---|---:|---:|---|---|
| 第一條 `IMNMX` 的 long_sb 樣本 | 2,942 | **650** | 大幅減少 | 成立 |
| Duration | 946.91 µs | **945.41 µs（−0.16%）** | 往 821 µs 靠近 | 不成立 |
| DRAM Throughput | 86.76% | 86.99% | 提高 | 幾乎不變 |
| DRAM 讀／寫 | 70.10／34.61 MB | 70.09／34.61 MB | — | 不變 |
| occupancy | 50%／48.92% | 50%／48.40% | 仍 50% | 成立 |
| PC sampling 總樣本數 | 17,211 | 17,227 | — | 不變（Long Scoreboard −3,772、MIO Throttle +2,043） |

**DRAM 實際可達頻寬**（`tools/bw_test.cu`，commit `b472f3b`）：每個陣列 256 MiB（遠大於 L2），`int4` 合併存取、grid-stride 迴圈，`ncu --section SpeedOfLight` 跳過暖身 launch。

| kernel | 讀 : 寫 | Duration | 有效頻寬 | DRAM Throughput |
|---|---|---:|---:|---:|
| `read` | 1 : 0 | 2.28 ms | 117.97 GB/s | 92.53% |
| `copy` | 1 : 1 | 4.86 ms | 110.37 GB/s | 86.43% |
| `min2`（`c = min(a, b)`） | 2 : 1 | 7.20 ms | 111.90 GB/s | **87.63%** |
| `cudaMemcpyAsync` D2D（cudaEvent 計時） | 1 : 1 | 4.82 ms | 111.38 GB/s | — |
| **phase3 pre 版** | ≈ 2 : 1（DRAM 層級） | 945.41 µs | **110.75 GB/s** | **86.99%** |

讀寫比要看 DRAM 層級：程式層級的 phase3 是 3 : 1（讀 `d_yx`、`pivotRow`、`pivotCol`，寫 1 個 tile），但約一半的 pivot 讀取在 L2 命中，到達 DRAM 的約 2 : 1（量測事實），所以用 `min2` 當 phase3 的實際上限（推論）。

**分析**
- 【參考答案，待改寫】延遲有被遮住：第一條 `IMNMX` 的 long_sb 從 2,942 降到 650，剩下的 Long Scoreboard 樣本移到載入 tile 的 `STS`（第一次需要還沒送達的 global 資料的地方）。
- 【參考答案，待改寫】但 Duration 不變：總樣本數不變，減少的 Long Scoreboard 轉成了 MIO Throttle 等 shared 端的等待，方向和 6.4、6.5 相反。
- 【參考答案，待改寫】phase3 的有效頻寬是 `min2` 的 99.0%，下限 104.70 MB ÷ 111.90 GB/s ≈ **935.7 µs**，實測只多約 1%。6.5 的 821.5 µs 是理論峰值下的下限，這張卡達不到。所以 6.5 的假設不成立：x2 版起 phase3 就在實際 DRAM 頻寬上限上，遮住延遲只能讓 DRAM 更忙，而 DRAM 已沒有餘裕。
- 【參考答案，待改寫】結論：在 BF = 64、**不改 block 執行順序**時，phase3 已沒有優化空間；要再快只能減少 DRAM 總流量——加大 BF（第 8 節），或提高 L2 重用（第 9 節）。
- 【待驗證假設】只讀能到 92.53%，有寫入時降到 86–88%，見第 11 節。

## 7. phase2 的優化

### 7.1 為什麼 phase2 變成主要成本（Nsight Systems，pre 版）

phase3 優化到 pre 版後，用 Nsight Systems 重看整體（`build\nsys_{c06,p01}_pre.log`）：phase2 從 baseline 起完全沒變（p01 596.077 → 595.021 ms），在 p01 佔 kernel 時間 25.7%；在 c06 的總時間 46.7 ms 已超過 phase3 的 44.5 ms，每次 launch（994 µs）也比 phase3（946 µs）長。

- 【參考答案，待改寫】Amdahl 定律：phase3 在 p01 加速 23.6 倍，但 phase2 不變，所以 `compute` 只加速 17.7 倍。下一個要優化的是 phase2（推論）。
- 【使用者】每個 SM 可放的 phase2 block 數：threads 限制 1024 / 256 = 4；registers 限制 65536 / (46 × 256) ≈ 5.57；shared memory 無限制。取最小值為 4。16 個 SM 同時可放 64 個 block，c06 每輪 92 個 block 無法同時放下。
- 【使用者】每波時間（agent 更正為無條件進位）：

| | block 數 | 同時可放 | 波數 | 一次 launch（nsys 平均） | 每波 ≈ 每個 block 的時間 |
|---|---:|---:|---:|---:|---:|
| phase3 | 2116 | 2 × 16 = 32 | ⌈66.1⌉ = 67 | 946 µs | ≈ 14 µs |
| phase2 | 92 | 4 × 16 = 64 | ⌈1.44⌉ = 2 | 994 µs | ≈ 497 µs |

- 【參考答案，待改寫】兩個獨立的原因：(1) GPU 填不滿（第二波只有 28 個 block），這是演算法結構造成的；(2) 每個 block 本身很慢，和 phase3 的 block 工作量相同卻慢約 35 倍。即使 (1) 完全消除也只到約 497 µs，所以先處理 (2)（推論）。

### 7.2 phase2 的 Nsight Compute baseline

從 Nsight Systems 切到 Nsight Compute 的原因：nsys 只能看出 phase2 整體很慢，要知道慢在哪裡（DRAM、L1、stall 原因）需要看 kernel 內部。報告：`reports\c06_phase2_base.ncu-repz`。

| 指標 | phase2 baseline |
|---|---:|
| Duration | 970.53 µs |
| Theoretical／Achieved Occupancy | 100%／96.07% |
| DRAM／L2／L1/TEX／SM Throughput | 68.76%／40.94%／28.50%／19.02% |
| DRAM 讀／寫 | **13.73 MB／72.34 MB**（必要量約 1.5／1.5 MB；寫入約 48 倍） |
| PC sampling | Long Scoreboard 76.2%、Barrier 15.2% |

- 【參考答案，待改寫】phase2 是 latency-bound：各項 throughput 都離 100% 很遠，76% 的樣本是 Long Scoreboard，只有 92 個 block，可輪替的 warp 不足以遮住延遲（推論）。
- 【參考答案，待改寫】寫入多 48 倍：global store 共 96.5 MB = 64 × 1.51 MB，每個 k 都把整個 tile 寫回 global 一次；92 個 tile 共 1.51 MB 大於 L2，L2 只擋下約 25%。這與 baseline phase3 的寫法相同。
- 【參考答案，待改寫】phase2 的 tile 依賴它自己（pivot row block 的 `d[k][x]`、pivot column block 的 `d[y][k]` 來自自己的 tile，迴圈中會被更新），所以 6.1 的「y 在外」不能用，k 必須在外、每步 `__syncthreads()`；shared memory 能用，而且是主要解法；register 重用只能部分套用（推論，依演算法）。

### 7.3 phase2 改用 shared memory（commit `572a9d4`）

`self` 與 `pivot` 各從 global 讀一次（共 32 KB），k 迴圈只讀寫 shared，最後寫回一次。ptxas：64 registers、32768 bytes smem；memcheck、racecheck 0 errors。報告：`reports\c06_phase2_smem.ncu-repz`、`reports\{c06,p01}_p2smem.nsys-rep`。

**racecheck 找到的問題**（量測事實）：第一版每步都寫 `self[y][tx] = min(self[y][tx], via)`，racecheck（c03）回報 70,280 個 error。改為「只在 `via` 較小時才寫入」後為 0 hazards。
- 【參考答案，待改寫】第 k 步更新第 k 列（或行）時 `via` = d[k][k] + 自己，d[k][k] ≥ 0（虛擬點為 INF），值不會變。第一版仍寫入相同的值，形式上仍是 data race；條件寫入後第 k 列（或行）在第 k 步根本不會被寫，race 從結構上消失，同一步內也不需要第二個 barrier（推論，依演算法；racecheck 0 hazards 支持）。

| 指標 | baseline | shared memory 版 | 7.2 的預測 |
|---|---:|---:|---|
| Duration（c06 `ncu`） | 970.53 µs | **251.71 µs（3.86×）** | 下降，但受 block 數限制 |
| DRAM 讀／寫 | 13.73／72.34 MB | **1.53／1.44 MB** | 約 1.5／1.5 MB（成立） |
| DRAM Throughput | 68.76% | 9.30% | — |
| L1/TEX／SM Throughput | 28.50%／19.02% | **91.26%／87.59%** | — |
| Theoretical／Achieved Occupancy | 100%／96.07% | 50%／47.97% | — |
| Waves Per SM | 1.44 | 2.88 | — |
| PC sampling | Long Scoreboard 76.2%、Barrier 15.2% | Short Scoreboard 33.6%、Wait 22.9%、MIO Throttle 20.3%、Barrier 5.3% | Long Scoreboard 下降（成立）、Barrier 比例上升（**不成立**） |
| p01 phase2 總時間（nsys） | 595.021 ms（25.7%） | **121.135 ms（6.6%）** | — |
| p01 `compute`（nsys） | 2,315.814 ms | **1,840.475 ms（−20.5%）** | — |

- 【參考答案，待改寫】新的瓶頸是 shared memory／LSU 的吞吐量（L1/TEX 91.26%、主要 stall 為 Short Scoreboard、Wait、MIO Throttle），與 6.3 phase3 的情形相同；每個 k 的同步不是主要成本（推論）。
- 【參考答案，待改寫】代價：occupancy 100% → 50%、波數 2 → 3，仍快 3.86 倍，因為消除的是約 28 倍的多餘 DRAM 流量。
- 【參考答案，待改寫】Amdahl 定律：phase2 在 p01 只剩 6.6%（swizzle 後為 8.3%），phase3 回到主要成本。

## 8. Blocking factor 實驗（BF = 32／64）

版本為 commit `572a9d4`（phase3 pre 版、phase2 shared memory 版，**swizzle 之前**）。BF = 32：`.\build.ps1 -Define BF=32`，程式碼不需修改（phase3 每個 thread 負責 4 個 y、1 個 x；phase1、phase2 的 block 為 32 × 4）。ptxas（BF = 32）：phase3 56 registers、8192 bytes smem。c01–c06 正確；memcheck、racecheck 0 errors。報告：`reports\{c06,p01}_bf32.nsys-rep`、`reports\c06_phase{2,3}_bf32.ncu-repz`。

### 8.1 Integer GOPS（Nsight Systems）

| | p01 BF = 64 | p01 BF = 32 | c06 BF = 64 | c06 BF = 32 |
|---|---:|---:|---:|---:|
| 輪數 | 157 | 313 | 47 | 94 |
| `compute` | **1,840.475 ms** | **3,454.980 ms（1.88×）** | 61.596 ms | 87.371 ms（1.42×） |
| phase3 總時間（平均） | 1,708.915 ms（10.885 ms） | 3,389.620 ms（10.829 ms） | 44.530 ms（947.5 µs） | 77.451 ms（823.9 µs） |
| phase2 總時間 | 121.135 ms | 58.685 ms | 13.154 ms | 7.216 ms |
| phase1 總時間 | 9.157 ms | 4.974 ms | 3.272 ms | 1.835 ms |
| **Integer GOPS** | **1,086.7** | **578.9** | 877.6 | 618.7 |

### 8.2 device memory 與 shared memory 頻寬（Nsight Compute，c06 單次 phase3）

| 指標 | BF = 64 | BF = 32 |
|---|---:|---:|
| grid | 46 × 46 | 93 × 93 |
| Duration | 945.41 µs | 868.64 µs |
| 每次 launch 的 GOPS：2 × (rounds − 1)² × BF³ ÷ Duration | **1,173** | **653** |
| DRAM 讀／寫 | 70.09／34.61 MB | 56.97／35.38 MB |
| DRAM Throughput／有效頻寬 | 86.99%／110.75 GB/s | 83.30%／106.31 GB/s |
| 整段執行的 phase3 DRAM 總量（每次 × 輪數） | 4,921 MB | 8,681 MB（**1.76×**） |
| L1/TEX Cache Throughput | 70.45% | 78.81% |
| shared wavefront（ld + st）／µs | 7,572 | 8,284 |
| 每次 relaxation 的 shared wavefront | 0.0129 | 0.0254 |
| Achieved Occupancy | 48.40% | **97.43%** |

c06 單次 phase2：BF = 64 為 251.71 µs、Achieved Occupancy 47.97%；BF = 32 為 69.54 µs、85.40%。

### 8.3 哪個 pivot tile 在 L2 命中（消去實驗）

Source 頁沒有逐條指令的 L2 hit（只有 `L2 Theoretical Sectors Global`），所以改用消去實驗：在 scratch 副本中讓 phase3 不載入 `pivotRow` 或 `pivotCol`（shared memory 填 0，結果是錯的，只用來量流量），以 DRAM 讀取的減少量估計該 tile 的流量。只收集 Duration、DRAM 讀寫、L2 sector 與 hit 的 metric。

| 版本 | Duration | DRAM 讀 | L2 讀取 sector（來自 L1） | 其中 L2 hit |
|---|---:|---:|---:|---:|
| BF = 64 原版 | 945.7 µs | 70.10 MB | 3,249,216 | 1,059,044 |
| BF = 64 不載入 `pivotRow` | **679.7 µs** | **35.49 MB** | 2,145,896 | 1,038,792 |
| BF = 64 不載入 `pivotCol` | 938.8 µs | 69.34 MB | 2,166,784 | **76** |
| BF = 32 原版 | 869.3 µs | 56.82 MB | 2,454,064 | 677,276 |
| BF = 32 不載入 `pivotRow` | 688.4 µs | 35.82 MB | 1,330,736 | 212,528 |
| BF = 32 不載入 `pivotCol` | 878.3 µs | 58.62 MB | 2,212,280 | 380,648 |

DRAM 寫入各列皆約 34.6 MB（BF = 32 約 35.4 MB）。相減得到各 tile 的 DRAM 讀取（推論）：

| tile | BF = 64 | BF = 32 |
|---|---:|---:|
| `d_yx` | ≈ 34.7 MB（幾乎全部 miss） | ≈ 35.4 MB（幾乎全部 miss） |
| `pivotCol` | ≈ 0.8 MB（幾乎全部命中） | ≈ 0（拿掉後反而多 1.8 MB，見第 11 節） |
| `pivotRow` | ≈ 34.6 MB（L2 hit 約 1.8%） | ≈ 21.0 MB（L2 hit 約 41%） |

### 8.4 分析

- 【參考答案，待改寫】Integer GOPS：BF = 32 約為 BF = 64 的一半（p01：1,086.7 → 578.9）。phase3 每次 launch 的時間幾乎不變（10.885 → 10.829 ms），但輪數加倍。phase1、phase2 在 BF = 32 時變快，但在 p01 只佔約 1.8%，抵不過 phase3。
- 【使用者】矩陣 3008 × 3008 × 4 B ≈ 36.19 MB；DRAM 讀取 70.09 MB 約為其 2 倍。（補充：phase3 寫入 34.61 MB = 46 × 46 個 tile × 16 KB；讀取中約 34.7 MB 為 `d_yx`，其餘約 35 MB 為 pivot tile。若兩個 pivot tile 全部打到 DRAM，讀取應約 104 MB，所以約一半的 pivot 讀取在 L2 命中。）
- 【參考答案，待改寫】device memory：DRAM 總流量的簡化模型是 ∝ n³ / BF（每輪矩陣進出 DRAM 約一次、共 n / BF 輪，總計算量 n³ 與 BF 無關；推論）。實測 1.76×，方向成立但少於 2 倍：寫入量相同，差異在 pivot tile 的讀取。8.3 顯示差異完全來自 `pivotRow` 的命中率（約 2% → 約 41%）；`pivotCol` 在兩種 BF 下都幾乎全部命中。所以模型的比例常數取決於 L2 重用，不是只看 BF。
- 【參考答案，待改寫】機制是重用距離（推論，前提：block 大致依編號順序被分派）：`pivotCol` 由同一列 grid 的 block 共用，一個接一個執行，重用距離很短；`pivotRow` 由同一行共用，兩次使用之間隔了一整列 grid，經手 L2 的資料約 3 × n × BF × 4 B——BF = 64 時約 2.3 MB，大於 1 MB 的 L2 → 幾乎全部 miss；BF = 32 時約 1.15 MB，與 L2 相當 → 部分命中。
- 【參考答案，待改寫】兩種 BF 的 phase3 都在實際 DRAM 上限附近（BF = 32 的讀寫比約 1.6 : 1，有效頻寬約為 `copy`、`min2` 的 96%），所以每次 launch 工作量減半，時間只少 8%。BF = 64 拿掉 `pivotRow` 後 Duration −28%、有效頻寬仍約 103 GB/s：phase3 的時間幾乎與 DRAM 流量成正比，若 `pivotRow` 也能在 L2 命中，上限約快 28%（推論）→ 第 9 節。
- 【參考答案，待改寫】shared memory：BF = 32 的 wavefront 總數與 BF = 64 幾乎相同，但 relaxation 只有一半，每次 relaxation 的 wavefront 加倍；原因是 BF = 32 時每個 thread 只負責 1 個 x，6.5 的 `pivotCol` 重用消失（推論，依程式碼）。L1/TEX 78.81% 仍低於 DRAM，不是主要限制。
- 【參考答案，待改寫】occupancy：BF = 32 的 Achieved Occupancy 從 48.40% 升到 97.43%，但沒有變快。瓶頸是 DRAM 頻寬，多出來的 warp 無法讓 DRAM 送得更快。
- 【參考答案，待改寫】結論：BF = 64 是兩個 tile 整塊放進 shared memory、每個 SM 放 2 個 block 的最大值（2 × 64² × 4 B = 32 KB）；BF = 128 需要 k 分段載入（第 10 節）。
- 限制：(1) BF 改變時 phase3 的 thread 分配也跟著改變，所以 shared memory 的比較不只反映 BF；DRAM 流量只取決於 tile 大小，不受影響（推論）。(2) 消去實驗拿掉一個 tile 也會改變其他資料在 L2 中的停留時間，相減只是近似值。(3) 本節是 swizzle 之前的版本；swizzle 後 BF = 32 的 c06 單次 phase3 為 871.2 → 762.9 µs，BF = 64 為 750.7 µs，BF = 32 每次 launch 只做一半的 relaxation，所以 BF = 64 的 GOPS 仍約為 2 倍（推論，由 `ncu` Duration 計算，未用 nsys 重量 BF = 32）。

## 9. block swizzling（commit `0b5d541`）

改動：phase3 改為 1D grid（`(rounds-1)²` 個 block），`blockIdx.x` 代表執行順序，在 kernel 內換算成 tile 座標。每 `SWIZZLE_G` 列 tile 為一組，組內先往下走完 G 列再往右；最後一組可能不足 G 列。G = 1 等同原本的逐列順序。每個 block 的工作不變，只改變執行順序。

- 前提（推論，未經查證）：硬體大致依 block 編號由小到大分派。
- c01–c06 正確（BF = 32 抽測 c03、c05、c06 也正確）；memcheck（c04）0 errors。
- 報告：`reports\c06_phase3_swz.ncu-repz`（G = 8，`--set full`）；`reports\{c06,p01}_swz.nsys-rep`。G 掃描用 8.3 的精簡 `ncu` 指令。

### 9.1 重用距離的估算（預測）

重用距離：同一份資料的兩次使用之間存取了多少不同的資料；小於 cache 容量就可能命中（推論，前提是 cache 為 fully associative、LRU，實際 L2 不完全符合，只能當估算）。BF = 64 時每個 block 的 `d_yx` 讀寫各 16 KB，`pivotRow`、`pivotCol` 各 16 KB。

| 順序 | `pivotRow` (r, x) 的重用距離 | `pivotCol` (y, r) 的重用距離 |
|---|---|---|
| 逐列（G = 1） | 隔一整列 46 個 block ≈ **2.2 MB** → miss | 約 1 個 block（≈ 48 KB）→ 命中 |
| swizzle（G） | 約 1 個 block → 命中 | 約 **G × 48 KB**；G = 8 約 400 KB，G = 16 約 784 KB |

預測：G 太小時 `pivotRow` 重用次數少；G 太大時 `pivotCol` 的重用距離超過 L2 可用容量。最佳 G 在兩者之間，且應小於 1 MB ÷ 48 KB ≈ 21。

### 9.2 實測

c06 單次 phase3（`ncu`）：

| G | Duration | DRAM 讀 | L2 hit rate |
|---:|---:|---:|---:|
| 1（原本順序） | 952.6 µs | 70.09 MB | 49.45% |
| 2 | 819.9 µs | 52.81 MB | 61.91% |
| 4 | 777.9 µs | 44.66 MB | 67.81% |
| 6 | 747.9 µs | 41.98 MB | 69.74% |
| **8（採用）** | **750.7 µs** | **41.16 MB** | **70.32%** |
| 12 | 767.9 µs | 44.54 MB | 67.94% |
| 16 | 825.8 µs | 50.33 MB | 63.71% |
| BF = 32，G = 1 → 8 | 871.2 → 762.9 µs | 56.32 → 40.40 MB | 50.66% → 69.17% |

DRAM 寫入各列不變。G = 6 與 8 的 Duration 差 0.4%，採用 DRAM 讀取最少的 G = 8。

G = 8 的 `--set full`：Duration 750.24 µs；DRAM Throughput 79.31%；**L1/TEX Cache Throughput 89.26%**；L2 Cache Throughput 48.87%；Compute (SM) Throughput 64.40%；Achieved Occupancy 48.34%。

整體（Nsight Systems）：

| | p01 swizzle 前 | p01 swizzle 後 | c06 swizzle 前 | c06 swizzle 後 |
|---|---:|---:|---:|---:|
| phase3 總時間 | 1,708.915 ms | **1,383.125 ms（−19.1%）** | 44.530 ms | 38.117 ms（−14.4%） |
| `compute` | 1,840.475 ms | **1,519.842 ms（−17.4%）** | 61.596 ms | 55.223 ms（−10.3%） |
| Integer GOPS | 1,086.7 | **1,315.9** | 877.6 | 978.8 |

### 9.3 分析

- 【參考答案，待改寫】預測成立：G 從 1 增加到 6–8 時 DRAM 讀取從 70.09 降到約 41 MB，L2 hit rate 從 49% 升到 70%；G 再增加又變差，表示 `pivotCol` 的重用距離開始超過 L2 可用容量。這也支持 8.4 的重用距離推論。
- 【參考答案，待改寫】最佳 G（6–8，約 300–400 KB）比估算的約 21 小得多，可能原因見第 11 節；重用距離的估算只能判斷趨勢，實際的 G 要靠掃描決定。
- 【參考答案，待改寫】DRAM 讀取 41.16 MB，比消去實驗的 35.49 MB 多約 5.7 MB；Duration 比消去實驗的 679.7 µs 多約 71 µs。消去實驗連從 L2 讀 `pivotRow`、寫進 shared 都省掉了，所以 679.7 µs 是達不到的上限（推論）。
- 【參考答案，待改寫】瓶頸轉移：以利用率反推的下限，L1/TEX 約 750 × 0.8926 ≈ 670 µs；DRAM（以 `min2` 的 87.63% 為實際上限）約 750 × 0.7931 ÷ 0.8763 ≈ 679 µs。兩者又同時接近滿載（類似 6.4），各自只剩約 10% 的餘裕；要再明顯變快，需要同時減少 shared memory wavefront 與 DRAM 流量（推論）。
- 【參考答案，待改寫】c06 的改善較小（`compute` −10.3%），因為 phase2 在 c06 佔 kernel 時間 24%，而 swizzle 只改 phase3。

## 10. 目前瓶頸與剩餘工作

目前狀態（量測事實）：p01 `compute` 1,519.8 ms、Integer GOPS 1,315.9；phase3 佔 kernel 時間 91.1%，c06 單次 phase3 的 L1/TEX 89.26%、DRAM 79.31%（實際上限的約 90%）；phase2 佔 8.3%、phase1 佔 0.6%（c06 為 24.1%、6.0%）。整個程式中 `compute` 只佔 p01 的 46.81%，CPU 端的 `build matrix` 佔 30.19%（4.1）。

候選（推論，未驗證）：

1. **k 分段載入以支援 BF = 128**：兩個 tile 整塊放進 shared memory 需 128 KB，超過 64 KB；改為每次載入一段 k，輪數減半、矩陣進出 DRAM 的次數也減半。但 L1/TEX 也接近滿載，做之前要先估算 shared wavefront 是否成為新瓶頸；需要重新設計 register 分配，工作量大。
2. **補 BF = 16 資料點**：需 `-Define BF=16,BLOCK_X_P3=16`（否則 `BF/BLOCK_X_P3` 為 0，程式會壞）。
3. **phase1 改用 shared memory**：結構與 phase2 相同，但只佔 p01 的 0.6%，價值低。BF = 32 時 phase1 總時間較短（p01 9.157 → 4.974 ms），BF 越大序列部分越長（量測事實，8.1），但 BF = 64 時仍只佔約 0.5%。
4. **CPU 端 `build matrix`**：p01 佔 30%，不在 GPU 上；是否處理由使用者決定（見 4.1 的【待寫】）。
5. 多 GPU／多節點版的概念討論（未實作、未實測）。

不做：
- 32 × 4（wavefront 2560、occupancy 25%、累積值 32 個）：同時改變 occupancy 與 register，且 Duration 已不只由 wavefront 決定（推論）。
- double buffering：原本的理由是「DRAM 已在實際上限」（6.6）。swizzle 後 DRAM 降到實際上限的約 90%，但 L1/TEX 也到 89%，兩個下限（670／679 µs）都只比 750 µs 少約 10%，讓載入與計算重疊最多能拿到這 10%（推論）。

## 11. 待驗證問題

| 問題 | 出處 | 驗證方式 |
|---|---|---|
| H2D 投影比 CPU 端區段短 11.4 ms 的原因（page fault，或第一次 `cudaMemcpy` 的準備成本） | 4.4 | 連做兩次 H2D，或改用 `cudaMallocHost` |
| `ncu` 和 nsys 的 Duration 差異來源；swizzle 版 nsys 平均（811.0 µs）反而比 `ncu`（750.7 µs）長 8%，與 baseline 方向相反 | 4.4 | `ncu --clock-control none --cache-control none` 重量；nsys 看各輪 phase3 時間是否隨輪數變化 |
| register 重用版 Long Scoreboard 上升的兩個來源（occupancy 下降 vs L2 hit rate 下降） | 6.2 | `__launch_bounds__` 或 `-maxrregcount` 把 register 壓到 64 以下（回到 4 個 block），確認陣列沒被放進 local memory |
| shared memory bank conflict 總計數器每次執行不同（逐條指令為 0）；可能是 L1 與 shared memory 共用 data bank 的競爭 | 6.3 | 未在文件中查到該計數器的定義，未經查證 |
| 對齊確定後編譯器仍只用 `LDS.U.64` 而非 `LDS.U.128` 的原因 | 6.4 | 未查明 |
| broadcast 的 `LDS.U.128` 需要 2 個 wavefront（`LDS.U.64` 只要 1 個）的規則 | 6.4 | Profiling Guide 沒有說明各存取寬度的 wavefront 規則 |
| L1/TEX Cache Throughput 恰為 `L1: Data Pipe Lsu Wavefronts` 的 2 倍（93.17 = 2 × 46.59、95.34 = 2 × 47.67、70.15 ≈ 2 × 35.07），且大於 breakdown 中的每一項 | 6.4、6.5 | 兩者隨 wavefront 數同步變化（6.5 支持）；2 倍的來源未查明 |
| 串流 kernel 只讀時 DRAM Throughput 92.53%，有寫入時降到 86–88%（可能是讀寫切換方向的成本，未經查證） | 6.6 | 用 3 : 1、4 : 1 等讀寫比的串流 kernel 量 DRAM Throughput |
| BF = 32 拿掉 `pivotCol` 後，L2 讀取 sector 只少 241,784 個（`pivotCol` 約 110 萬個 sector），且 DRAM 讀取反而增加 1.8 MB；推論為部分在 L1 命中、且拿掉 tile 改變了其他資料在 L2 的停留時間 | 8.3 | 量 `l1tex__t_sector_hit_rate.pct` |
| 最佳 G（6–8）遠小於以 1 MB L2 估算的約 21；可能原因：同時約 32 個 block 在執行（存取窗口較寬）、L2 分 slice 且非 fully associative、寫回的 `d_yx` 也佔用 L2 | 9.3 | 未設計 |
| swizzle 的前提「硬體大致依 block 編號由小到大分派」 | 9 | 查官方文件；或在 kernel 記錄各 block 的 `%globaltimer` 與 `%smid` |

## 附錄 A：已修正的推論

完整原文見 `report_full.md`。

| 原本的推論 | 修正 | 依據 |
|---|---|---|
| 編譯器只合併成 64 bit，是因為無法證明 16 bytes 對齊 | 不成立：加 `__align__(16)` 後 SASS 不變 | 6.4，`cuobjdump -sass` |
| int4 版預測 509–550 µs：時間正比於 LSU 指令數 | 不成立：與時間有關的是 shared wavefront 數，且 DRAM 也接近滿載 | 6.4 |
| int4 版的瓶頸只是 shared wavefront 數 | 修正為 shared wavefront 與 DRAM 兩個上限同時接近滿載；x2 減少 wavefront 後改由 DRAM 決定 | 6.5 |
| DRAM 下限 818／821 µs（以理論峰值 100% 反推） | 實際上限約 87.6%（2 : 1 讀寫比），下限約 935.7 µs | 6.6，`bw_test` |
| DRAM 只到 87%，是因為 `d_yx` 延遲沒被遮住、k 迴圈期間 DRAM 閒置 | 不成立：pre 版遮住延遲但 Duration 不變；87% 已是實際上限 | 6.6 |
| phase3 已沒有優化空間 | 只對「不改 block 執行順序」成立；swizzle 後快 21% | 6.6、9.2 |
| phase2 改用 shared 後 Barrier 比例會上升 | 不成立：5.3%，同步不是主要成本 | 7.3 |
| BF = 32 的 pivot 讀取較少，是因為每輪 pivot row + column 放得進 L2 | 不成立：差異完全來自 `pivotRow`，原因是重用距離 | 8.3 |
| DRAM 總流量 ∝ n³ / BF（BF 減半流量加倍） | 只是簡化模型：實測 1.76×；比例常數取決於 L2 重用 | 8.4 |
| 不做 double buffering，因為 DRAM 已在實際上限 | swizzle 後 DRAM 約為實際上限的 90%，但 L1/TEX 也到 89%；理由改為「兩個上限都只剩約 10% 餘裕」 | 10 |
