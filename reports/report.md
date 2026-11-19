# Blocked Floyd-Warshall（單 GPU）效能分析報告

> **草稿狀態說明**
> - 數據、表格、量測方法由 agent 彙整，數字皆可追溯到下列 log 或報告檔。
> - 「分析」段落分三種標記：
>   - **【使用者】**：使用者在討論中自己寫出的結論（照原意整理）。
>   - **【參考答案，待改寫】**：使用者請 agent 直接回答的部分，交報告前要用自己的話重寫。
>   - **【待寫】**：尚未作答，由使用者撰寫。

## 1. 實驗環境

| 項目 | 內容 |
|---|---|
| GPU | NVIDIA GeForce GTX 1650（sm_75），16 SM，L2 1,048,576 bytes（deviceQuery） |
| CUDA Toolkit | 13.4 |
| Nsight Systems | 2026.3.2 |
| Nsight Compute | 2026.3.0 |
| 編譯 | `nvcc -arch=sm_75 -lineinfo -res-usage -Xcompiler /utf-8`（`build.ps1`） |
| BF | 64；block 形狀 64 × 4（`BLOCK_Y = 4`），每個 thread 負責 1 欄、16 列 |

測資：

| 名稱 | V | 用途 |
|---|---|---|
| c06 | 3001（補齊至 n = 3008，47 輪） | 正確性 + `ncu` |
| p01 | 補齊後 n = 10048（157 輪），矩陣 403.849 MB | 只量效能，**輸出未驗證正確性** |

## 2. 實作摘要

- 矩陣補齊到 BF 的倍數，補的值填 INF，kernel 不需邊界判斷。
- 每輪在 default stream 依序 launch `phase1`（1 個 block）、`phase2`（`2*rounds-2` 個 block，前半 pivot row、後半 pivot column）、`phase3`（`(rounds-1)²` 個 block）。迴圈內沒有 host 同步。
- 迴圈結束後的 `cudaDeviceSynchronize()` 只為了讓 NVTX `compute` 區段涵蓋實際的計算時間。
- phase 3 的版本演進見第 6 節。

## 3. 量測方法

### Nsight Systems（整體時間線）

在 host 程式用 NVTX（`nvtxRangePushA`／`nvtxRangePop`）標出 7 個區段：`read input`、`build matrix`、`cudaMalloc + CUDA init`、`H2D copy`、`compute`、`D2H copy`、`write output`。

```
nsys profile --trace=cuda,nvtx --output=<名稱> --force-overwrite=true --stats=true apsp.exe <input> <output>
```

`--stats=true` 產生 `nvtx_sum`、`cuda_api_sum`、`cuda_gpu_kern_sum`、`cuda_gpu_mem_time_sum` 等摘要表（log 存於 `build\nsys_*.log`）。

### Nsight Compute（kernel 內部）

```
ncu --kernel-name phase3 --launch-skip 23 --launch-count 1 --set full --import-source yes -o <名稱> -f apsp.exe c06.in <output>
```

只量 c06 第 24 輪（共 47 輪）的一次 phase3。`ncu` 預設 `--clock-control boost`（鎖定時脈）、`--cache-control all`，所以 Duration 和 Nsight Systems 的量測不完全相同（見 4.5）。

## 4. Baseline profiling（global memory，phase3 k 在外）

### 4.1 Time distribution

來源：`build\nsys_p01_baseline.log`、`build\nsys_c06_baseline.log` 的 `nvtx_sum`。百分比的分母是 7 個區段的加總。

| 區段 | p01 (ms) | p01 % | c06 (ms) | c06 % |
|---|---:|---:|---:|---:|
| read input | 120.781 | 0.28 | 36.028 | 2.07 |
| build matrix | 1,103.314 | 2.57 | 131.637 | 7.58 |
| cudaMalloc + CUDA init | 340.875 | 0.79 | 349.874 | 20.14 |
| H2D copy | 63.140 | 0.15 | 6.194 | 0.36 |
| compute | 41,025.623 | 95.47 | 1,143.877 | 65.85 |
| D2H copy | 51.272 | 0.12 | 5.710 | 0.33 |
| write output | 266.417 | 0.62 | 63.854 | 3.68 |

依題目分類：

| 類別 | 組成 | p01 (ms) | p01 % | c06 (ms) | c06 % |
|---|---|---:|---:|---:|---:|
| computing | compute | 41,025.623 | 95.47 | 1,143.877 | 65.85 |
| memory copy | H2D + D2H | 114.412 | 0.27 | 11.904 | 0.69 |
| I/O | read input + write output | 387.198 | 0.90 | 99.882 | 5.75 |
| other | build matrix + cudaMalloc + CUDA init | 1,444.189 | 3.36 | 481.511 | 27.72 |
| communication | — | 0 | 0 | 0 | 0 |

**分析**
- 【使用者】`build matrix` 和 `cudaMalloc + CUDA init` 不屬於題目的三類，另列 other。
- 【待寫】communication 為何接近 0（單 GPU 沒有 GPU 間傳輸）。
- 【待寫】other 的組成說明（`build matrix` 在 p01 有 1.1 s，比 I/O 與 memory copy 加總還多）。

### 4.2 Kernel 佔比（p01）

來源：`cuda_gpu_kern_sum`。

| kernel | 次數 | 總時間 (ms) | 平均 | % |
|---|---:|---:|---:|---:|
| phase3 | 157 | 40,420.375 | 257.455 ms | 98.5 |
| phase2 | 157 | 596.077 | 3.797 ms | 1.5 |
| phase1 | 157 | 8.204 | 52.254 µs | < 0.1 |

【使用者】最值得優化的是 phase 3。

### 4.3 時間線觀察

- 同一輪的 phase1、phase2、phase3 **沒有重疊**，依 launch 順序執行（量測事實，驗證了「同一 stream 依序執行」的推論）。
- kernel 之間有空檔：使用者在時間線上量到 1.052 µs、1.084 µs；`cuda_gpu_trace` 470 個空檔的中位數 1.024 µs、最大 9.95 µs、合計 0.53 ms，約佔 compute 區段的 0.001%。
- 【使用者】空檔和 phase3 平均 257 ms 相比影響不大。（註：和 phase1 的平均 52.254 µs 相比約 2%。）

### 4.4 Host 端行為與 launch 成本

| 項目 | 數值 | 來源 |
|---|---|---|
| `cudaLaunchKernel` | 471 次，平均 12.3 µs，合計 5.806 ms | `cuda_api_sum` |
| `cudaDeviceSynchronize` | 41.019 s | `cuda_api_sum` |
| kernel 總時間 | 41,024.657 ms | `cuda_gpu_kern_sum` 三列加總 |

- 【使用者】`cudaLaunchKernel` 回傳時 kernel 還沒執行完。launch 和 kernel 執行是非同步的，kernel 執行的同時就可以 launch 下一個，所以 host 在約 6 ms 內排完 471 個 launch，剩下的時間都在 `cudaDeviceSynchronize` 等待。
- 【使用者】`cudaDeviceSynchronize` 的作用：避免 kernel 的計算時間被算進 D2H 區段。
- 【參考答案，待改寫】launch 成本在兩種情況下會拖慢總時間：kernel 比 launch 還短（例如 phase1 只有 launch 的 4 倍長，若再更短，GPU 會等 host），或迴圈內每輪都做 host 同步。

### 4.5 同一量測在兩個工具中的呈現

- **NVTX 投影**：CUDA HW 列下的 NVTX 是區段投影到 GPU 的結果，從區段內第一個 GPU 操作開始，到最後一個結束（官方文件：nsys 內附 `nvtx_gpu_proj_sum.py` 說明）。

  | 區段 | CPU 端 | GPU 投影 |
  |---|---:|---:|
  | compute | 41,025.623 ms | 41,025.189 ms |
  | H2D copy | 63.140 ms | 51.787 ms |
  | D2H copy | 51.272 ms | 51.055 ms |

  H2D 多出的 11.4 ms 全部在 GPU 開始搬資料之前（`cudaMemcpy` 開始 1874.273 ms，GPU memcpy 開始 1885.673 ms），來源是 pageable memory。成因是**待驗證假設**：page fault，或第一次 `cudaMemcpy` 的準備成本（見第 8 節）。
- **c06 phase3（改前）**：`ncu` Duration 23.94 ms；Nsight Systems 47 次平均 23.29 ms（min 22.60、max 24.30）。
  - 【使用者】數字不一致，原因是 profiling overhead 不同。
  - 【待寫】具體原因：單次 vs 平均、`ncu` 鎖時脈、清 cache。可用 `--clock-control none --cache-control none` 驗證。

### 4.6 小測資的提醒

【使用者】小測資時，bottleneck 很容易變成非 compute 的區段，因此用小測資量效能時，很難看清楚 compute 的改善。

（補充數據：`cudaMalloc + CUDA init` 的絕對時間兩者相近，約 341 ms／350 ms；佔比卻是 c06 20.14%、p01 0.79%。）

## 5. 第一個優化：phase3 改為 y 在外、k 在內（commit `aac6b22`）

累積值放在區域變數，k 迴圈結束才寫回。

### 5.1 整體（Nsight Systems，p01）

來源：`build\nsys_p01_youter.log`。

| kernel | baseline 平均 | y 在外 平均 | speedup |
|---|---:|---:|---:|
| phase3 | 257.455 ms | 36.783 ms | 7.00× |
| phase2 | 3.797 ms | 3.781 ms | — |
| phase1 | 52.254 µs | 59.067 µs | — |

phase3 佔 kernel 時間：98.5% → 90.5%。

【使用者】phase3 時間變短，其他 phase 大致不變，所以 phase3 的佔比下降。

### 5.2 kernel 內部（Nsight Compute，c06 第 24 輪 phase3）

報告：`reports\c06_phase3_base.ncu-repz`、`reports\c06_phase3_y.ncu-repz`。

| 指標 | 改前（k 在外） | 改後（y 在外） |
|---|---:|---:|
| Duration | 23.94 ms | 3.38 ms |
| registers / thread | 46 | 64 |
| global load requests | 35,751,936 | 34,939,392（−2.27%） |
| global store requests | 17,334,272 | 270,848（−98.44%） |
| load Sectors/Req | 2.545 | 2.511 |
| store Sectors/Req | 4.0 | 4.0 |
| L1 hit rate（global loads） | 39.58% | 89.85% |
| L2 hit rate | 88.10% | 82.41% |
| DRAM read | 487.338 MB | 56.989 MB |
| DRAM write | 1,867.557 MB | 34.653 MB |
| Memory SOL | 77.89% | 84.97% |
| Compute (SM) SOL | 17.86% | 84.97% |
| DRAM Throughput | 77.89% | 21.27% |

**預測與實測**

| 項目 | 預測（推論） | 實測 |
|---|---|---|
| 每個 thread 的 global load | 2112 → 2064（比例 1.023） | requests 比例 1.023 |
| 每個 thread 的 global store | 1024 → 16（64 倍） | requests 比例 64.0 |
| load Sectors/Req，改前 | (64×4 + 1024×1 + 1024×4) / 2112 = 2.545 | 2.545 |
| load Sectors/Req，改後 | (16×4 + 1024×1 + 1024×4) / 2064 = 2.511 | 2.511 |

（`d[y][x]`、`d[k][x]` 由連續 32 個 thread 讀取，4 sector；`d[y][k]` 在同一 warp 內是同一位址，1 sector。）

**分析**
- 【使用者】改前改後 Sectors/Req 的預測值分別為 2.545、2.511。
- 【使用者】L1 hit rate（global loads）上升，多出來的 `d[k][x]` 讀取被 cache 接住了。
- 【使用者】改前同一格被寫了將近 64 次（程式碼每個 k 寫一次）。
- 【參考答案，待改寫】DRAM 寫入次數：1,867,557,216 ÷ 34,653,120 ≈ 53.9 次。34.65 MB 正好是 phase3 的 46×46 個 tile × 16 KB，所以改後每格只寫回一次。
- 【參考答案，待改寫】寫入是被誰接住的：
  - L1 收到的 store sectors 和 L1 送到 L2 的寫入 sectors（`lts__t_sectors_srcunit_tex_op_write.sum`）完全相等（改前 69,337,088，改後 1,083,392），所以 L1 沒有擋下任何寫入。
  - 送到 L2 的 2,218,786,816 bytes 中，有 1,867,557,216 寫回 DRAM，L2 只擋下約 15.8%。
  - 推論：每個 SM 同時 4 個 block，16 個 SM 約 64 個 block × 16 KB ≈ 1 MB，等於整個 L2，再加上讀取資料的競爭，所以重複寫入大多被擠出去。
  - 驗證：看 Occupancy section 的 block 數，或改 `BF=32` 看 L2 擋下的比例是否上升。
- 【參考答案，待改寫】瓶頸：
  - 改前卡在 DRAM 頻寬：Memory SOL = DRAM: Cycles Active = 77.89%，實際 98.36 GB/s，約為理論 128 GB/s 的 77%。理論值以 4001 MHz × 2 × 128 bit 計算，「× 2」未經查證。
  - 改後卡在 LSU：Compute SOL 的第一列是 `SM: Inst Executed Pipe Lsu` 84.97%，Memory SOL 的第一列是 `L1: Lsuin Requests`；DRAM 只剩 21.27%，FMA 23.40%、ALU 18.80%。
  - 也就是被 load／store 指令數限制（每次 `min` 要發出 2 條 global load），所以 store 少了 64 倍，speedup 只有 7 倍。

## 6. 第二個優化：register 重用（phase3 k 在外，16 個累積值放 register）

每個 k 只讀一次 `d[k][x]`，16 個累積值留在 register，k 迴圈結束才寫回。

### 6.1 反例：陣列被放進 local memory（未 commit）

第一版以 `y = y_start; y < y_end; y += BLOCK_Y` 當迴圈變數，陣列 index 寫成 `(y - y_start) / BLOCK_Y`。`y_start` 含有 `threadIdx.y`，迴圈次數不是編譯期常數，無法完全展開，所以陣列被放進 local memory（推論，依據 CUDA Programming Guide 2.3.3.4：「Arrays for which it cannot determine that they are indexed with constant quantities」會放到 local memory）。

報告：`reports\c06_phase3_reg_lmem.ncu-repz`。

| 指標 | y 在外 | register 重用（local memory 版） |
|---|---:|---:|
| ptxas | 64 regs，0 bytes stack | 35 regs，**64 bytes stack frame** |
| Duration | 3.38 ms | **19.52 ms** |
| global load requests | 34,939,392 | 18,688,512（每個 warp 1104） |
| local load／store requests | 0 | 17,605,120／17,605,120（每個 warp 1040） |
| DRAM write | 34.653 MB | 1,221.658 MB |

global load 減少的目的達成了，但每次更新累積值都要讀寫 local memory，又回到每格寫 64 次的情況。

### 6.2 修正版：迴圈變數改為常數範圍的 index（commit `fd323ea`、`a26ef39`）

- ptxas：**0 bytes stack frame**，72 registers。
- c01–c06 正確。
- 報告：`reports\c06_phase3_reg.ncu-repz`，量的是 `fd323ea` 的建置。`a26ef39` 只整理寫法，ptxas 結果相同。

| 指標 | y 在外 | register 重用 |
|---|---:|---:|
| Duration | 3.38 ms | **2.00 ms**（1.69×） |
| global load requests | 34,939,392（每個 warp 2064） | 18,688,512（每個 warp **1104**，符合預測） |
| local load requests | 0 | 0（符合預測） |
| Block Limit Registers | 4 | 3 |
| 理論／Achieved Occupancy | 100%／98.23% | 75%／74.06% |
| Stall Long Scoreboard（cycles／指令） | 4.03 | 7.55（+87.20%） |
| SOL Memory／Compute | 84.97%／84.97% | 76.17%／76.17% |
| Breakdown 第一列 | `Pipe Lsu`／`Lsuin Requests` | `Pipe Lsu`／`Lsuin Requests`（不變） |
| 總指令數（Executed） | 109,405,664 | 80,391,072（−26.52%） |

數據來源：Duration 取自 GPU Speed Of Light；global／local load requests 取自 Raw 頁的 `l1tex__t_requests_pipe_lsu_mem_{global,local}_op_ld.sum`；occupancy 取自 Occupancy section；指令數取自 Instruction Statistics。

**分析**：

- 【使用者】speedup 1.69×。每個 warp 的 global load requests 從 2064 降到 1104，local load 為 0，表示累積值確實留在 register。
- 【使用者】一個 thread 72 registers × 256 threads/block = 18,432 registers/block。sm_75 每個 SM 有 64 K registers（CUDA Programming Guide 的 compute capability 表），65,536 / 18,432 = 3.56，所以每個 SM 只放得下 3 個 block，和 `Block Limit Registers` = 3 一致。
- 【參考答案，待改寫】occupancy 的代價：
  - 理論 occupancy = 3 blocks × 8 warps / 32 warps（sm_75 每個 SM 最多 32 個 warp）= 75%。Achieved 74.06% 幾乎等於理論值，表示限制來自 register，而不是排程。
  - occupancy 從 98% 降到 74%，速度卻快了 1.69×：occupancy 只是隱藏延遲的手段，本身不是目標。這一版用較少的 warp 換到較少的 LSU 指令，換算下來是值得的。
- 【參考答案，待改寫】瓶頸仍在 LSU，以及百分比為何下降：
  - 總指令數只少了 26.52%（0.735 倍），時間卻是原來的 0.59 倍，所以不能只看總指令數。瓶頸在 LSU，該看經過 LSU 的工作量：每個 warp 的 global load requests 是 2064 / 1104 = 1.87 倍。
  - LSU 的工作量少了 1.87 倍，速度卻只快 1.69 倍，差額來自 LSU 利用率從 84.97% 降到 76.17%：1.87 × (76.17 / 84.97) ≈ 1.68，和實測的 1.69 吻合（推論）。
  - 【參考答案，待改寫】利用率下降的原因：每個 SM 同時駐留的 warp 從約 31 個降到約 24 個。warp 在等 load 回來時（L2 load hit rate 從 81.30% 降到 20.55%，等待時間較長），能補位發出 LSU 指令的 warp 變少，LSU 就有空檔。
  - 【使用者】驗證：Warp State Statistics 的 `Stall Long Scoreboard` 從 4.03 升到 7.55 cycles／指令（+87.20%），和上面的假設一致。

## 7. 尚未完成

- shared memory 版本（方向 2：pivot tile 放進 shared memory），以及 bank conflict 分析。
- blocking factor 實驗（BF = 32／64／128）：Integer GOPS、device／shared memory 頻寬。
- 多 GPU／多節點版的概念討論（未實作、未實測）。

## 8. 待驗證問題

- H2D 投影比 CPU 端區段短 11.4 ms 的原因（page fault，或第一次 `cudaMemcpy` 的準備成本）。驗證方式：連做兩次 H2D，或改用 `cudaMallocHost`。
- `ncu` 和 Nsight Systems 的 Duration 差異來源（見 4.5）。
