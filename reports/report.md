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
- phase 3 的版本演進見第 5–7 節。

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

  H2D 多出的 11.4 ms 全部在 GPU 開始搬資料之前（`cudaMemcpy` 開始 1874.273 ms，GPU memcpy 開始 1885.673 ms），來源是 pageable memory。成因是**待驗證假設**：page fault，或第一次 `cudaMemcpy` 的準備成本（見第 11 節）。
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
  - 限制（推論）：stall 上升有兩個可能來源，這個量測分不開——warp 變少使延遲較難隱藏，以及 L2 hit rate 下降使每次 load 本身變慢。可用 `__launch_bounds__` 或 `-maxrregcount` 把 register 壓到 64 以下（回到 4 個 block）來分離，前提是陣列不被放進 local memory。

## 7. 第三個優化：pivot tile 放進 shared memory（commit `9888208`）

每個 block 先合作把 `d[y][k]`（程式中的 `pivotCol`）與 `d[k][x]`（`pivotRow`）兩個 64×64 tile 以合併存取載入 shared memory，`__syncthreads()` 一次後，k 迴圈改從 shared memory 讀取。tile 在 k 迴圈中只讀不寫，所以不需要第二個 barrier。

- ptxas：77 registers，**32768 bytes smem**，0 bytes stack frame。
- c01–c06 正確。
- 報告：`reports\c06_phase3_smem.ncu-repz`（同樣是 c06 第 24 輪 phase3，從 ASCII 路徑編譯）。

### 7.1 預測與實測

| 指標 | register 重用 | 預測 | shared memory |
|---|---:|---|---:|
| Duration | 2.00 ms | — | **989.86 µs（2.02×）** |
| global load requests | 18,688,512（每個 warp 1104） | 每個 warp 48 | 812,544（每個 warp **48**） |
| Block Limit Registers／Shared Mem | 3／— | ?／2 | 3／**2** |
| 理論／Achieved Occupancy | 75%／74.06% | 50%／≈ 50% | 50%／48.72% |
| Stall Long Scoreboard（cycles／指令） | 7.55 | 大幅下降 | 0.61（−91.86%） |
| Stall Short Scoreboard（cycles／指令） | 0.03 | 上升 | 0.97 |
| SOL Memory／Compute | 76.17%／76.17% | — | 92.57%／92.57% |
| Breakdown 第一列 | `Pipe Lsu` | — | `Pipe Lsu`（92.57%，+21.52%） |
| k 迴圈的 shared load 指令（每個 warp） | — | 1088 | **576** |
| shared load bank conflicts | — | 0 | 52,106（逐條指令：0） |

數據來源：Duration、SOL 取自 GPU Speed Of Light；global load requests 取自 Raw 頁 `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`；occupancy 取自 Occupancy section；stall 取自 Warp State Statistics；shared load 指令數與逐條 wavefront 取自 Source 頁（`ncu --page source --print-source sass --csv` 匯出）；bank conflicts 取自 Memory Tables → Shared Memory。

### 7.2 分析

- 【使用者】預測的計算：每個 warp 的 global load 為 `d[y][x]` 16 + `d[y][k]` tile（64×64 / 32 / 8 = 16）+ `d[k][x]` tile 16 = 48 條，實測相符。一個 block 用 2 × 64 × 64 × 4 B = 32 KB shared memory，sm_75 每個 SM 最多 64 KB（CUDA Programming Guide 的 compute capability 表），所以每個 SM 只放得下 2 個 block，理論 occupancy 2 × 8 / 32 = 50%，實測 48.72%。
- 【使用者】shared memory 也走 LSU（Nsight Compute Profiling Guide：「The LSU pipeline issues load, store, atomic, and reduction instructions to the L1TEX unit for global, local, and shared memory」），所以 k 迴圈的 load 指令數不會因為改用 shared memory 而減少，改善來自等待時間。
- 【使用者】瓶頸仍是 LSU，利用率從 76.17% 升到 92.57%。
- 【參考答案，待改寫】為什麼 occupancy 降到 50% 仍快 2.02×：
  - 等待從 L1TEX（global）換成 MIO（shared）。Profiling Guide 定義 Long Scoreboard 為等待 L1TEX（local、global、surface、texture）、Short Scoreboard 主要為等待 shared memory。每條指令的 Long Scoreboard 減少 6.94 cycles，Short Scoreboard 只增加 0.94 cycles。
  - 等待變短，較少的 warp 也足以讓 LSU 保持忙碌，所以 LSU 利用率反而上升到 92.57%。
- 【參考答案，待改寫】編譯器的自動向量化（Source 頁）：
  - 有被執行的 shared load 為 4 條 `LDS.U` 與 32 條 `LDS.U.64`，每條 `Instructions Executed` = 270,848 = 16,928 warps × 16，表示 k 迴圈被展開成每次 4 個 k。
  - `pivotCol[offset][k]` 與 `[k+1]` 相鄰，被合併成一條 64 bit 讀取；`pivotRow[k][threadIdx.x]` 不同 k 相隔一整列，每個 k 一條。
  - 每個 warp 共 (4 + 32) × 16 = 576 條，而不是預測的 1088 條。另有 35 條 `LDS` 的執行次數為 0（推論：處理剩餘迴圈次數的路徑，未驗證）。
- 【參考答案，待改寫】bank conflict：
  - 52,106 只佔 shared load wavefronts（9,801,470）的 0.53%【使用者】。
  - Source 頁逐條指令的 `L1 Wavefronts Shared` = `Ideal` = 9,750,528，`Excessive` = 0，所以存取模式本身沒有 bank conflict：`pivotRow` 為同一列連續 32 個 int，`pivotCol` 為 broadcast。
  - 總計數器多出的 50,942 個 wavefront 與 bank conflict 計數相近。重跑 3 次得 52,664／50,876／50,365（store：14,432／14,321／14,594），每次不同，而固定的存取模式應產生固定的 conflict 數，所以推論來自執行時的競爭。
  - 【待驗證假設】Turing 的 L1 與 shared memory 共用 data bank，搬 tile 的 global load 可能和 shared memory 存取競爭。未在文件中查到該計數器包含此類競爭。
  - 結論：判斷 bank conflict 以 Source 頁逐條指令的 `Excessive` 為準。
- 【待寫】下一步的方向：LSU 已接近滿載，要再變快必須減少 LSU 指令數（例如更寬的 shared load，或讓一個 thread 負責多個 x 以重用 `pivotRow` 的讀取）。

## 8. 第四個嘗試：`pivotCol` 改用 `int4` 讀取（commit `5d56c03`）

`pivotCol` 宣告加上 `__align__(16)`；k 迴圈每次前進 4，以 `*reinterpret_cast<const int4*>(&pivotCol[offset][k_shared])` 一次讀 4 個值，再依序用 `.x .y .z .w` 更新 `d_yx`；`pivotRow` 照舊每個 k 一條。

- ptxas：**80 registers**（shared memory 版 77），32768 bytes smem，0 bytes stack frame。
- c01–c06 正確；`compute-sanitizer --tool memcheck`（c03）0 errors，128 bit 讀取沒有對齊錯誤。
- 報告：`reports\c06_phase3_int4.ncu-repz`（c06 第 24 輪 phase3，從 ASCII 路徑編譯），baseline 為 `c06_phase3_smem`。

### 8.1 先做的對照實驗：只加 `__align__(16)`

k 迴圈不改、只在 `pivotCol` 加 `__align__(16)`，`cuobjdump -sass` 的 phase3 靜態指令與 shared memory 版完全相同：

| 版本 | `LDS.U` | `LDS.U.64` | `LDS.U.128` |
|---|---:|---:|---:|
| shared memory 版 | 23 | 48 | 0 |
| 只加 `__align__(16)` | 23 | 48 | 0 |
| int4 版 | 4 | 0 | 16 |

7.2 的推測「編譯器無法證明 16 bytes 對齊，所以只合併成 64 bit」不成立：對齊確定後編譯器仍只用 `LDS.U.64`，原因未查明。要拿到 128 bit 讀取必須在程式中明確寫出。

### 8.2 預測與實測

| 指標 | shared memory | 預測 | int4 |
|---|---:|---|---:|
| Duration | 989.86 µs | 509–550 µs | **964.51 µs（1.03×）** |
| shared load 指令（每個 warp） | 576 | 320 | **320**（`LDS.U` 64 + `LDS.U.128` 256） |
| 每條指令的 shared wavefront | `LDS.U` 1、`LDS.U.64` 1 | `LDS.U.128`：1 或 4 | `LDS.U` 1、**`LDS.U.128` 2** |
| shared load wavefront（每個 warp） | 576 | — | **576** |
| shared load wavefront 總數（Raw） | 9,801,470 | — | 9,790,266 |
| `L1 Wavefronts Shared Excessive`（逐條） | 0 | — | 0 |
| SOL Memory／Compute | 92.57%／92.57% | — | 84.84%／58.62% |
| `Pipe Lsu`（`sm__inst_executed_pipe_lsu`） | 92.71% | — | 58.98% |
| L1: Lsuin Requests | 92.57% | — | 58.62% |
| L1/TEX Cache Throughput | 93.17% | — | 95.34% |
| L1: Data Pipe Lsu Wavefronts | 46.59% | — | 47.67% |
| DRAM: Cycles Active | 82.93% | — | 84.84% |
| DRAM 讀／寫 | 70.10 MB／34.61 MB | — | 70.09 MB／34.63 MB |
| Block Limit Registers／Shared Mem | 3／2 | 3／2 | 3／2 |
| Achieved Occupancy | 48.72% | — | 48.78% |
| Executed Instructions | 48,854,208 | — | 44,419,072（−9%） |
| Stall Long Scoreboard（cycles／指令） | 0.61 | — | 1.53（+148.59%） |
| Stall Short Scoreboard（cycles／指令） | 0.97 | — | 1.02 |

數據來源：Duration、SOL、Breakdown 取自 GPU Speed Of Light（`--page details --section SpeedOfLight --print-details all`）；指令與逐條 wavefront 取自 Source 頁 CSV（每條 `LDS` 的 `Instructions Executed` 皆為 270,848 = 16,928 warps × 16）；wavefront 總數、DRAM 位元組、Executed Instructions 取自 Raw 頁（`l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum`、`dram__bytes_{read,write}.sum`、`smsp__inst_executed.sum`）；occupancy 取自 Occupancy section；stall 取自 Warp State Statistics。

PC sampling 的 stall 樣本數（Raw 頁 `smsp__pcsamp_warps_issue_stalled_*`，只列變化較大的項目）：

| stall | shared memory | int4 | 變化 |
|---|---:|---:|---:|
| 總樣本數 | 18,029 | 17,548 | −481 |
| MIO Throttle | 7,045 | 4,976 | −2,069 |
| Long Scoreboard | 1,291 | 3,335 | +2,044 |
| Not Selected | 2,010 | 1,349 | −661 |

Long Scoreboard 樣本在 Source 頁的位置（Profiling Guide：樣本記在等待結果的 consumer 指令上）：

| 指令 | shared memory | int4 |
|---|---:|---:|
| k 迴圈中第一次使用 `d_yx` 的 `IMNMX` | 741 | 1,574 |
| 載入 tile 時寫 shared memory 的 `STS` | 414 + 118 | 1,563 + 159 |

### 8.3 分析

- 【使用者】預測：LSU 利用率 92.57%，若時間由 LSU 指令數決定，利用率維持不變時為 989.86 × 320 / 576 ≈ 550 µs；利用率拉到 100% 的下限為 989.86 × 0.9257 × 320 / 576 ≈ 509 µs。
- 【使用者】occupancy：shared memory 仍是 32 KB，每個 SM 只放得下 2 個 block；register 增為 80 後 65536 / (80 × 256) = 3.2，仍可放 3 個 block，所以 occupancy 不受影響。
- 【使用者】wavefront：`LDS.U.128` 每條 `Instructions Executed` 270,848 = 16,928 warps × 16，`L1 Wavefronts Shared` = `Ideal` = 541,696，每條 2 個 wavefront、沒有 bank conflict。原本的兩個假設（只看需要的位元組數 → 1 個；固定以 1/4 warp 為單位 → 4 個）都不成立。
- 【使用者】每個 warp 的 shared load wavefront：shared memory 版 64 + 512 × 1 = 576，int4 版 64 + 256 × 2 = 576；576 × 16,928 = 9,750,528，與 Raw 頁的 9,801,470、9,790,266 相近。
- 【參考答案，待改寫】為什麼幾乎沒變快：
  - shared memory 版受限的不是 LSU 指令數，而是處理 shared memory wavefront 的吞吐量。一個 wavefront 在 L1TEX 的處理階段佔一個 cycle（Profiling Guide 2.3.7），而每個 warp 的 wavefront 數兩版都是 576。
  - 指令數減少只降低了指令發出端的負擔：`Pipe Lsu`、Lsuin Requests 從 92% 降到 59%，Duration 只少 2.6%。
  - 預測失準的原因：假設「時間正比於 LSU 指令數」，但與時間成正比的應是 wavefront 數（推論，前提為 wavefront 吞吐量是瓶頸；驗證方式見下一版）。
- 【參考答案，待改寫】為什麼 Long Scoreboard 增加：
  - 指令只少 9%，Long Scoreboard 樣本卻多了 2,044 個，所以不是平均值的分母變小，而是等待 global load 的時間真的變長。
  - MIO Throttle 減少的樣本數（−2,069）與 Long Scoreboard 增加的（+2,044）幾乎相同。Profiling Guide 定義 MIO Throttle 為等待 MIO 指令佇列不再滿，shared memory 指令是其中之一。
  - Long Scoreboard 樣本落在 global load 的使用端：載入 tile 時的 `STS`、讀完初始 `d_yx` 後第一次使用它的 `IMNMX`，兩者都約增為 2 倍。global store 不需等待結果，不會出現在此（推論）。
  - 量測事實只支持「等待原因從 MIO Throttle 轉成 Long Scoreboard，總時間與 DRAM 讀寫量幾乎不變」。為什麼 tile 與 `d_yx` 的 global load 要等更久，確切機制未查明【待驗證假設】：每個 SM 同時有 2 個 block，一個在跑 k 迴圈時另一個可能在載入 tile，兩者共用 L1TEX，可能互相競爭；DRAM 本來就忙（82.93% → 84.84%）。
- 【待寫】下一步的方向：要減少的是 wavefront 數，而不是指令數；同時 DRAM 已接近 85%，可能成為下一個上限。

## 9. 第五個優化：一個 thread 負責 2 個 x（commit `564cc36`）

phase3 改用自己的 block 形狀 `BLOCK_X_P3 × BLOCK_Y_P3` = 32 × 8（phase1、phase2 維持 64 × 4）。每個 thread 負責 8 個 y、2 個 x（x 與 x+32），累積值為 `d_yx[8][2]`；k 迴圈每次前進 4，每列 y 讀一次 `int4` 的 `pivotCol`，供 2 個 x 共用。

- ptxas：**64 registers**（int4 版 80），32768 bytes smem，0 bytes stack frame。
- SASS（k 迴圈每次迭代）：`LDS.U` 8 條、`LDS.U.128` 8 條（int4 版 4 條、16 條），`IMNMX` 64 條不變。
- c01–c06 正確；`compute-sanitizer --tool memcheck`（c03）0 errors。
- 報告：`reports\c06_phase3_x2.ncu-repz`（c06 第 24 輪 phase3，從 ASCII 路徑編譯），baseline 為 `c06_phase3_int4`。

### 9.1 預測

原理：`pivotCol[y][k]` 對同一列 y 的所有 x 都相同，且為 broadcast 讀取，一個 wavefront 服務整個 warp 涵蓋的 x。現在一個 warp 只涵蓋 32 個 x，同一列 y 的 `pivotCol` 由兩個 warp 各讀一次；一個 thread 負責 2 個 x 後，一個 warp 涵蓋整個 tile 寬度（64），每列只讀一次。`pivotRow[k][x]` 每個 x 都不同，總量省不了。

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
- 【使用者】兩個上限（推論，前提：Duration 與 wavefront 數成正比；`DRAM: Cycles Active` 近似 DRAM 頻寬利用率；DRAM 讀寫量不變）：
  - shared memory wavefront：964.51 × 3072 / 4608 ≈ **643 µs**。
  - DRAM 忙到 100%：964.51 × 0.8484 ≈ **818 µs**。
  - Duration 由較慢者決定，所以預測約 818 µs 以上（約 1.18×）。做完後 phase3 的瓶頸可能轉為 DRAM，要再變快須減少 DRAM 流量（見 10.1）。

### 9.2 實測

| 指標 | int4 | 預測 | x2 |
|---|---:|---|---:|
| Duration | 964.51 µs | ≥ 818 µs | **946.91 µs（1.02×）** |
| block 的 k 迴圈 shared load wavefront（Source 頁逐條加總 ÷ 2116 個 block） | 4608 | 3072 | **3072**（`LDS.U` 2,166,784 + `LDS.U.128` 4,333,568） |
| shared load wavefront 總數（Raw） | 9,790,266 | — | 6,531,455（−33.3%） |
| L1/TEX Cache Throughput | 95.34% | — | 70.15% |
| L1: Data Pipe Lsu Wavefronts | 47.67% | — | 35.07% |
| `Pipe Lsu`／L1: Lsuin Requests | 58.98%／58.62% | — | 50.81%／50.81% |
| DRAM: Cycles Active（= DRAM Throughput） | 84.84% | 接近 100% | **86.76%** |
| DRAM 讀／寫 | 70.09 MB／34.63 MB | 不變 | 70.10 MB／34.61 MB |
| L2 hit rate | 49.46% | — | 49.49% |
| Block Limit Registers／Shared Mem | 3／2 | — | 4／2 |
| Theoretical／Achieved Occupancy | 50%／48.78% | 50% | 50%／48.92% |
| Executed Instructions | 44,419,072 | — | 43,081,760（−3.0%） |
| Stall MIO Throttle（cycles／指令） | 2.40 | — | 1.34（−44.07%） |
| Stall Long Scoreboard（cycles／指令） | 1.53 | — | 2.77（+81%） |
| Stall Short Scoreboard（cycles／指令） | 1.02 | — | 0.90 |

數據來源：Duration、SOL、Breakdown 取自 GPU Speed Of Light；逐條 wavefront 取自 Source 頁；wavefront 總數、DRAM 位元組、L2 hit rate、Executed Instructions、block limit 取自 Raw 頁（`l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum`、`dram__bytes_{read,write}.sum`、`lts__t_sector_hit_rate.pct`、`smsp__inst_executed.sum`、`launch__occupancy_limit_*`）；stall 取自 Warp State Statistics。Raw 總數換算成每個 warp：6,531,455 / 16,928 ≈ 386，與 3072 / 8 = 384 相近。

PC sampling 的 stall 樣本數（Raw 頁 `smsp__pcsamp_warps_issue_stalled_*`）：

| stall | int4 | x2 | 變化 |
|---|---:|---:|---:|
| 總樣本數 | 17,548 | 17,211 | −337 |
| MIO Throttle | 4,976 | 2,627 | −2,349 |
| Long Scoreboard | 3,335 | 5,803 | +2,468 |
| Short Scoreboard | 2,097 | 1,752 | −345 |
| Not Selected | 1,349 | 1,241 | −108 |
| Barrier | 364 | 519 | +155 |

Long Scoreboard 樣本在 Source 頁的位置：

| 指令 | int4 | x2 |
|---|---:|---:|
| k 迴圈中第一次使用 `d_yx` 的 `IMNMX` | 1,574 | 2,942（佔 50.7%） |
| 載入 tile 時寫 shared memory 的 `STS` | 1,563 + 159 | 2,216 + 632（佔 49.1%） |

### 9.3 分析

- 【使用者】用利用率反推 100% 時的時間下限：

  | | Duration | L1/TEX 下限 | DRAM 下限 |
  |---|---:|---:|---:|
  | int4 | 964.51 µs | 964.51 × 0.9534 ≈ 919.6 µs | 964.51 × 0.8484 ≈ 818.3 µs |
  | x2 | 946.91 µs | 946.91 × 0.7015 ≈ 664.3 µs | 946.91 × 0.8676 ≈ 821.5 µs |

  x2 版的 L1 下限比 9.1 預測的 643 µs 多約 21 µs；DRAM 下限與預測的 818 µs 只差約 3 µs。實際 Duration 最接近 DRAM 的下限。
- 【使用者】int4 版的 L1 下限（919.6 µs）和 DRAM 下限（818.3 µs）都接近實際 Duration，是 2 個資源同時接近滿載。x2 版把 wavefront 減少約 1/3，L1 下限降到 664.3 µs，但 Duration 只降 1.8%，因為被 DRAM 卡住。所以「瓶頸是 wavefront 數」要修正為：瓶頸是 DRAM。
  - 【參考答案，待改寫】補充：int4 版最接近 Duration 的是 L1 下限，DRAM 次之；x2 版換成 DRAM 最接近，但 DRAM 只到 86.76%，Duration 比 DRAM 下限仍多約 125 µs（13%）。所以較精確的說法是「最接近的上限從 shared memory wavefront 換成 DRAM，但 DRAM 也沒有一直保持忙碌」，剩下的差距見下面 Long Scoreboard 與 DRAM 利用率兩點。DRAM 讀寫量（70.10／34.61 MB）與 int4 版相同，所以 DRAM 下限幾乎不變，和每個 block 讀寫 global 的量沒有改變一致（推論）。
- 【參考答案，待改寫】為什麼 Long Scoreboard 增加：
  - MIO Throttle 減少的樣本數（−2,349）與 Long Scoreboard 增加的（+2,468）幾乎相同，總樣本數與 Duration 也幾乎不變，和 8.3 的情形一樣：減少的 shared memory 等待，轉成了 global load 的等待。
  - Long Scoreboard 樣本幾乎全部落在兩個 consumer：`__syncthreads()` 之後讀 `d_yx` 的 `LDG` 的第一個使用者 `IMNMX`（1,574 → 2,942），以及載入 tile 的前兩條 `STS`（1,722 → 2,848）。warp 現在主要在等 global load。
  - 推論：DRAM 讀寫量不變、DRAM 已近 87%，資料送達的速度大致固定；k 迴圈變快後，warp 更早走到需要 global 資料的指令，等待時間就變長。前提是 DRAM 的有效頻寬是主要限制；驗證方式見下一點的實驗。
- 【參考答案，待改寫】【待驗證假設】為什麼 DRAM 只到 87%：每個 block 依序執行「載入 tile → `__syncthreads()` → 讀 `d_yx` → k 迴圈 → 寫回」，DRAM 請求只出現在第 1、3、5 段。每個 SM 只放得下 2 個 block（受 shared memory 限制），兩個 block 同時在 k 迴圈時，這個 SM 對 DRAM 沒有需求，因此 DRAM 平均只忙 87%。另外，讀 `d_yx` 排在 barrier 之後，延遲完全沒被遮住，這和第一個 `IMNMX` 累積一半 long_sb 樣本一致；`d_yx` 是每個 thread 自己的 register，邏輯上不需要等 barrier（推論，依據程式碼）。
  - 驗證實驗：把讀 `d_yx` 的迴圈移到 `__syncthreads()` 之前、與載入 tile 並排。預期：(1) 該 `IMNMX` 的 long_sb 樣本大幅減少；(2) Duration 往 821 µs 靠近、DRAM 利用率提高；(3) register 可能增加，要確認沒有 local memory、occupancy 仍為 50%。若 (1) 成立而 (2) 不明顯，表示主因是「k 迴圈期間 DRAM 閒置」，需要減少總流量（BF 實驗，見 10.1）或讓載入與計算重疊（double buffering）。
- 對第 11 節「L1/TEX Cache Throughput 為 Data Pipe Lsu Wavefronts 的 2 倍」的檢驗（量測事實）：wavefront 總數減少 33.3% 後，兩者同步下降（95.34% → 70.15%、47.67% → 35.07%），仍恰為 2 倍（70.15 ≈ 2 × 35.07）。L1/TEX Cache Throughput 隨 shared wavefront 數變化這點得到支持；但 Duration 沒有同比例下降，因為 DRAM 上限擋在前面，所以這次實驗無法單獨證明「wavefront 數是 int4 版唯一的瓶頸」。

## 10. 尚未完成

- 驗證 9.3 的假設：讀 `d_yx` 移到 `__syncthreads()` 之前。
- 視結果決定是否試 32 × 4（wavefront 2560、occupancy 25%、累積值 32 個）。目前 Duration 已不由 wavefront 決定，預期效益有限（推論）。
- blocking factor 實驗（BF = 32／64／128）：Integer GOPS、device／shared memory 頻寬；是減少 DRAM 流量的主要手段（見 10.1）。
- 多 GPU／多節點版的概念討論（未實作、未實測）。

### 10.1 DRAM 流量與 BF 的關係

- 每個 block 把 `d[y][x]` 從 DRAM 讀進 register 一次、寫回一次，中間在 k 迴圈中用了 BF 次；pivot tile 的每個值也被用 BF 次。每輪整個矩陣進出 DRAM 一次，共 n / BF 輪，所以 DRAM 總流量約與 n² × (n / BF) 成正比，總計算量 n³ 與 BF 無關（推論）。
- c06 的量測對照（int4 版）：
  - 【使用者】矩陣 3008 × 3008 × 4 B ≈ 36.19 MB；DRAM 讀取 70.09 MB 約為其 2 倍。
  - phase3 的 grid 為 46 × 46（第 r 個 block 列與行由 phase1、phase2 負責），46 × 46 × 64 × 64 × 4 B ≈ 34.67 MB，與 DRAM 寫入 34.63 MB 相符。
  - 讀取中約 34.7 MB 為 `d[y][x]`，其餘約 35 MB 為 pivot tile。若兩個 pivot tile 全部打到 DRAM，讀取應約為 3 倍（約 104 MB），所以約一半的 pivot 讀取在 L2 命中（推論）。哪個 tile 命中【待驗證假設】：看 Memory Workload Analysis 的 L2 hit rate，或 Source 頁逐條 `LDG` 的 L2 sectors。
- 限制：兩個 tile 的 shared memory 用量為 2 × BF² × 4 B，BF = 128 時 128 KB 超過 sm_75 每個 SM 的 64 KB；BF 越大，phase1 只有一個 block 的序列部分越長（推論，blocking factor 實驗時以 nsys 看各 phase 時間比例驗證）。

## 11. 待驗證問題

- H2D 投影比 CPU 端區段短 11.4 ms 的原因（page fault，或第一次 `cudaMemcpy` 的準備成本）。驗證方式：連做兩次 H2D，或改用 `cudaMallocHost`。
- `ncu` 和 Nsight Systems 的 Duration 差異來源（見 4.5）。
- register 重用版 `Stall Long Scoreboard` 上升的兩個來源（occupancy 下降 vs L2 hit rate 下降）如何分離（見 6.2）。
- shared memory 版 bank conflict 總計數器每次執行不同的原因（見 7.2）。
- 編譯器只把 `pivotCol` 的讀取合併成 `LDS.U.64` 而非 `LDS.U.128` 的原因：「無法證明 16 bytes 對齊」已被 8.1 推翻，實際原因未查明。
- broadcast 的 `LDS.U.128` 為什麼需要 2 個 wavefront（`LDS.U.64` 只要 1 個）：Profiling Guide 沒有說明 shared memory 各存取寬度的 wavefront 規則。
- L1/TEX Cache Throughput 恰為 L1: Data Pipe Lsu Wavefronts 的 2 倍（93.17 = 2 × 46.59、95.34 = 2 × 47.67），且大於 Breakdown 中列出的每一項（int4 版最大為 Lsuin Requests 58.62%）。Profiling Guide 說 throughput 取組成計數器的最大值，但沒有這兩個 metric 的定義。推論：throughput 用的組成項與 breakdown 顯示的峰值不同，95.34% 反映的是 wavefront 吞吐量；未驗證。x2 版（9.3）中兩者同步下降且仍為 2 倍（70.15 ≈ 2 × 35.07），支持它隨 wavefront 數變化；2 倍的來源仍未查明。
- x2 版 DRAM 只到 86.76%、Duration 比 DRAM 下限多約 125 µs 的原因（9.3 的假設：k 迴圈期間 DRAM 閒置、`d_yx` 讀取排在 barrier 之後）。驗證方式：讀 `d_yx` 移到 `__syncthreads()` 之前。
