# 題目規格摘要

來源：`PP_2018_HW4_rev3.pdf`（CS542200 Parallel Programming 作業 4，2018）。細節以 PDF 為準；本檔是為了省去每次讀 PDF 的摘要。

## 演算法

- Blocked Floyd-Warshall（APSP），每輪分 3 個 phase：pivot block → pivot row／column → 其餘 block。
- blocking factor 為可調參數（`build.ps1 -Define BF=64` 會轉成 `-DBF=64`）。

## 輸入／輸出

- 輸入：二進位檔，依序為 V、E，接著 E 組 (src, dst, weight)，皆 32-bit int；0 ≤ weight ≤ 100；單 GPU 版 V ≤ 20000，多 GPU 版 V ≤ 50000。
- 輸出：V×V 個 32-bit int 的最短距離，對角線為 0，無路徑為 `1000000000`。
- 執行格式：`./exe <input> <output>`。
- 讀寫實作在 `src/graph_io.h`，`apsp.cu` 與 `tools/` 共用。

## 三個版本

| 版本 | 原始碼 | Makefile target |
|---|---|---|
| 單 GPU | `apsp.cu` | `apsp` |
| 單機雙 GPU（OpenMP／stream） | `mutli_gpu.cu` | `multi_gpu` |
| MPI 雙節點 | `multi_node.cu` | `multi_node` |

`mutli_gpu.cu` 是規格原文的拼法，與 target 名稱 `multi_gpu` 不同；兩者都照規格原樣使用。

## 本專案範圍（已與使用者確認）

- 本機只有單張 GPU，只實作 `apsp.cu`。
- `mutli_gpu.cu`、`multi_node.cu` 不實作，只做概念討論（資料切分、pivot row／column 交換等），並明說未經實測。

## 報告要求

規格指定的 nvprof／nvvp 改用 Nsight Systems（`nsys`）與 Nsight Compute（`ncu`）取得對應數據。

- profiling 結果。
- time distribution：computing、communication、memory copy、I/O 各佔多少時間，並說明如何量測（適合用 Nsight Systems）。
  - 單 GPU 版的 communication 接近 0，報告中說明原因即可。
- blocking factor 對 Integer GOPS 與 device／shared memory 頻寬的影響（適合用 Nsight Compute）。
- 各項優化（bank conflict、shared memory、occupancy、streaming、unroll 等）。
- weak scalability 只適用於多 GPU，略過。
