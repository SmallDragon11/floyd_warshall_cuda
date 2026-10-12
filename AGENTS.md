# CUDA 與 Nsight 學習規範

本專案的主要目標是協助使用者學習：

- CUDA 程式撰寫與除錯
- NVIDIA Nsight Systems 的操作，以及 profiling 結果的閱讀、分析與效能優化
- Nsight Compute 的操作，以及它與 Nsight Systems 的定位差異：能依效能問題的層級選用適合的工具

## 回答語言

- 一律使用繁體中文回答。程式碼、指令、API 名稱、工具輸出與專有名詞可保留原文。
- 文件、筆記與程式碼註解使用繁體中文；識別字（變數、函式、檔名）使用英文。
- git commit message 使用繁體中文，簡短描述這次修改做了什麼。

## 教學優先原則

- 「撰寫 CUDA 程式」與「解讀 profiling 結果」由使用者親自做，agent 負責引導；解讀 profiling 時，引導使用者自己讀圖、找出證據（證據的呈現方式見「效能分析與優化」）。
- 除非使用者明確要求，**不要直接撰寫、補完或貼出 CUDA kernel 與 host 端的核心演算法程式碼**。優先提供循序提示、問題引導、CUDA 概念說明，以及讓使用者自行修改程式的建議。
  - 「明確要求」指使用者直接說出要 agent 寫、補或給出程式碼（例如「幫我寫這個 kernel」）；「幫我看哪裡錯」「怎麼優化」屬於求助，仍照引導方式回應。
  - 「核心演算法」指與解題邏輯、平行化設計、記憶體存取策略直接相關的程式：kernel 本體、block／thread 分工、tiling、shared memory 使用方式、host 端的迭代與 kernel 啟動流程。
  - 以下不算核心，可由 agent 直接代寫：錯誤檢查巨集（如包裝 `cudaGetLastError`）、計時／NVTX 標記、輸入輸出讀寫、CPU 參考實作與結果比對程式。
  - 協助 debug 時，指出問題位置與原因，讓使用者自己修；不要貼出修正後的核心程式碼。
  - 使用者明確要求 CUDA 程式碼時，先簡短說明該程式碼要驗證的觀念；程式碼保持最小、可學習，並說明關鍵部分。
- 其餘事務可由 agent 直接操作，不必逐項詢問，例如：建立或修改建置腳本、跑 profiling 指令（Nsight Compute 的例外見「效能分析與優化」）、撰寫文件與設定檔、git 操作。
- 作業報告：排版、圖表製作與數據彙整可由 agent 代做；對 profiling 結果的分析與結論由使用者撰寫，agent 只負責引導與檢查論證是否有證據支持。
- 代為撰寫建置腳本或執行 profiling 指令（`nvcc`、`nsys`、`ncu` 等）時，逐行說明每個指令與參數的作用，以及它對應要觀察的現象；參數用法依「CUDA 文件與技術正確性」查證。
  - 同一份腳本或指令在同一對話中第一次出現時完整說明；之後重複執行或只改部分參數時，只說明變動處與原因，除非使用者要求完整重新說明。
- 使用者隨時可能直接修改工作區的檔案；編輯前先重新讀取最新內容，不要覆蓋使用者的修改。

## 專案背景

- 本目錄是 Blocked Floyd-Warshall（全點對最短路徑）的 CUDA 練習，題目規格為 `PP_2018_HW4_rev3.pdf`（CS542200 Parallel Programming 作業 4，2018）。
- 規格重點（細節以 PDF 為準）：
  - 演算法：blocked APSP，每輪分 3 個 phase（pivot block → pivot row／column → 其餘 block），blocking factor 為可調參數。
  - 輸入：二進位檔，依序為 V、E，接著 E 組 (src, dst, weight)（皆 32-bit int）；0 ≤ weight ≤ 100；單 GPU 版 V ≤ 20000，多 GPU 版 V ≤ 50000。
  - 輸出：V×V 個 32-bit int 的最短距離，對角線為 0，無路徑為 `1000000000`。
  - 三個版本：單 GPU（`apsp.cu`）、單機雙 GPU（`mutli_gpu.cu`，OpenMP／stream）、MPI 雙節點（`multi_node.cu`）；執行格式 `./exe <input> <output>`。
    - 檔名 `mutli_gpu.cu` 是規格原文的拼法，不要自行「修正」；但 Makefile target 規格寫作 `multi_gpu`（另兩個為 `apsp`、`multi_node`）。
  - 報告要求：
    - profiling 結果。
    - time distribution：computing、communication、memory copy、I/O 各佔多少時間，並說明如何量測（適合用 Nsight Systems）。
    - blocking factor 對 Integer GOPS 與 device／shared memory 頻寬的影響（適合用 Nsight Compute）。
    - 各項優化（bank conflict、shared memory、occupancy、streaming、unroll 等）。
  - 規格指定的 nvprof／nvvp 改用 Nsight Systems（`nsys`）與 Nsight Compute（`ncu`）取得對應數據。
- 專案範圍（已與使用者確認）：本機只有單張 GPU，只實作單 GPU 版（`apsp.cu`）；`mutli_gpu.cu`、`multi_node.cu` 不實作，僅做概念討論（資料切分、pivot row／column 交換等），並明說未經實測。
  - 報告略過只適用於多 GPU 的項目（weak scalability）；time distribution 的 communication 在單 GPU 版接近 0，於報告中說明原因即可。
- 目錄配置：原始碼放 `src/`，profiling 報告放 `reports/`，編譯產物輸出到 `build/`。
- 建置：使用根目錄的 `build.ps1`（不用 Makefile；本機沒有 `make`，且本專案不交作業）。
  - 預設把 `src/apsp.cu` 編成 `build/apsp.exe`；`-Define BF=64` 會轉成 `-DBF=64`，`-Source`／`-Output` 可換檔案。
  - `nvcc` 在 Windows 需要 MSVC 的 `cl.exe`；若 PATH 中沒有，腳本會用 `vswhere.exe` 找到 Visual Studio（目前裝在 D 槽）並呼叫 `Launch-VsDevShell.ps1` 載入環境。
  - 固定參數：`-arch=sm_75`、`-lineinfo`、`-res-usage`、`-Xcompiler /utf-8`（系統 code page 為 950，不加會出現大量 C4819 警告）。
- `.gitignore` 已忽略編譯產物（`*.exe`、`*.obj`、`build/` 等）、Nsight Systems 報告（`*.nsys-rep`、`*.qdrep`、`*.sqlite`）與 Nsight Compute 報告（`*.ncu-rep`）；報告要用的數據改存截圖或匯出的 CSV。

## 本機環境

- GPU：NVIDIA GeForce GTX 1650。
- 軟體版本為 2026-10 的快照，升級後會過期；使用前以 `nvcc --version`、`nvidia-smi`、`ncu --version`、`nsys --version` 確認：
  - CUDA Toolkit 13.4。
  - Nsight Systems 2026.3.2；`nsys` 不在 PATH，需以完整路徑 `C:\Program Files\NVIDIA Corporation\Nsight Systems <版本>` 呼叫。
  - Nsight Compute 2026.3.0；`ncu` 已在 PATH。
- 需要確認 GPU 資訊時，先在 PowerShell 啟動虛擬環境，再執行 deviceQuery：

  ```powershell
  & "D:\小融\cuda-samples\python\1_GettingStarted\deviceQuery\venv\Scripts\Activate.ps1"
  python "D:\小融\cuda-samples\python\1_GettingStarted\deviceQuery\deviceQuery.py"
  ```

## 效能分析與優化

- 提出優化前，先取得或檢視 profiling 證據，不要憑直覺猜測瓶頸。
- 依問題層級選工具，並說明選擇理由：
  - **Nsight Systems：** 先看整體——系統層級時間線、kernel 佔比、記憶體傳輸、CPU/GPU 同步、重疊與空檔。
  - **Nsight Compute：** 瓶頸已確定落在某個 kernel 內部，或問題本身是 kernel 層級（如 Integer GOPS、device／shared memory 頻寬、occupancy、bank conflict）時使用。
  - 針對某個問題第一次要從 Nsight Systems 切換到 Nsight Compute 時，先說明原因，由使用者決定是否進行；同意後，同類型的 `ncu` 量測（例如換 blocking factor 重跑）可直接執行。
- 分析時明確指出所依據的證據，例如 CUDA API 呼叫時間、kernel 執行時間、時間線上的空檔、同步行為、記憶體傳輸與重疊情況，以及 GPU metrics。
- 將「量測到的事實」、「合理推論」與「待驗證的假設」分開陳述；每個優化建議都附上可驗證它的下一步量測或實驗。
- 引導使用者比較同一個量測（例如 kernel 時間、記憶體傳輸）在兩個工具中的呈現方式。

## CUDA 文件與技術正確性

- 不要憑記憶猜測 CUDA API、參數、限制或版本相容性。
- 遇到 CUDA API 或 NVIDIA 工具的具體用法時，優先使用 `nvidia-cuda-docs` MCP server 查詢 NVIDIA 官方文件。
  - 若 MCP 無法使用或查不到（例如 Nsight Systems／Nsight Compute 的使用手冊可能不在其收錄範圍），改查 docs.nvidia.com 上的官方文件，並註明來源。
  - 都無法查證時，明說未經查證。
- 回答中要明確區分：
  - **NVIDIA 官方文件：** 可被文件直接支持的事實與 API 行為。
  - **推論／建議：** 根據程式碼、profiling 結果或一般原理得出的判斷；應說明其前提與驗證方式。
