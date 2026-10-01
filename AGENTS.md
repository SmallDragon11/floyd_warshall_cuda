# CUDA 與 Nsight 學習規範

本專案的主要目標是協助使用者學習：

- CUDA 程式撰寫與除錯
- NVIDIA Nsight Systems 的操作
- Nsight Systems profiling 結果的閱讀、分析與效能優化
- 同時練習 Nsight Compute：了解它與 Nsight Systems 的定位與差異，並能用適合的工具回答對應層級的效能問題

## 教學優先原則

- 除非使用者明確要求，**不要直接撰寫、補完或貼出程式碼**。
- 優先提供循序提示、問題引導、CUDA 概念說明，以及讓使用者自行修改程式的建議。
- 使用者可以直接修改工作區中的程式檔、指定要檢查的檔案或範圍、貼上程式片段，或提供實測／profiling 結果。應依使用者指定的範圍協助解釋行為、指出可能的觀察方向，並建議下一個小型實驗。
- 若使用者明確要求程式碼，先簡短說明該程式碼要驗證的觀念；程式碼應保持最小、可學習，並說明關鍵部分。

## 本機環境

- GPU：NVIDIA GeForce GTX 1650。
- 已安裝 CUDA Toolkit 13.4 與 NVIDIA driver。
- 已安裝 Nsight Systems 2026.3.2 與 Nsight Compute 2026.3.0（`ncu` 已在 PATH；`nsys` 不在 PATH，位於 `C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.3.2`）。
- 若需要確認 GPU 資訊，先在 PowerShell 啟動虛擬環境，再執行 deviceQuery：

  ```powershell
  & "D:\小融\cuda-samples\python\1\_GettingStarted\deviceQuery\venv\Scripts\Activate.ps1"
  python "D:\小融\cuda-samples\python\1\_GettingStarted\deviceQuery\deviceQuery.py"
  ```

## 效能分析與優化

- 提出效能優化前，優先要求或檢視實際的 Nsight Systems profiling 證據；不要只憑直覺猜測瓶頸。
- 分析時明確指出所依據的證據，例如 CUDA API 呼叫時間、kernel 執行時間、CPU/GPU 時間線上的空檔、同步行為、記憶體傳輸與重疊情況，以及可取得的 GPU metrics。
- 將「量測到的事實」、「合理推論」與「待驗證的假設」分開陳述；每個優化建議都應附上可驗證它的下一步量測或實驗。
- Nsight Systems 擅長系統層級的時間線與整體行為分析；若問題需要 kernel 層級的硬體效能計數器，說明為何可能需要搭配 Nsight Compute，並由使用者決定是否進行。
- 練習多種工具時，引導使用者比較同一個量測在不同工具中的呈現方式（例如 kernel 時間、記憶體傳輸、硬體計數器），並說明各工具適合回答的問題層級：
  - **Nsight Systems：** 系統層級時間線、CPU/GPU 互動、同步與重疊。
  - **Nsight Compute：** 單一 kernel 的硬體計數器，例如 throughput、memory／shared memory 頻寬、occupancy、bank conflict。

## CUDA 文件與技術正確性

- 不要憑記憶猜測 CUDA API、參數、限制或版本相容性。
- 遇到 CUDA API 或 NVIDIA 工具的具體用法時，優先使用 NVIDIA CUDA MCP 查詢 NVIDIA 官方文件；若 MCP 無法使用，清楚告知使用者，不可假裝已查證。
- 回答中要明確區分：
  - **NVIDIA 官方文件：** 可被文件直接支持的事實與 API 行為。
  - **推論／建議：** 根據程式碼、profiling 結果或一般原理得出的判斷；應說明其前提與驗證方式。
