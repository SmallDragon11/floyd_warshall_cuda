# 編譯 tools/ 的 CPU 工具，並在 testcases/ 產生測資
#   cXX.in／cXX.out：正確性測資與 ref.exe 算出的標準答案
#   pXX.in：效能測資（V 太大，CPU 參考實作跑不完，只用來量 GPU 時間）
# 用法：
#   .\tools\make_testcases.ps1            # 工具已存在就不重編
#   .\tools\make_testcases.ps1 -Rebuild   # 強制重編工具
#   .\tools\make_testcases.ps1 -SkipPerf  # 不產生大型效能測資
param(
    [switch]$Rebuild,
    [switch]$SkipPerf
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot
Set-Location $root

foreach ($tool in "gen", "ref", "cmp") {
    if ($Rebuild -or -not (Test-Path "build/$tool.exe")) {
        & "$root/build.ps1" -Source "tools/$tool.cpp" -Output "build/$tool.exe" -HostFlags /O2, /openmp
        if ($LASTEXITCODE -ne 0) { throw "編譯 $tool.cpp 失敗" }
    }
}

New-Item -ItemType Directory -Force testcases | Out-Null

# 名稱, V, E, seed；V 刻意混合 BF 的倍數與非倍數，以檢查邊界處理
$cases = @(
    @("c01", 1,    0,       1),   # 單一頂點、沒有邊
    @("c02", 7,    20,      2),   # 小到可以手算
    @("c03", 100,  300,     3),   # 稀疏，大量 INF（無路徑）
    @("c04", 999,  50000,   4),   # 不是 32／64 的倍數
    @("c05", 2048, 400000,  5),   # 是 32／64 的倍數
    @("c06", 3001, 1500000, 6)    # 較大，非倍數
)
foreach ($c in $cases) {
    $name, $V, $E, $seed = $c
    & build/gen.exe $V $E "testcases/$name.in" $seed
    if ($LASTEXITCODE -ne 0) { throw "gen 失敗：$name" }
    & build/ref.exe "testcases/$name.in" "testcases/$name.out"
    if ($LASTEXITCODE -ne 0) { throw "ref 失敗：$name" }
}

if (-not $SkipPerf) {
    $perf = @(
        @("p01", 10000, 5000000,  11),
        @("p02", 20000, 10000000, 12)   # 規格中單 GPU 版的 V 上限；輸出檔約 1.6 GB
    )
    foreach ($c in $perf) {
        $name, $V, $E, $seed = $c
        & build/gen.exe $V $E "testcases/$name.in" $seed
        if ($LASTEXITCODE -ne 0) { throw "gen 失敗：$name" }
    }
}
