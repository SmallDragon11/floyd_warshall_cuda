# 編譯 CUDA 程式（預設 src/apsp.cu → build/apsp.exe）
# 用法：
#   .\build.ps1                         # 編譯 src/apsp.cu
#   .\build.ps1 -Define BF=64           # 等同在程式中 #define BF 64
#   .\build.ps1 -Source src/foo.cu -Output build/foo.exe
#   .\build.ps1 -Source tools/ref.cpp -Output build/ref.exe -HostFlags /O2,/openmp
param(
    [string]$Source = "src/apsp.cu",
    [string]$Output = "build/apsp.exe",
    [string[]]$Define = @(),
    [string[]]$HostFlags = @()   # 每項轉成 -Xcompiler <flag> 傳給 cl.exe
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# 載入 MSVC 環境：nvcc 在 Windows 需要 cl.exe 當 host compiler
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $installer = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer"
    $env:PATH = "$installer;$env:PATH"   # 讓 Launch-VsDevShell.ps1 內部找得到 vswhere.exe
    $vs = & "$installer\vswhere.exe" -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vs) { throw "找不到含 MSVC 的 Visual Studio，請安裝「使用 C++ 的桌面開發」工作負載" }
    & "$vs\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation | Out-Null
}

New-Item -ItemType Directory -Force (Split-Path $Output) | Out-Null

$nvccArgs = @(
    "-arch=sm_75",            # GTX 1650 的 compute capability 7.5
    "-lineinfo",              # 產生 device 端行號資訊，讓 Nsight Compute 對應到原始碼
    "-res-usage",             # 編譯時印出每個 kernel 的 register／shared memory 用量
    "-Xcompiler", "/utf-8",   # cl.exe 以 UTF-8 讀原始碼，避免 C4819 與中文註解亂碼
    "-o", $Output
) + ($Define | ForEach-Object { "-D$_" }) + ($HostFlags | ForEach-Object { "-Xcompiler", $_ }) + @($Source)

Write-Host "nvcc $($nvccArgs -join ' ')"
& nvcc @nvccArgs
exit $LASTEXITCODE
