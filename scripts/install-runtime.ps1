[CmdletBinding()]
param(
    [ValidateSet("cpu", "cu126")]
    [string]$TorchBackend = "cpu",

    [ValidateSet("cpu", "cu118")]
    [string]$PaddleBackend = "cpu",

    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$requirementsRoot = Join-Path $projectRoot "requirements"
$baseLock = Join-Path $requirementsRoot "base.lock"
$torchLock = Join-Path $requirementsRoot "torch-$TorchBackend.lock"
$paddleLock = Join-Path $requirementsRoot "paddle-$PaddleBackend.lock"
$paddleCpuLock = Join-Path $requirementsRoot "paddle-cpu.lock"

if (-not (Test-Path -LiteralPath $baseLock)) {
    throw "Missing dependency lock: $baseLock"
}
if (-not (Test-Path -LiteralPath $torchLock)) {
    throw "Missing PyTorch profile lock: $torchLock"
}
if (-not (Test-Path -LiteralPath $paddleLock)) {
    throw "Missing PaddlePaddle profile lock: $paddleLock"
}
if (-not (Test-Path -LiteralPath $paddleCpuLock)) {
    throw "Missing PaddlePaddle CPU fallback lock: $paddleCpuLock"
}

if ($TorchBackend -eq "cpu") {
    $torchIndex = "https://download.pytorch.org/whl/cpu"
} else {
    $torchIndex = "https://download.pytorch.org/whl/cu126"
}

Write-Host "Installing locked base dependencies..."
& $Python -m pip install --require-hashes -r $baseLock
if ($LASTEXITCODE -ne 0) {
    throw "Base dependency installation failed."
}

Write-Host "Selecting PaddlePaddle profile: $PaddleBackend..."
# Paddle 官方明确要求 CPU/GPU 发行物不要共存。先卸载旧 profile，避免 pip
# 留下同名模块而让 OCR 误以为自己具备 CUDA 能力。
& $Python -m pip uninstall -y paddlepaddle paddlepaddle-gpu
if ($LASTEXITCODE -ne 0) {
    throw "Unable to remove the previous PaddlePaddle profile."
}
$paddleInstallArgs = @("install", "--require-hashes", "-r", $paddleLock)
if ($PaddleBackend -eq "cu118") {
    # GPU wheel 发布在 Paddle 官方索引，不在 PyPI 主索引中。
    $paddleInstallArgs += @("--extra-index-url", "https://www.paddlepaddle.org.cn/packages/stable/cu118/")
}
& $Python -m pip @paddleInstallArgs
if ($LASTEXITCODE -ne 0) {
    if ($PaddleBackend -ne "cu118") {
        throw "PaddlePaddle profile installation failed."
    }
    Write-Warning "Paddle GPU profile installation failed; installing the locked CPU profile instead."
    & $Python -m pip install --require-hashes -r $paddleCpuLock
    if ($LASTEXITCODE -ne 0) {
        throw "Paddle GPU profile installation failed, and CPU fallback installation also failed."
    }
    $PaddleBackend = "cpu"
}
& $Python -c "import paddle; print(f'paddle={paddle.__version__}'); print(f'compiled_with_cuda={paddle.is_compiled_with_cuda()}'); print(f'cuda_available={paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0}')"
if ($LASTEXITCODE -ne 0) {
    throw "PaddlePaddle verification failed after installation."
}

Write-Host "Installing locked PyTorch profile: $TorchBackend..."
# PyTorch wheels reside on the official PyTorch index while Torch's small Python
# dependencies are resolved from PyPI. The exact local-version requirement in
# the lock (for example ``+cu126``) prevents pip from selecting a CPU wheel.
& $Python -m pip install --require-hashes --extra-index-url $torchIndex -r $torchLock
if ($LASTEXITCODE -ne 0) {
    throw "PyTorch profile installation failed."
}

& $Python -c "import torch; print(f'torch={torch.__version__}'); print(f'cuda_build={torch.version.cuda}'); print(f'cuda_available={torch.cuda.is_available()}')"
if ($LASTEXITCODE -ne 0) {
    throw "PyTorch verification failed after installation."
}

if ($TorchBackend -eq "cu126") {
    Write-Host "CUDA wheel installed. cuda_available must be True before selecting CUDA in the application."
}
if ($PaddleBackend -eq "cu118") {
    Write-Host "Paddle GPU wheel installed. OCR will try GPU first and automatically fall back to CPU if runtime initialization fails."
}
