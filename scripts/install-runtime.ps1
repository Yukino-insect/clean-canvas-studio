[CmdletBinding()]
param(
    [ValidateSet("cpu", "cu126")]
    [string]$TorchBackend = "cpu",

    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$requirementsRoot = Join-Path $projectRoot "requirements"
$baseLock = Join-Path $requirementsRoot "base.lock"
$torchLock = Join-Path $requirementsRoot "torch-$TorchBackend.lock"

if (-not (Test-Path -LiteralPath $baseLock)) {
    throw "Missing dependency lock: $baseLock"
}
if (-not (Test-Path -LiteralPath $torchLock)) {
    throw "Missing PyTorch profile lock: $torchLock"
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
