$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    New-Item -ItemType Directory -Force .tools | Out-Null
    $uvPath = Join-Path $repoRoot '.tools/uv/uv.exe'
    if (-not (Test-Path -LiteralPath $uvPath)) {
        Invoke-WebRequest -Uri 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile .tools/uv.zip
        Expand-Archive -LiteralPath .tools/uv.zip -DestinationPath .tools/uv -Force
    }
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $repoRoot '.tools/python'
    $env:UV_CACHE_DIR = Join-Path $repoRoot '.tools/cache'
    & $uvPath python install 3.11 --no-bin --no-registry
    if ($LASTEXITCODE -ne 0) { throw 'Python installation failed' }
    if (-not (Test-Path .venv/Scripts/python.exe)) {
        & $uvPath venv --python 3.11 --seed .venv
        if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
    }
    & $uvPath pip install --python .venv/Scripts/python.exe torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
    & $uvPath pip install --python .venv/Scripts/python.exe -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
    & .venv/Scripts/python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
    if ($LASTEXITCODE -ne 0) { throw 'Environment check failed' }
} finally {
    Pop-Location
}
