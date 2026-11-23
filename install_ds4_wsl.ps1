# DS4 development environment bootstrap for Windows Server + WSL2 Ubuntu.
# This installs/builds upstream DS4 only. It does not certify two-GPU model sharding.
param(
  [string]$Distribution = "Ubuntu",
  [string]$SourceRef = "main",
  [switch]$SkipPackages
)
$ErrorActionPreference = "Stop"
function Check([string]$Title, [string[]]$Args) {
  Write-Host "== $Title =="
  & $Args[0] $Args[1..($Args.Length-1)]
  if ($LASTEXITCODE -ne 0) { throw "$Title failed ($LASTEXITCODE)" }
}
if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
  throw "WSL2 is not available. Windows Server may require administrator approval to enable WSL and virtualization."
}
& wsl.exe --list --quiet
if ($LASTEXITCODE -ne 0) { throw "Cannot query installed WSL distributions" }
Write-Host "Checking GPU access inside WSL..."
& wsl.exe -d $Distribution -- nvidia-smi -L
if ($LASTEXITCODE -ne 0) { throw "NVIDIA GPUs not visible inside WSL. Check host driver and WSL GPU support." }
$script = @'
set -euo pipefail
echo "[DS4] checking Ubuntu toolchain"
if ! command -v git >/dev/null || ! command -v make >/dev/null || ! command -v cmake >/dev/null || ! command -v cargo >/dev/null || ! command -v nvcc >/dev/null; then
  if [ "$SKIP_PKGS" = "1" ]; then echo "Missing tools; install git make cmake cargo CUDA toolkit"; exit 2; fi
  echo "Installing Ubuntu dependencies (sudo may prompt for password)"
  sudo apt-get update
  sudo apt-get install -y git make cmake build-essential pkg-config libssl-dev clang curl python3
fi
if ! command -v nvcc >/dev/null; then
  echo "CUDA nvcc missing in WSL. Install a supported Linux CUDA toolkit without replacing Windows driver."; exit 2
fi
if ! command -v cargo >/dev/null; then
  echo "Installing official Rust toolchain bootstrap"
  curl --proto "=https" --tlsv1.2 -fsSL https://sh.rustup.rs -o /tmp/ds4-rustup-init.sh
  sh /tmp/ds4-rustup-init.sh -y --profile minimal
  . "$HOME/.cargo/env"
fi
if [ -f "$HOME/.cargo/env" ]; then . "$HOME/.cargo/env"; fi
mkdir -p "$HOME/AI"
cd "$HOME/AI"
if [ ! -d ds4-dfm-rs/.git ]; then git clone https://github.com/Baekpica/ds4-dfm-rs.git; fi
cd ds4-dfm-rs
if [ -n "$(git status --porcelain)" ]; then echo "Working tree modified; not updating"; exit 2; fi
git fetch origin
git checkout "$SOURCE_REF"
if [ "$SOURCE_REF" = "main" ]; then git pull --ff-only origin main; fi
echo "[DS4] building CUDA architecture sm_90 for H100"
make cuda CUDA_ARCH=sm_90
test -x ./ds4-server
./ds4-server --help >/dev/null
echo "[DS4] build finished at $PWD/ds4-server"
echo "CAUTION: compilation and --help do not prove 2-GPU sharding, 1M context, DSpark, or 30 tok/s."
'@
$env:DS4_SOURCE_REF = $SourceRef
$env:DS4_SKIP_PKGS = $(if ($SkipPackages) { "1" } else { "0" })
# Pass source via stdin to avoid Windows/WSL path translation.
$script = "export SOURCE_REF='$SourceRef'; export SKIP_PKGS='$($env:DS4_SKIP_PKGS)';" + [Environment]::NewLine + $script
$script | & wsl.exe -d $Distribution -- bash -s
if ($LASTEXITCODE -ne 0) { throw "DS4 WSL installation/build failed (exit $LASTEXITCODE)" }
Write-Host "DS4 source/build installed inside WSL. H100 two-device inference remains unvalidated."
