# DS4 experimental installation — Windows Server / H100 NVL x2

This is a **development bootstrap**, not yet a working one-click VQ inference release.

## Install DS4 (WSL2 Ubuntu)

Download the feature branch ZIP and extract it to a separate directory.
In PowerShell (no administrator permissions needed *if WSL2/Ubuntu are already provisioned*):

```powershell
cd D:\AI\easy_llm_manager_vq
powershell -ExecutionPolicy Bypass -File .\install_ds4_wsl.ps1
```

The script checks that Ubuntu under WSL can see NVIDIA GPUs, then obtains the upstream DS4 repository and builds using `make cuda CUDA_ARCH=sm_90`. It may ask for Ubuntu sudo credentials to install ordinary build tools. It installs Rust via official rustup when missing. It does not install WSL, a GPU driver or the Linux CUDA toolkit; those are system-level prerequisites. Existing llama.cpp software is untouched. It also requires external network access.

Expected WSL engine path: `~/AI/ds4-dfm-rs/ds4-server`.

### Important limitations

- DS4's reference hardware is one DGX Spark. Neither 2 x H100 NVL with inactive NVLink nor Windows-hosted WSL2 inference is currently qualified.
- Two GPUs recognized by `nvidia-smi` do not mean that DS4 partitions VQ weights across them.
- The 106 GiB resident trunk cannot fit on a single 94 GB GPU, so full-GPU 2-device partitioning is an absolute prerequisite.
- The experimental Python `ds4_oneclick.py` still expects a directly executable DS4 path and is **not** wired to WSL yet.
- No VQ model auto-download, vision smoke test, real 1M session, MTP quality/throughput test, or 30 tok/s guarantee.
- Do not switch the production manager to this branch as a replacement. Run in a separate test directory.

## Required acceptance gates

1. DS4 compiles for sm_90, launches, and supports the VQ model's custom CUDA kernels.
2. All model trunk tensors load across **both** H100 cards without spilling to CPU and without NVLink.
3. Native 1,048,576 context and NVFP4 KV / SSD tier operate correctly.
4. Image + text chat and Anthropic/OpenAI API tool calling pass real prompts.
5. DSpark increases actual decode performance on representative workloads.
6. Sustained >=30 decode tok/s is recorded, separately at 32K, 128K, 512K, and filled 1M context.

Without 1–2, an installation success is **not** an inference success.
