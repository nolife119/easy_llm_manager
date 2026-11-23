# Windows Inkling Experiment Lab — experimental

## Purpose
A Windows-native local dashboard for *configurable experiments*, not a replacement for actual inference backends. It provides no new VQ CUDA kernels and no automatic cross-GPU weight sharding. DS4 VQ is currently unverified on this Windows H100 x2 host. Do not delete or modify the existing working llama.cpp setup.

## Quick start
1. Download the branch ZIP, extract to a new directory.
2. Double-click `start_experiment_lab.bat` (Python 3.10+).
3. Open `http://127.0.0.1:8766`.
4. Edit profile JSON: replace `executable` with a **real Windows .exe path** and `model` with a real GGUF first shard or VQ model folder.
5. Save. Select a profile, start, inspect GPU/status, run benchmark. Stop before switching.
6. Click *진단 ZIP 생성* to get an archive path for logs and benchmark results. Review logs for confidential contents before sharing.

All profile data is saved outside the repository in `%LOCALAPPDATA%\InklingManager\experiment-platform\platform.json`. Running a new ZIP retains prior edits. The dashboard is bound to 127.0.0.1 only. Use on a trusted machine; do not port-forward or expose it to a network.

## Profile matrix
- `llama-32k-baseline` and `llama-128k-baseline`: known existing llama.cpp path; Windows.
- `ds4-vq-32k-preflight`: DS4 command template, not a working H100 execution guarantee.
- `ds4-vq-1m-dspark3`: requested 1M + DSpark 3 configuration, not a verified setup.

The interface edits JSON directly, so it is possible to change additional modes, GPU settings, KV types and draft depths once the relevant backend is available. Do not include passwords or secrets in profiles.

## Critical limitations
- Only controls **already installed executables**. It does not compile DS4, automatically download a 118GB model, or implement Windows DS4 native build.
- DS4 `--check-config` may validate the flags/memory estimate, but does not prove actual dual-GPU execution.
- The UI benchmark is elapsed wall time including prompt prefill and HTTP overhead; **never report it as decode-only speed**.
- 1M context *configured* is not 1M tokens *filled*; long-context and image acceptance tests remain to be built.
- GPU memory outputs are snapshots, not precise continuous utilization.
- `nvidia-smi` device enumeration is not proof of multi-GPU sharding.
- Logs can contain private prompts. Review before sending diagnostic archives.
- If your OS or corporate controls forbid untrusted scripts or executables, do not bypass those controls.

## Support cycle
Send the diagnostic ZIP (after review), OS version, CUDA driver version, and the profile name. Normally new combinations can be tried by editing `platform.json` without replacing the platform code. Missing CUDA features / an unsupported kernel still require upstream engine code changes.

## Acceptance gates before claiming success
1. Native DS4-compatible Windows executable built and verified on H100 sm_90.
2. Model weights split across both PCIe H100 GPUs; no major CPU expert offload.
3. 32K/128K/512K/filled 1M context prompt and decode tests.
4. Vision image+text and tool calls.
5. DSpark throughput versus plain mode; sustained decode-only >=30 tok/s.
