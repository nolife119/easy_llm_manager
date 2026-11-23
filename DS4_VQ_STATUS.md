# Inkling 2 Small VQ / DS4 — experimental branch

The existing llama.cpp application and **main** are untouched.

## Requested end state

One click on a Windows H100 NVL x2 server without active NVLink:
- image + text inputs
- 1,048,576-token maximum context
- DSpark draft acceleration when feasible
- >=30 **actual decode** tokens/s, not end-to-end HTTP tokens/s.

These are **acceptance criteria**, not achieved results.

## Implemented

- `ds4_vq.py`: validates the local DS4 VQ release directory (presence check only), builds a DS4 command, checks OpenAI model discovery and provides a minimal end-to-end benchmark.
- `ds4_oneclick.py`: requests 1M context and DSpark 3, runs `--check-config`, and fails closed until two-GPU placement has been independently validated.
- `start_ds4_vq.bat`: Windows double-click entry point.
- `tests/test_ds4_vq.py`: offline adapter tests.

To try the launcher **after** a compatible ds4-server exists, copy `ds4_oneclick.example.json` to `ds4_oneclick.local.json`, edit paths, then double-click `start_ds4_vq.bat`. The acknowledgment flag is intentionally false. It is NOT a multi-GPU enable switch.

## Not yet supported / not verified

1. DS4 VQ inference kernels on Windows H100 (sm_90).
2. Sharding the 106 GiB device-resident trunk across the two non-NVLink H100 cards. Merely showing two CUDA devices is insufficient.
3. GPU placement controls for this Inkling-2 VQ family, PCIe communication costs and whether tensor/layer parallelism works at all.
4. 1M context with **all active** KV on GPU, GPU+SSD KV strategy, and long-context decode at 30 tok/s.
5. DSpark on both GPUs and concurrent vision preprocessing.
6. Production manager UI integration, authenticated access, verified latency breakdown and 30 tok/s benchmark gate.

The upstream DS4 documentation describes a single DGX Spark and calls for device-specific qualification on other NVIDIA devices. A `--check-config` pass checks configuration, not actual GPU multi-device execution. DS4 needs genuine model-parallel support or a port for two H100s before the requested one-click system is feasible.

## Suggested acceptance benchmark

Record separately: 8K, 32K, 128K, 512K, and 1M **filled** context. Report prompt prefill, TTFT, decode-only rate, actual GPU0/GPU1 residency, disk I/O, DSpark accepted/proposed tokens, and image+text input success. Run with drafting off and on on identical inputs; report sustained medians. Do not claim 30 tok/s from a 32K-only run.
