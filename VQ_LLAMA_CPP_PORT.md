# Inkling 2 Small VQ → native llama.cpp port (H100 NVL 2 × 94 GB, no NVLink)

## Scope / acceptance (hard gates)
- **Only** Baekpica/Inkling-2-Small-VQ-GGUF, 52-part GVQ1 hVQ release.
- Windows Server x64 CUDA 13, sm_90 H100 NVL GPUs, PCIe peer path, no NVLink.
- Full text inference, then image+text; 1,048,576 context; DSpark opt-in.
- Reproducible **decode-only** throughput >=30 tokens/s at stated prompt frontier. Report separate 32K, 128K, 512K, filled 1M metrics. Never imply 30 tok/s across all frontiers from one measurement.
- Must preserve VQ compression on GPU; converting full expert weights to BF16 or conventional quantization violates the VRAM/speed goal.

## Exact upstream starting points
- llama.cpp newer tree contains `src/models/inkling2.cpp` for ordinary Inkling2.
- Baekpica/ds4-dfm-rs has `tools/vq_calib/inkling2_vq_format.py` (strict GGUF GVQ1 reader), `crates/ds4-core/src/inkling2_vq/` (release contract), native CUDA VQ expert execution, vision path and DSpark.
- Model's `release.json` identifies 48 expert GGUF parts, plus nonexpert, embedding, drafter and vision parts. Do **not** treat as llama.cpp split GGUF shards.
- DS4's 29.79 tok/s mean decode is on **one DGX Spark**, not Windows H100 x2; port performance is unknown.

## Core llama.cpp changes required (in order)
1. Add GVQ1 v1 reader with strict part layout, tensor type, codebook and SHA validation; standalone parser golden tests against Baekpica reader.
2. Add `GGML_TYPE_GVQ1...`/a private backend type or an alternate expert tensor storage interface (avoid conflicting ggml type IDs); connect to model loader. Existing gguf loader assumes conventional splits and cannot parse this 52-part release.
3. Port CUDA hVQ 12-bit expert GEMV and prefill GEMM, codebook lookup and row transforms for H100 sm_90. Tests against DS4 golden/reference tensors. GPU compute must operate on compact codes, not expand entire 106 GiB trunk.
4. Implement per-layer expert placement across two PCIe GPUs (and explicit router/activation transfers). No NVLink, no unified VRAM assumption. Audit whether expert partitioning or layer placement minimizes PCIe transfers; detect CPU spills.
5. Reconnect Inkling 2 forward pass: 48 layers, 384 experts/top-8, 8 full attention / 40 sliding, 4-tap convolutions, residual taps, output protocol.
6. Integrate multi-part BF16 embeddings, Q4_K dense body and output head, official tokenizer/template; handle image tower separately.
7. Implement NVFP4 KV page layout, resident-cache strategy and spill-to-SSD after resident threshold (native 1M does not imply 30 tok/s at 1M).
8. Integrate DSpark drafter as opt-in *only after* plain decode passes quality tests.
9. Enable Windows CUDA13 CMake builds, signed/approved enterprise distribution and `llama-server` OpenAI API.
10. Benchmark every frontier for prefill, TTFT, decode, GPU VRAM and PCIe throughput; compare with reproducible input and outputs.

## Blockers
- Current upstream llama.cpp has no Baekpica GVQ1/Inkling2 VQ reader or H100 kernels validated for this release.
- Running the existing Unsloth b12347 binary with `-m release.json` or the first expert part will **not** load the model.
- This branch is a **port workstream**, not a runnable VQ inference implementation. No claimed 30 tok/s.

## Engineering decision
Target a separate VQ-enabled llama.cpp source checkout and preserve the proven existing binary untouched. Only integrate into easy_llm_manager after plain decode runs on both GPUs. Avoid DS4-as-a-service substitution: the requested runtime is llama.cpp, not DS4.
