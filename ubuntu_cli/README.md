# Easy LLM Manager — Ubuntu offline CLI

Python 3.12, Linux x86_64, A100 SM80. No Studio, Node/npm, venv, sudo, GUI, vLLM or external runtime download. Only model downloads need approved Hugging Face proxy and huggingface_hub.

## Quick start
Upload this directory and approved Linux CUDA portable tar.gz (do not unpack on Windows). Put archive next to llm.py or in archives/. If multiple archives exist, specify one.
```sh
cd ubuntu_cli
python llm.py setup /path/to/archive.tar.gz
python llm.py download --all
python llm.py smoke
python llm.py load 512k
python llm.py run
```
If huggingface_hub is not installed, install from approved internal PyPI: python -m pip install huggingface_hub. Existing HF_ENDPOINT/proxy environment is respected. No changes to shell startup files. Optional: . ./env.sh

Recommended starting engine: [Unsloth release b12649-mix-c7059d6](https://github.com/unslothai/llama.cpp/releases/tag/b12649-mix-c7059d6), Linux x64 CUDA13 portable. Check driver compatibility and SM80 support; successful --list-devices is not proof of model kernel compatibility. If smoke fails, inspect logs and try an approved SM80-compatible build. GGUF quantization does not make all FP4/FP8 kernels compatible. No GPU test was performed during authoring.

## Commands
python llm.py without args opens a terminal menu.
- setup [archive]: safe tar extraction, locate executable, version/GPU check, write env.sh and config.
- doctor: version, devices, nvidia-smi
- download [--all]: download selected quant; optional projector and DSpark; --all downloads all three. HF client resumes incomplete transfers. Bounded retries; verify remote size, GGUF magic, SHA256 if LFS metadata provides it. Re-run after interruption. For hash mismatch, move corrupt file aside before retrying. Metadata access is required for first verification.
- status [--all]: local manifest/file verification; no network
- smoke: 8K FP16 text-only generation, stops automatically; logs preserved.
- run: foreground server with health polling, Ctrl+C stops; log file is printed. 
- bench [tokens]: from another terminal while server is running, synthetic raw completion and timings saved. Does not test full context or tool/vision compatibility.
- save NAME / load NAME / presets / restore / history / show
- init: generate config.json on first run.
Edit config.json directly. Built-ins safe, 128k, 512k, 512k-dspark are loaded by command. Start at safe then 128k then 512k; DSpark last. Presets do not change selected quant or engine. Long context and 60 t/s are targets, not guarantees.

Set layout to mainline if using the Mainline_Support directory in the HF repository (verify exact filenames with proxy). Set vision or dspark independently. extra_args is a JSON string array, e.g. ["--reasoning-budget","2048"] only if supported by installed engine. Keep host loopback and use approved forwarding. API base http://127.0.0.1:8888/v1. Claude Code needs compatible Anthropic endpoint or an approved adapter; OpenAI endpoint alone is not enough.

Each run saves config, command, engine probes, server.log and result under state/runs. Last-ready is health-only, not proof of quality. Bench timings use server timings, not SSE chunk counts. Do not run multiple managers against the same port/state concurrently. Model binaries and runtime are not included in ZIP.

## Recovery
Check server.log for missing shared libraries, unsupported SM80 kernels, OOM or unsupported GGUF architecture. For OOM reduce context/ubatch. If q8 KV fails, use safe preset. If DSpark fails disable it. Do not override security policy or disable TLS checks.
