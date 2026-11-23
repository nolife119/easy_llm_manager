"""Inkling 2 Mini / Unsloth DSpark: verified GGUFs and Windows llama-server launcher.

Designed for one selected CUDA GPU, no WSL/Docker, stdlib control plane.
The supplied pre-extracted Unsloth CUDA 13 newer binary is reused as-is.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = "unsloth/Inkling-2-Mini-GGUF"
MODEL = "Inkling-2-Mini-UD-Q8_K_XL.gguf"
DRAFT = "dspark-Inkling-2-Mini-Q8_0.gguf"
PROJECTOR = "mmproj-F16.gguf"
FILES = (MODEL, DRAFT, PROJECTOR)
EXPECTED_GB = (22.87, 0.402, 0.639)
CONTEXTS = (8192, 32768, 65536, 131072, 262144, 524288, 1048576)
DEFAULTS = {
    "engine": "", "model_dir": "", "context": 32768, "parallel": 1,
    "batch": 2048, "ubatch": 512, "kv": "f16", "spec_tokens": 3,
    "dspark": True, "vision": True, "flash": True, "metrics": True,
    "port": 8080, "gpu_index": 0, "gpu_layers": 99
}


def root():
    return Path(os.environ.get("INKLING_MINI_HOME") or
                Path(os.environ.get("LOCALAPPDATA", Path.home())) / "InklingMiniManager")


def model_dir(cfg):
    return Path(cfg["model_dir"]).expanduser() if cfg["model_dir"] else root() / "models"


def config_path():
    return root() / "mini-settings.json"


def validate(data):
    if not isinstance(data, dict):
        raise ValueError("설정은 JSON 객체여야 합니다")
    unknown = set(data) - set(DEFAULTS)
    if unknown:
        raise ValueError("알 수 없는 설정: " + ", ".join(sorted(unknown)))
    cfg = dict(DEFAULTS)
    cfg.update(data)
    for key in ("engine", "model_dir"):
        if not isinstance(cfg[key], str) or len(cfg[key]) > 2048:
            raise ValueError(key + " 경로가 잘못되었습니다")
    if cfg["context"] not in CONTEXTS or type(cfg["context"]) is not int:
        raise ValueError("컨텍스트는 지원 목록 중 선택하세요")
    for key, low, high in (
        ("parallel", 1, 2), ("batch", 128, 8192), ("ubatch", 32, 2048),
        ("spec_tokens", 1, 7), ("port", 1024, 65535),
        ("gpu_index", 0, 15), ("gpu_layers", 1, 999),
    ):
        v = cfg[key]
        if type(v) is not int or not low <= v <= high:
            raise ValueError(f"{key}: {low}~{high} 정수가 필요합니다")
    if cfg["ubatch"] > cfg["batch"]:
        raise ValueError("ubatch는 batch 이하")
    if cfg["port"] == 8766:
        raise ValueError("8766은 매니저 포트입니다")
    if cfg["kv"] not in ("f16", "q8_0"):
        raise ValueError("KV 캐시는 f16 또는 q8_0")
    for key in ("dspark", "vision", "flash", "metrics"):
        if type(cfg[key]) is not bool:
            raise ValueError(key + "은 true/false")
    return cfg


def load_config():
    try:
        cfg = json.loads(config_path().read_text(encoding="utf-8"))
        return validate(cfg)
    except FileNotFoundError:
        return dict(DEFAULTS)


def save_config(cfg):
    cfg = validate(cfg)
    root().mkdir(parents=True, exist_ok=True)
    tmp = config_path().with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, config_path())
    return cfg


def find_engine():
    """Small bounded search, never scan a whole drive."""
    bases = (Path(__file__).resolve().parent, Path.cwd(), root())
    for base in bases:
        for name in ("llama-server.exe", "llama.cpp/llama-server.exe",
                     "bin/llama-server.exe", "build/bin/Release/llama-server.exe"):
            p = base / name
            if p.is_file():
                return str(p.resolve())
        try:
            for child in base.iterdir():
                if child.is_dir() and (child.name.lower().startswith(("app-", "llama", "unsloth"))):
                    p = child / "llama-server.exe"
                    if p.is_file():
                        return str(p.resolve())
        except (OSError, PermissionError):
            continue
    return ""


def gpu_info():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"ok": False, "error": "nvidia-smi를 찾을 수 없습니다"}
    try:
        cmd = [exe, "--query-gpu=index,name,memory.total,memory.free,driver_version",
               "--format=csv,noheader,nounits"]
        p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=8)
        if p.returncode:
            raise RuntimeError(p.stderr.strip() or "nvidia-smi 오류")
        rows = [r.strip().split(", ") for r in p.stdout.splitlines() if r.strip()]
        devices = [
            {"index": int(r[0]), "name": r[1], "total_mb": float(r[2]),
             "free_mb": float(r[3]), "driver": r[4]} for r in rows if len(r) >= 5
        ]
        return {"ok": bool(devices), "devices": devices}
    except (ValueError, OSError, subprocess.SubprocessError, RuntimeError) as exc:
        return {"ok": False, "error": str(exc)}


def probe_engine(path):
    p = Path(path)
    if not p.is_file() or p.name.lower() != "llama-server.exe":
        raise ValueError("압축을 푼 폴더의 llama-server.exe 전체 경로를 입력하세요")
    try:
        proc = subprocess.run([str(p), "--help"], capture_output=True,
                              text=True, errors="replace", timeout=30,
                              cwd=str(p.parent))
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("엔진을 실행할 수 없습니다: " + str(exc)) from exc
    help_text = (proc.stdout + "\n" + proc.stderr).lower()
    required = ("--model-draft", "--spec-type", "--spec-draft-n-max",
                "--n-gpu-layers-draft", "--mmproj", "--jinja")
    missing = [x for x in required if x not in help_text]
    if missing:
        raise ValueError("이 llama-server는 필요한 Inkling DSpark 옵션이 없습니다: " +
                         ", ".join(missing) + ". Unsloth CUDA13 newer 빌드를 지정하세요.")
    return {"path": str(p.resolve()), "dspark": True,
            "vision": True, "metrics": "--metrics" in help_text,
            "flash": "--flash-attn" in help_text,
            "kv": "--cache-type-k" in help_text and "--cache-type-v" in help_text,
            "anthropic": "messages" in help_text or "/v1/messages" in help_text,
            "help_excerpt": help_text[:350]}


def build_command(cfg, capabilities=None):
    cfg = validate(cfg)
    ex = Path(cfg["engine"])
    if not ex.is_file():
        raise ValueError("기존 llama-server.exe 경로를 지정하세요")
    folder = model_dir(cfg)
    for name in (MODEL, *([DRAFT] if cfg["dspark"] else []),
                 *([PROJECTOR] if cfg["vision"] else [])):
        p = folder / name
        if not p.is_file() or p.stat().st_size < 100_000:
            raise ValueError("모델 파일 없음: " + str(p))
    caps = capabilities if capabilities is not None else probe_engine(str(ex))
    cmd = [str(ex.resolve()), "--model", str((folder / MODEL).resolve()),
           "--host", "127.0.0.1", "--port", str(cfg["port"]),
           "--ctx-size", str(cfg["context"]), "--parallel", str(cfg["parallel"]),
           "--n-gpu-layers", str(cfg["gpu_layers"]),
           "--batch-size", str(cfg["batch"]), "--ubatch-size", str(cfg["ubatch"]),
           "--jinja", "--temp", "1", "--top-p", "1", "--top-k", "0",
           "--min-p", "0"]
    if cfg["vision"]:
        cmd += ["--mmproj", str((folder / PROJECTOR).resolve())]
    if cfg["dspark"]:
        cmd += ["--model-draft", str((folder / DRAFT).resolve()),
                "--spec-type", "draft-dspark", "--spec-draft-n-max",
                str(cfg["spec_tokens"]), "--n-gpu-layers-draft", "99"]
    if cfg["flash"]:
        if not caps.get("flash"):
            raise ValueError("이 엔진은 --flash-attn을 지원하지 않습니다")
        cmd += ["--flash-attn", "on"]
    if cfg["kv"] != "f16":
        if not caps.get("kv"):
            raise ValueError("엔진이 KV 캐시 설정을 지원하지 않습니다")
        cmd += ["--cache-type-k", cfg["kv"], "--cache-type-v", cfg["kv"]]
    if cfg["metrics"] and caps.get("metrics"):
        cmd.append("--metrics")
    return cmd


def ensure_hub(log):
    try:
        import huggingface_hub
        return huggingface_hub
    except ImportError:
        log("huggingface_hub 및 hf_xet 설치 중 (최초 1회)")
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                        "huggingface_hub[hf_xet]>=0.32,<2"], check=True, timeout=300)
        import huggingface_hub
        return huggingface_hub


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_models(cfg, log, progress):
    """Resume via HF hub/Xet; independent SHA-256 of all three LFS/Xet artifacts."""
    hub = ensure_hub(log)
    from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
    from huggingface_hub.file_download import get_hf_file_metadata
    info = HfApi().model_info(REPO, repo_type="model")
    revision = info.sha
    folder = model_dir(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    remaining_gb = sum(gb for fn, gb in zip(FILES, EXPECTED_GB)
                       if not (folder / fn).is_file())
    free_gb = shutil.disk_usage(folder).free / (1024 ** 3)
    if free_gb < remaining_gb + 3:
        raise RuntimeError(f"디스크 공간 부족: 남은 {free_gb:.1f}GiB / 필요한 약 {remaining_gb + 3:.1f}GiB")
    manifest_path = folder / ".inkling-mini-verified.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {}
    for i, filename in enumerate(FILES, 1):
        progress(i - 1, len(FILES), filename, "metadata")
        metadata = get_hf_file_metadata(hf_hub_url(repo_id=REPO, filename=filename, revision=revision))
        expected_hash = (metadata.etag or "").strip('"').lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise RuntimeError(filename + "의 SHA-256 메타데이터를 확인할 수 없어 다운로드를 중단했습니다")
        expected_size = metadata.size
        destination = folder / filename
        record = manifest.get(filename, {})
        same = destination.is_file() and (
            record.get("sha256") == expected_hash and
            record.get("size") == destination.stat().st_size and
            record.get("mtime_ns") == destination.stat().st_mtime_ns
        )
        if same and (expected_size is None or destination.stat().st_size == expected_size):
            log(f"[{i}/3] 이미 검증된 {filename} 재사용")
            progress(i, len(FILES), filename, "verified")
            continue
        if not destination.exists():
            log(f"[{i}/3] 다운로드: {filename} (중단 후 재실행 시 이어받기)")
        else:
            log(f"[{i}/3] 기존 파일 검증 및 필요시 복구: {filename}")
        progress(i - 1, len(FILES), filename, "downloading")
        actual = Path(hf_hub_download(repo_id=REPO, filename=filename,
                                      revision=revision, local_dir=str(folder)))
        progress(i - 1, len(FILES), filename, "sha256")
        if expected_size is not None and actual.stat().st_size != expected_size:
            raise RuntimeError(filename + " 크기가 원본과 일치하지 않습니다")
        result = sha256(actual)
        if result != expected_hash:
            actual.unlink(missing_ok=True)
            raise RuntimeError(filename + " SHA-256 불일치. 파일을 제거했습니다.")
        with actual.open("rb") as header:
            if header.read(4) != b"GGUF":
                raise RuntimeError(filename + " 파일이 GGUF가 아닙니다")
        stat = actual.stat()
        manifest[filename] = {"sha256": result, "size": stat.st_size,
                              "mtime_ns": stat.st_mtime_ns, "revision": revision}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        log(f"[{i}/3] SHA-256 확인 완료: {filename}")
        progress(i, len(FILES), filename, "verified")
    return {"revision": revision, "directory": str(folder), "verified_files": list(FILES)}


def is_verified(cfg):
    folder = model_dir(cfg)
    try:
        marker = json.loads((folder / ".inkling-mini-verified.json").read_text(encoding="utf-8"))
        for fn in FILES:
            stat = (folder / fn).stat()
            r = marker[fn]
            if r["size"] != stat.st_size or r["mtime_ns"] != stat.st_mtime_ns or not re.fullmatch(
                    r"[a-f0-9]{64}", r["sha256"]):
                return False
        return True
    except (OSError, KeyError, TypeError, ValueError):
        return False


def parse_metrics(text):
    values = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_:][\w:]*)\s+([-+0-9.eE]+)\s*$", line)
        if m:
            try:
                values[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return values


def rate(values, token_key, time_key, prior=None):
    dtokens = values.get(token_key, 0)
    dseconds = values.get(time_key, 0)
    if prior:
        dtokens -= prior.get(token_key, 0)
        dseconds -= prior.get(time_key, 0)
    return round(dtokens / dseconds, 1) if dtokens > 0 and dseconds > 0 else None
