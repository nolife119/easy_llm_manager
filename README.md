# Inkling Manager

Windows 기반 llama.cpp 로컬 추론 서버의 다운로드, 컴파일, 실행과 모니터링을 위한 소형 관리 도구.
Python 표준 라이브러리로 관리 웹 UI를 제공하며 모델 다운로드에만 `huggingface_hub`를 사용합니다.

> 초기 개발 버전(v0.1). Python/HTTP API 및 명령 생성의 모의 테스트는 수행했지만 실제 Windows H100에서 컴파일/추론하지는 않았습니다. 회사 승인된 소프트웨어, 모델 및 네트워크 환경에서 사용하세요.

## 빠른 시작

1. 네 파일(`manager.py`, `config.json`, `start.bat`, `README.md`)을 같은 폴더에 복사합니다.
2. Python 3.12 x64 설치 후 `start.bat` 더블 클릭 (또는 PowerShell `py -3.12 manager.py`).
3. 자동으로 열린 브라우저(기본 `http://127.0.0.1:8765/`)에서 환경 진단을 실행합니다. 다른 포트가 선택되면 터미널에 실제 주소를 표시합니다.
4. 다운로드 탭에서 `UD-IQ2_XXS`를 선택하고 원격 파일/실제 용량 조회 후 다운로드합니다. 같은 파일/폴더/리비전에서 다운로드 버튼을 다시 누르면 Hugging Face의 기존 파일/캐시를 재사용합니다. 다운로드 시점의 `main`이 변경될 수 있으므로 완전한 재현성이 필요하면 `config.json`의 모델 `revision`을 커밋 SHA로 고정하세요. 부분 캐시가 항상 재개되는 것은 아니며 HF/Xet 동작에 따릅니다.
5. 컴파일 탭에서 시작합니다. 기존 llama.cpp를 인식하면 재사용합니다. 빌드가 이미 있다면 실행 파일을 직접 지정할 수 있습니다.
6. 서버 실행 탭에서 모델 선택, **첫 구동 (2 GPU)**, 예상 자원 계산, 서버 시작 순으로 진행합니다. 모니터에서 `/health`가 응답하면 로컬 llama.cpp UI로 대화해 보세요.

### 필요한 환경

- [Python 3.12](https://www.python.org/downloads/), [Git](https://git-scm.com/download/win), [CMake](https://cmake.org/download/)
- [NVIDIA Data Center 드라이버](https://www.nvidia.com/Download/index.aspx), [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) — C++ 데스크톱 개발, MSVC x64/x86, Windows SDK 및 CMake 구성 요소. CUDA와 MSVC 버전 호환성 확인.
- [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub/installation) — GUI 환경 탭에서 설치 가능. PyYAML wheel 오류는 Python 3.12 및 `py -3.12 -m pip install --upgrade pip` 확인.
- Visual Studio 2022/2026의 C++ 설치 경로는 `vswhere`로 찾습니다. 빌드 실패 시 **x64 Native Tools Command Prompt / Developer PowerShell**에서 `start.bat`을 다시 실행하세요. 설치된 CMake가 해당 Visual Studio 생성기를 지원해야 합니다.

## 실행 모드의 정확한 의미

| 설정 | 동작 |
|---|---|
| GPU 전체 상주 | `-ngl 99` (Inkling 2 등 99층 이하 모델의 모든 레이어), CPU MoE 오프로드 해제 |
| GPU + CPU MoE | `--n-cpu-moe N`: 처음 N개 MoE 레이어의 Expert를 호스트에 배치; Attention 등은 GPU 대상으로 유지 |
| 동적 Expert 캐시 | `--moe-cache-mib`와 `--moe-cache-cpu-share`: CPU에 있는 Expert를 GPU에 캐싱하고 miss 일부를 CPU에서 연산; llama.cpp 자체 기능이며 별도 Strata 포팅이 아님 |
| 활성 KV GPU | 기본 GPU KV; 빠른 추론 우선 |
| 활성 KV RAM / CPU | `--no-kv-offload`: 활성 KV 및 관련 연산이 CPU로 이동할 수 있음; 1M 생성 속도 특히 주의 |
| RAM prompt cache | `--cache-ram` 및 공유 KV에서 `--cache-idle-slots`: 유휴/재사용 KV 저장. **활성 KV의 투명한 자동 paging과 동일하지 않음** |
| 공유 KV | `--kv-unified`, `--kv-unified-per-slot`, `-c`: pool과 슬롯별 상한 별도 관리. 이 옵션을 지원하지 않으면 빌드 업데이트 또는 공유 모드 끄기 |

동적 Expert 캐시, 공유 KV per-slot, 8/4bit KV 등의 **실제 바이너리 지원 여부는 시작 전 `--help`로 확인**합니다. 아키텍처별 커널 호환성은 도움말만으로 검증되지 않습니다. 최신 소스가 해당 모델을 지원해야 합니다. `--fit off`를 사용해 명시한 배치를 자동으로 변경하지 않도록 합니다.

### 현재 서버 (H100 NVL 2 × 94GB, Xeon Silver 4410Y, 256GB RAM)

- 권장 첫 테스트: IQ2_XXS, CPU MoE 8 layers, GPU KV FP16, 32,768 context, 슬롯 1, batch 512 / ubatch 128, Expert cache 0, CPU threads 12. 안전한 출발점이지 성공/속도 보장은 아닙니다.
- 먼저 이 구성으로 모델 로딩 로그에서 실제 VRAM과 t/s를 기록합니다. CPU MoE를 2~4층씩 줄여 보고 Y/N 경고와 실측으로 확인하세요. 동적 Expert cache를 4096~8192 MiB부터 실험할 때에는 GPU 여유량을 반드시 확인합니다. 위 캐시는 기존 CPU Expert의 추가 GPU 복제본으로 계산됩니다.
- Q3는 약 285.1 GB (265.5 GiB)로 2장에는 전부 들어가지 않습니다. 20~24층 정도 CPU MoE 오프로드가 가능한 후보이지만 모델별 레이어 크기가 달라 반드시 예상치/실측으로 확인하세요.
- 1M은 32K → 128K → 256K → 512K → 1M 순으로 올려 보기. `1M / GPU KV`와 `1M / RAM KV`를 비교합니다. 1M RAM KV 프리셋은 전부 GPU 상주를 선택하므로 2장의 경우 IQ1급 양자화 등으로 바꾸거나 CPU MoE를 선택해야 합니다. 긴 문맥을 채웠을 때의 prefill/TTFT 및 decode 속도를 각각 기록하세요.
- 4장/800GB 서버에서는 Q3 + GPU 활성 KV + 공유 KV + 유휴 RAM cache를 권장 후보로 검토할 수 있습니다. 15인 프리셋의 공유 pool 1.5M은 15명이 모두 1M을 사용하는 것을 보장하지 않습니다. 과부하 시 요청을 제한하거나 대기시켜야 하며, llama.cpp 스케줄러가 버전에 따라 거절할 수도 있습니다. 앱 자체에 별도 요청 스케줄러는 없습니다.

### 메모리 예측 방식

- 모델 카드의 GB는 10진수, GPU/RAM 표시는 GiB입니다. 파일이 있으면 shard 합계로 가중치 용량을 보정합니다. 파일 헤더와 분할 파일 존재를 검사하며 해시/전체 무결성 검증은 HF 캐시와 엔진에 의존합니다.
- CPU Expert 용량은 총 가중치 × Expert 비율(기본 추정 0.92) × CPU MoE 층 비율로 추산합니다. 실제 층별 정밀도/배치 차이가 반영되지 않으므로 오차가 클 수 있습니다.
- Inkling 2의 8 full + 40 sliding 레이어, KV head 8, head dim 128, window 1024 기반. K와 V를 합산하고 Q8_0/Q4_0 scale overhead를 포함합니다. sliding cache가 실제로 절약되는 경우와 전체 버퍼가 필요한 경우를 함께 표시합니다. 별도 상태/체크포인트, allocator, vision projector, 운영체제, mmap 캐시 등은 여유 예산으로만 추정합니다.
- GPU별 예상치, 현재 여유 VRAM, RAM 상한, 미인증 네트워크 노출에 대한 경고를 확인합니다. 위험을 허용해도 실행 파일/필수 shard/지원 옵션/사용 가능 GPU가 없으면 실행하지 않습니다. 추정 통과는 OOM 방지를 보장하지 않습니다.

## VS Code / Claude Code 연결

1. 서버의 로컬 UI에서 대화합니다. 팀 네트워크 사용 승인 후 서버 설정에서 `0.0.0.0`을 선택하고 필요 시 API 키를 활성화합니다.
2. 서버의 PowerShell에서 `ipconfig`로 사내 IP 확인. Windows 방화벽은 승인된 내부 대역으로만 인바운드 TCP 8080을 허용해야 합니다(관리 도구에서 자동 변경하지 않습니다).
3. 노트북 PowerShell: `Test-NetConnection <서버IP> -Port 8080`.
4. OpenAI 호환 클라이언트는 `http://<서버IP>:8080/v1`; Anthropic 호환 클라이언트는 `http://<서버IP>:8080`을 사용합니다. 실제 llama.cpp 버전의 `/v1/messages` 지원 및 tool call / reasoning parsing을 확인하세요.
5. Claude Code의 설정 예(노트북 PowerShell에서 실행하거나 확장 설정에 입력):

```powershell
$env:ANTHROPIC_BASE_URL = 'http://<서버IP>:8080'
$env:ANTHROPIC_AUTH_TOKEN = '<발급된 API 키 또는 미인증 테스트용 local>'
$env:ANTHROPIC_MODEL = 'inkling-2-small'
code .
```

확장의 기존 프로세스가 환경 변수를 상속하지 않으면 VS Code를 완전히 종료 후 다시 실행하거나 확장의 해당 설정을 사용합니다. Claude Code 버전에 따라 별도 모델 환경변수나 API 호환 어댑터가 필요할 수 있습니다. 사내 코드가 외부 API로 전송되지 않도록 승인된 네트워크 정책을 적용하세요. 본 도구는 도구 호출 호환성과 외부 전송 차단을 보장하지 않습니다.

## API 키 / 운영 보안

- 기본은 인증 없음 + `127.0.0.1` 추론 API입니다. 관리 UI는 항상 loopback이며 외부 바인딩하지 않습니다. 관리 세션은 로컬 Origin/Host와 임의 세션 토큰을 검사합니다. 같은 OS 계정은 신뢰한다고 가정합니다.
- API 키 탭에서 키 생성·회수, 실행 설정에서 인증 활성화. 키를 적용하려면 서버를 재시작합니다. `--api-key-file`을 사용하며 키 값은 명령 로그 및 config.json에 저장하지 않습니다. 서버별 사용자 키 파일은 평문이므로 OS 계정 권한/ACL, 승인된 역방향 프록시 + TLS 등을 별도로 적용할 것. 미인증 LAN에서는 `/slots` 노출을 끕니다. `/health` 등 일부 엔드포인트는 엔진 정책에 따라 공개될 수 있습니다.
- 사용자별 quota/키 별 통계/스케줄링, TLS, SSO, 시스템 서비스 등록, 서버 재부팅 자동 실행은 이 버전에 포함되지 않습니다. 15명 서비스 전 교차 사용자 데이터 격리, 동시 tool call, 유휴 캐시 복원 테스트를 필수로 수행하세요.
- 상태/로그는 이 앱이 시작한 프로세스에 한해 모니터링합니다. 로그는 최근 900줄 메모리 보관이며 사용자 프롬프트가 포함될 수 있으므로 공유하지 마세요. 터미널 Ctrl+C로 정상 종료하면 관리 작업을 중지합니다. 강제 종료나 로그오프 시 작업이 남거나 중단될 수 있으므로 작업 상태를 확인하세요. Windows 서비스가 아닙니다.

## 다른 모델 지원

`config.json`의 `model_profiles`에 모델 key를 추가하고 프로그램을 재실행하면 선택 목록에 나타납니다. 아래 양식을 복사해 실제 모델 카드 및 config에 맞게 수정하세요. `pattern`은 Hugging Face repo 상대경로 glob(예: `Q4_K_M/*` 또는 `*Q4_K_M.gguf`)입니다. 모델 계열을 변경하면 직접 지정 경로를 초기화합니다. 일반 GGUF에 Inkling 전용 reasoning template을 적용하지 마세요. 99층 이상 모델은 `manager.py`의 `-ngl 99`를 `-ngl all`로 바꾸세요(지원 빌드 한정).

```json
"another-model": {
  "label": "Another MoE GGUF", "repo": "ORG/MODEL-GGUF", "revision": "main",
  "max_context": 131072, "layers": 32, "full_layers": 32,
  "sliding_layers": 0, "window": 0, "kv_heads": 8, "head_dim": 128,
  "expert_fraction_estimate": 0.90, "reasoning_template": false,
  "quants": {"Q4_K_M": {"gb": 50.0, "agreement": null, "pattern": "*Q4_K_M.gguf"}}
}
```

이는 예시 값이며 실제 모델 사양이 아닙니다. MLA/SSM/다른 KV 구조는 별도 estimator 변경이 필요합니다. Dense 모델은 expert_fraction_estimate=0으로 하고 GPU 모드 사용. 모델 교체 시 필요한 별도 템플릿/추론 파서도 직접 적용해야 합니다.

경로 자동 인식: 기존 `AI/llama.cpp` 또는 `AI/models` 우선, 없으면 여유 공간이 큰 고정 디스크. GUI에서 바꿀 수 있으며 개인 설정은 `%LOCALAPPDATA%\InklingManager\settings.json` (키는 같은 폴더 `api-keys.txt`)에 저장됩니다. GitHub에는 이 네 소스 파일만 올리고 모델, build 결과, API 키, 개인 로그를 추가하지 마세요.

## 문제 해결

| 현상 | 확인 사항 |
|---|---|
| NVIDIA GPU 미감지 | `nvidia-smi`, 장치 관리자, 드라이버, RDP/관리자 환경 확인 |
| 다운로드 API/403/SSL 오류 | HF 접근 권한 및 회사 승인 네트워크/HF CDN 연결 확인. 회사 CA 설정을 사용하고 TLS 검증을 끄지 마세요 |
| Download 중 공간 부족 | 디스크 여유, 경로 확인. partial cache가 존재해도 잔여 전체 파일 기준 보수적 계산을 사용할 수 있음 |
| CMake / CUDA / MSVC 오류 | CUDA 버전-MSVC 호환성 확인, 개발자 PowerShell에서 다시 실행. 기존 build 폴더 이름 변경 후 재빌드. 변경 파일이 있는 Git 소스는 직접 확인 |
| llama-server 실행 불가 | CUDA DLL, VC++ Redistributable, 실행 파일 직접 경로, Inkling2 지원 빌드 확인 |
| `unknown argument` | 최신 빌드로 업데이트하거나 해당 옵션 해제. 자동 무시하지 않음 |
| `out of memory` | context/pool/parallel/ubatch/Expert cache 줄이기, CPU MoE 층 늘리기, GPU별 split 조정 |
| `unknown model architecture` | 최신 Inkling 2 호환 llama.cpp 빌드/정확한 GGUF 확인 |
| RAM KV 느림 | 활성 KV GPU 사용, 더 낮은 KV 정밀도를 검증하거나 컨텍스트 축소 |
| 다른 노트북에서 접속 실패 | 호스트 설정, 사내망 ACL/방화벽, 포트 및 인증 확인 |

## 참고 원문 (2026-11-20 확인)

- [Unsloth Inkling 2 Small GGUF, quant table / 실행 방법](https://huggingface.co/unsloth/Inkling-2-Small-GGUF)
- [Inkling 2 config](https://huggingface.co/thinkingmachines/Inkling-2-Small/blob/main/config.json)
- [llama.cpp server 옵션](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [llama.cpp MoE cache](https://github.com/ggml-org/llama.cpp/blob/master/docs/moe-cache.md)
- [llama.cpp Windows 빌드](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- [HF snapshot_download](https://huggingface.co/docs/huggingface_hub/package_reference/file_download)
