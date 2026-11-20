# Easy LLM Manager v0.2

Windows 기반 llama.cpp 로컬 추론 서버의 다운로드, 컴파일, 실행과 모니터링을 위한 소형 관리 도구.
Python 표준 라이브러리로 관리 웹 UI를 제공하며 모델 다운로드에만 `huggingface_hub`를 사용합니다.

> 개발 버전(v0.2). Python/JavaScript 문법과 모의 GPU/HTTP 테스트를 검증했습니다. 실제 Windows/H100에서 CUDA 컴파일, 공식 ZIP 설치, Inkling 추론 및 1M 커널 호환성은 아직 실측하지 않았습니다. 회사 승인된 소프트웨어, 모델 및 네트워크 환경에서 사용하세요.

## 빠른 시작

1. 네 파일(`manager.py`, `config.json`, `start.bat`, `README.md`)을 같은 폴더에 복사합니다.
2. Python 3.12 x64 설치 후 `start.bat` 더블 클릭 (또는 PowerShell `py -3.12 manager.py`).
3. 자동으로 열린 브라우저(기본 `http://127.0.0.1:8765/`)에서 환경 진단을 실행합니다. 다른 포트가 선택되면 터미널에 실제 주소를 표시합니다.
4. 다운로드 탭에서 `UD-IQ2_XXS`를 선택하고 원격 파일/실제 용량 조회 후 다운로드합니다. 같은 파일/폴더/리비전에서 다운로드 버튼을 다시 누르면 Hugging Face의 기존 파일/캐시를 재사용합니다. 다운로드 시점의 `main`이 변경될 수 있으므로 완전한 재현성이 필요하면 `config.json`의 모델 `revision`을 커밋 SHA로 고정하세요. 부분 캐시가 항상 재개되는 것은 아니며 HF/Xet 동작에 따릅니다.
5. 컴파일 탭에서 CUDA Toolkit을 지정하거나 자동 탐색한 후 직접 컴파일합니다. 또는 **공식 CUDA 바이너리 설치**로 GitHub의 실행 파일과 대응 DLL을 설치할 수 있습니다. 완료 후 **설치/빌드 결과 실행 파일 적용**을 누르세요. 기존 바이너리를 직접 지정해도 됩니다.
6. 서버 실행 탭에서 모델 선택, **첫 구동 (2 GPU)**, 예상 자원 계산, 서버 시작 순으로 진행합니다. 모니터에서 `/health`가 응답하면 로컬 llama.cpp UI로 대화해 보세요.

### 필요한 환경

- [Python 3.12](https://www.python.org/downloads/), [Git](https://git-scm.com/download/win), [CMake](https://cmake.org/download/)
- [NVIDIA Data Center 드라이버](https://www.nvidia.com/Download/index.aspx), [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) — C++ 데스크톱 개발, MSVC x64/x86, Windows SDK 및 CMake 구성 요소. CUDA와 MSVC 버전 호환성 확인.
- [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub/installation) — GUI 환경 탭에서 설치 가능. PyYAML wheel 오류는 Python 3.12 및 `py -3.12 -m pip install --upgrade pip` 확인.
- Visual Studio 2022/2026의 C++ 설치 경로는 `vswhere`로 찾습니다. 빌드 실패 시 **x64 Native Tools Command Prompt / Developer PowerShell**에서 `start.bat`을 다시 실행하세요. 설치된 CMake가 해당 Visual Studio 생성기를 지원해야 합니다.

## v0.2 주요 추가 기능

### CUDA 12.2 / 컴파일 실패 대응

- 환경 진단에서 실제 `nvcc --version`을 검사하여 CUDA Toolkit 후보를 탐색합니다. 필요하면 컴파일 탭에서 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2` 같은 Toolkit 루트 경로를 직접 지정할 수 있습니다. `nvidia-smi`의 CUDA 표시는 드라이버 지원 수준이지 Toolkit 설치 버전이 아닙니다.
- 빌드는 Toolkit/Visual Studio/SM별 `build-cuda-*` 폴더로 분리합니다. **새 빌드 폴더 사용**을 켜면 기존 컴파일 파일을 삭제하지 않고 새로 구성합니다. CUDA/컴파일러/공통 CMake 오류에는 로그 기반 원인 후보를 표시합니다. CUDA 12.2와 최신 MSVC 또는 최신 소스의 호환성을 보장하지 않습니다.
- 컴파일 대신 공식 Windows x64 CUDA 12 또는 13 배포본을 선택할 수 있습니다. GitHub 릴리스 API에서 동일한 태그/정확한 CUDA 버전의 실행 파일 ZIP과 DLL ZIP을 찾아 저장하고, 크기와 제공되는 SHA-256 digest를 확인합니다. digest가 없는 릴리스는 로그에 표시합니다. 공식 ZIP은 새 폴더에 설치하며 기존 바이너리를 덮어쓰지 않습니다. 배포본은 Toolkit 없이도 동작할 수 있지만 **NVIDIA 드라이버 및 VC++ 런타임, 모델/옵션 지원**은 별개입니다. 릴리스가 최신이라도 실제 CUDA 파일이 오래된 경우가 있으므로 `--version`과 모델 로딩을 검증해야 합니다. 네트워크 차단 시 공식 [Releases](https://github.com/ggml-org/llama.cpp/releases)에서 승인된 경로로 두 ZIP을 받아 직접 지정하세요.

### H100 NVL 2장 + Q4 1M

서버 실행 화면에서 **Q4 1M / 2 GPU**를 선택하면 Q4_K_XL, 1인, 1,048,576 토큰, Q8_0 GPU KV, CPU MoE 32층, GPU Expert cache 8GiB(초기 후보)를 설정합니다. 이는 실행 보장이 아닙니다. 먼저 32K/128K에서 시험하고, 이후 1M으로 늘리세요. 256GB RAM에서도 실제 여유량에 따라 1M을 수용하지 못할 수 있습니다.

- **자동 메모리 배치 추천**: 현재 모델, 정밀도, 슬롯 수, 컨텍스트를 변경하지 않고 정적 GPU 우선 및 동적 Expert cache 8/16/24GiB 후보를 계산합니다. 물리 GPU마다 2GiB, 시스템 RAM 4GiB의 추가 예산을 요구합니다. 선택한 runtime/RAM reserve도 포함됩니다. 결과는 *추정치*이며 GPU별 레이어 편차, Expert 정밀도, SWA 전체 할당 여부에 따라 실측이 달라집니다. 후보를 선택한 뒤 다시 예상 자원을 검사하세요. 전체 컨텍스트 길이를 자동 축소하지 않습니다.
- **실측 벤치마크**: 관리 도구에서 실행 중인 단일 슬롯 서버에 합성 입력과 출력 토큰을 넣고 `/tokenize`, `/completion`의 `timings`로 prefill/decode 속도를 측정합니다. 사용자 프롬프트를 사용하지 않습니다. 실제로 생성한 토큰 수와 요청한 값이 다를 수 있습니다. 기존 KV/cache에 영향이 있으므로 다른 사용자가 작업하지 않을 때만 사용합니다. 실행 중인 수동 벤치마크를 중단하려면 서버를 중지하세요.
- **자동 튜닝**: 서버를 중지하고 단일 슬롯, GPU + CPU MoE 모드를 선택한 뒤 성능 튜닝 화면을 사용합니다. 정적 후보와 8/16/24GiB 캐시 후보(가능한 것만), 가장 큰 캐시 후보에서 CPU share 0을 추가하여 최대 다섯 구성을 차례로 실제 로딩합니다. 로딩 및 메모리 검사를 실패한 구성은 건너뜁니다. 각 후보에서 동일한 길이의 합성 요청 두 개를 실행하고 두 번째(warm) decode 속도를 비교합니다. 모델 리로드가 매우 오래 걸릴 수 있습니다. 512 입력 / 128 생성부터 시험하세요. 1M 컨텍스트를 *설정*한 벤치마크와 1M 토큰을 *채운* 벤치마크는 다릅니다. 긴 입력은 실제 입력 길이를 늘려 재측정하세요. 합성 입력만 측정하므로 실제 Claude Code 작업이나 모든 캐시 적중 패턴의 최적값은 보장하지 않습니다. 가장 빠른 후보는 사용자 확인 후 적용하며 자동 실행하거나 설정을 저장하지 않습니다.
- 측정 이력은 `%LOCALAPPDATA%\InklingManager\benchmark-history.json`에 최대 50건 저장합니다(수치와 설정만, 프롬프트/생성내용 제외). Expert cache 적중률은 실행 화면의 **Expert 캐시 상세 로그**를 선택했을 때 지원 엔진이 내보내는 로그에서 최근 900줄 범위를 집계합니다. 해당 로그가 없는 빌드에서는 값이 나타나지 않습니다. 장시간 상세 로깅은 성능과 로그 양에 영향을 줄 수 있습니다.

### v0.2 범위 밖

일반 서버 요청의 자동 OOM 재실행, 자동 GitHub 자체 업데이트, Windows 서비스 등록, TLS, 키별 quota 및 사용량 집계는 아직 제공하지 않습니다. 자동 튜닝에서만 실패 후보를 건너뜁니다. 기존 v0.1의 모델 다운로드 이어받기, YAML 없는 표준 라이브러리 UI, localhost 관리자 접근 제한, API 키 설정은 유지됩니다.

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
- 상태/로그는 이 앱이 시작한 프로세스에 한해 모니터링합니다. 로그는 작업별 최근 900줄 메모리 보관이며 사용자 프롬프트가 포함될 수 있으므로 공유하지 마세요. 새 작업을 시작하면 해당 그룹의 로그가 초기화됩니다. 터미널 Ctrl+C로 정상 종료하면 관리 작업을 중지합니다. 강제 종료나 로그오프 시 작업이 남거나 중단될 수 있으므로 작업 상태를 확인하세요. Windows 서비스가 아닙니다.

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
| CMake / CUDA / MSVC 오류 | CUDA 버전-MSVC 호환성 확인, 개발자 PowerShell에서 다시 실행. 새 빌드 폴더 사용 체크 후 재빌드. 상세 로그의 자동 진단 확인 또는 공식 바이너리 사용. 변경 파일이 있는 Git 소스는 직접 확인 |
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

## v0.2.1 — 서버/다운로드 동시 작업 및 오류 복구

- 다운로드 탭과 서버 실행 탭에서 각각 모델/양자화/vision을 선택합니다. 다운로드를 시작해도 선택한 서버 모델은 바뀌지 않습니다. 다운로드 대상 경로/패턴과 실행 중인 모든 GGUF shard 및 projector를 비교하고 충돌 시 차단합니다. 다른 모델은 다운로드 중에도 실행할 수 있습니다. 외부 프로그램이 파일을 변경하지 않도록 주의하세요. 다운로드는 디스크/RAM 대역폭을 사용하므로 속도가 감소할 수 있습니다. 빌드/자동 튜닝 충돌 제한은 유지됩니다.
- 서버 실행 탭의 **8K 안전 구동** 프리셋은 선택 모델에 맞는 CPU MoE 층 후보, F16 KV, 8192 context, Expert cache 0을 설정합니다. H100 NVL 두 장 기준의 시작점이며 특히 대형 모델의 OOM을 보장하지 않습니다.
- **오류 복구 옵션**에 JSON 객체를 입력하면 생성된 명령을 수정합니다. 예: `{"--flash-attn":"off","--no-warmup":true,"--no-kv-unified":null}`. `null`은 기존 옵션 제거, `true`는 단독 스위치 추가, 문자열은 허용된 옵션 값 변경입니다. `{"--no-mmap":true}`는 mmap 비활성화 실험이며 모델 전체 메모리 로딩 때문에 RAM/커밋 한도를 크게 늘릴 수 있으므로 주의하세요. `{"--fit":"on"}`은 엔진이 배치를 변경할 수 있습니다. 복구 옵션을 적용한 경우 예상 메모리와 실제 명령이 다를 수 있어 재확인합니다. 모델/경로/포트/API 키 등은 GUI에서 변경하며 임의 shell 실행을 지원하지 않습니다. 복구 옵션은 저장 시 사용자 설정에 보관되며 초기화 버튼으로 제거할 수 있습니다. 자동 튜닝을 시작하기 전에 비워야 합니다.
- 모니터링 탭에 최근 실행 요청 오류 및 실패 로그 핵심 줄을 표시합니다. `llama-server --version`, `llama-server --help`, `llama-server --list-devices`, `nvidia-smi`, `nvidia-smi -q`, `nvcc --version`을 진단 명령에 입력해 로컬 출력으로 확인합니다. 별도의 임의 PowerShell 명령은 승인된 터미널에서 직접 실행하세요. 로그를 외부로 보내는 기능은 없습니다.
- 오류 원인을 확인할 때 마지막 `failed to load model`보다 그 앞의 구체적 오류(예: unknown architecture, allocation failure, assertion)를 우선 확인하세요.
- Python/JavaScript 구문 확인과 모의 동시 작업 및 복구 옵션 검증을 수행했습니다. 실제 Windows H100 로딩은 검증되지 않았으며 기존 모델 파일의 다운로드 무결성과 엔진 호환성은 별도 확인해야 합니다.


## v0.2.2 간편 실행

서버 실행 탭에서 모델, 사용 목적, 슬롯당 컨텍스트, KV 위치, 가중치 배치 방식을 선택하고 추천값 적용 또는 추천값으로 실행을 누릅니다. GPU/RAM 여유를 기준으로 CPU MoE 층, 캐시, 런타임 여유 공간, 배치와 스레드를 추천합니다. 요청 컨텍스트를 자동으로 낮추지 않으며 실제 속도/메모리 성공은 실측이 필요합니다. 팀 설정은 슬롯별 독립 컨텍스트로 계산합니다. 동적 Expert 캐시는 실험 옵션이며 지원 빌드가 필요합니다. 복구 옵션과 기존 인증/네트워크 설정은 유지됩니다. 고급 설정에서 모든 숫자를 수정할 수 있습니다.

텍스트 전용 모드는 명시적으로 `--no-mmproj`를 전달하고 상속된 projector 관련 환경변수를 제거합니다. `unknown projector type: inkling2` 오류는 언어 모델의 `inkling2`와 upstream vision projector의 `inkling2v` 형식 차이로 발생할 수 있습니다. Vision 체크 해제, 첫 GGUF shard 경로 확인, 필요하면 별도의 projector 파일만 호환 버전으로 교체하세요. 복구 JSON에서 `{"--mmproj":null,"--no-mmproj":true}`도 사용할 수 있습니다. Windows H100 실측은 아직 수행하지 않았습니다.
