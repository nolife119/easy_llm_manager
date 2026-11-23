# Inkling 2 Mini · Windows H100 DSpark 매니저

**새 브랜치 전용 진입점: \`start-mini.bat\`**. 기존 \`start.bat\`, \`manager.py\`, \`ui.html\`, \`config.json\`은 건드리지 않습니다.

## 검증된 조합

| 구성 | 내용 |
| --- | --- |
| 하드웨어 | Windows Server x64, H100 NVL 94GB 단일 GPU |
| 서버 엔진 | **사용자가 미리 압축 해제한** Unsloth llama.cpp Windows x64 CUDA13 **newer**, release \`b12588-mix-c0c3de8\` |
| 모델 | \`unsloth/Inkling-2-Mini-GGUF\`의 루트 \`Inkling-2-Mini-UD-Q8_K_XL.gguf\` (약 22.87GB) |
| DSpark | 루트 \`dspark-Inkling-2-Mini-Q8_0.gguf\` (약 402MB) |
| 이미지 | 루트 \`mmproj-F16.gguf\` (약 639MB) |
| 컨텍스트 | 기본 32K, 8K~1M 실험적인 선택 가능 |
| API | 엔진 원본 \`/v1/chat/completions\`, 지원 엔진의 \`/v1/messages\` |
| 대시보드 | \`http://127.0.0.1:8766\` |
| 추론 서버 | \`http://서버IP:8080\` 기본, 별도 프록시 없음 |

**중요:** 이 8-bit 모델은 원본 FP8 체크포인트와 동일한 파일이 아닙니다. GGUF UD-Q8_K_XL 양자화입니다. 원본 약 20GB 체크포인트의 텐서들은 혼합 NVFP4/MXFP8 형식이며, 단순히 FP8이라 부르는 것과 차이가 있습니다. Unsloth의 DSpark는 **드래프트/가속용 별도 작은 모델**이며 8-bit 본 모델을 대체하지 않습니다.

Unsloth의 루트 GGUF는 looped layout 전용으로, 이 빌드에 맞춰 사용합니다. *Mainline_Support* 폴더의 \`.gguf_file\`은 mainline 엔진을 위한 다른 레이아웃이므로 이 매니저와 혼용하지 마세요.

출처:
- [Unsloth 모델과 사이즈](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF)
- [메인라인/Unsloth 엔진 호환성 및 DSpark 실행 예시](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/blob/main/Mainline_Support/README.md)
- [Unsloth의 Windows CUDA13 newer 다운로드](https://github.com/unslothai/llama.cpp/releases/tag/b12588-mix-c0c3de8)
- [llama.cpp metrics API](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)

## 시작

1. Python 3.10 이상 권장 (3.12 테스트), Windows x64의 **\`start-mini.bat\`**를 실행하면 브라우저가 열립니다.
2. 이미 다운로드/압축 해제해 둔 **Unsloth CUDA13 newer \`llama-server.exe\`의 전체 경로**를 입력하고 **① 엔진 확인**을 클릭합니다. **엔진을 다시 다운로드하지 않습니다.** NVIDIA 드라이버에서 실행 파일의 종속 DLL 로드가 가능한지 확인해야 합니다. 필요하면 재배포 패키지 등을 별도로 준비합니다.
3. **② 모델 받기**를 눌러 ~24GB의 3개 GGUF를 받습니다. 최초 1회 \`huggingface_hub[hf_xet]\`를 자동 설치하며, HF 인증/프록시가 필요한 환경에서는 해당 환경에 맞춰 선행 설정해야 합니다. 허깅페이스 캐시의 중단 재개 기능을 사용하고 파일별 SHA-256을 검사합니다.
4. **③ 서버 시작**을 누릅니다. 서버가 Health OK가 되면 **기능 검증**을 실행해 텍스트, Anthropic count_tokens, tool call, 이미지 입력을 시험합니다. 검사 실패가 있는 경우 로그에서 이유를 확인하세요.

다운로드 파일은 기본적으로 \`%LOCALAPPDATA%\InklingMiniManager\models\`에 보관합니다. 기본 \`n_ctx=32768\`, \`parallel=1\`, \`ngl=99\`, \`DSpark=3 tokens\`, \`KV=FP16\`. 설정 파일은 \`%LOCALAPPDATA%\InklingMiniManager\mini-settings.json\`.

### Claude Code 연결 (PowerShell)

아래 명령은 llama-server에 해당 **Anthropic Messages API가 실제 구현되어 있다**는 전제입니다. 반드시 대시보드 검증 결과 및 로컬 \`/v1/messages\` 호출을 확인하세요.

\`\`\`powershell
$env:ANTHROPIC_BASE_URL="http://127.0.0.1:8080"
$env:ANTHROPIC_AUTH_TOKEN="local-inkling"
$env:ANTHROPIC_API_KEY=""
claude --model inkling-mini
\`\`\`

Tool calling은 \`--jinja\`와 모델 템플릿을 사용합니다. Claude Code의 도구 호출과 실제 파일 작업은 별도의 통합 테스트가 필요합니다. \`/v1/messages/count_tokens\`가 구현되지 않았다면 Claude Code의 일부 기능에 제약이 있을 수 있습니다.

## 안전 및 성능

- 대시보드와 추론 서버 모두 **127.0.0.1 전용**. 사내망에 공개하려면 인증/방화벽 정책을 따로 설계해야 합니다.
- 관리 API는 localhost로만 수신하며 Origin/Host 검사도 합니다. 브라우저 제어 API이므로 공용 PC나 악성 로컬 프로세스에 대한 보호 경계는 아닙니다.
- Windows에서 \`nvidia-smi\` 가 감지한 GPU 인덱스를 \`CUDA_VISIBLE_DEVICES\`로 제한합니다. NVLink, 두 번째 GPU, WSL2, Docker, Ubuntu를 사용하지 않습니다.
- 기본 배치 2048, micro batch 512, \`--flash-attn on\`은 해당 **엔진 --help가 지원할 경우에만** 사용합니다. 실행 시 버전에 따른 flag 차이가 있으면 로그를 보고 설정을 바꾸세요.
- 1M 컨텍스트는 모델 상한/요청이며 **실제 H100 한 장 메모리 적재 및 속도 검증을 뜻하지 않습니다**. 긴 컨텍스트는 프리필 지연 및 KV 크기 증가로 성능이 급격히 나빠질 수 있습니다.
- DSpark 미사용은 설정에서 명시적으로 끄는 경우에만 사용합니다. 자동으로 조용히 가속을 포기하지 않습니다.
- GPU 모니터링은 nvidia-smi, prefill/decode는 엔진의 Prometheus \`/metrics\` 카운터의 구간 비율을 사용합니다. 최근 구간에 요청이 없으면 t/s가 표시되지 않을 수 있습니다.
- 다운로드 무결성은 Hugging Face 파일 metadata의 SHA-256과 받은 파일의 실제 SHA-256을 비교합니다. 이후 동일한 크기·수정시각인 파일은 검증 기록을 재사용합니다.
- 환경/실기 제약으로 **Windows Server/H100/Claude Code 전체 통합 시험 및 실제 t/s는 이 저장소에서 미검증**입니다.

## 개발자 회귀 시험

\`\`\`powershell
python -m py_compile mini_runtime.py mini_platform.py
python -m unittest discover -s tests -p "test_mini*.py" -v
\`\`\`

GitHub Actions가 Windows와 Ubuntu에서 CPU-only 테스트, JS 문법 검사 수행. Windows H100 통합 검증은 별도입니다.


## Windows 방화벽 자동 설정 및 LAN 접속

`start-mini.bat` 실행 시 `mini_firewall.py`가 Windows 방화벽을 점검합니다. 
LAN 설정(기본 켜짐)이면 `Inkling Mini LLM API (LocalSubnet)` 이름의 인바운드 규칙을 생성합니다.
규칙은 **TCP 추론 포트(기본 8080)** 만 허용하고, **Domain/Private 네트워크 프로필**과
**RemoteAddress=LocalSubnet**으로 범위를 제한합니다. 기존 규칙이 정확하면 재설정하지 않습니다.
필요 시 **Windows UAC 관리자 권한 승인**을 요청하며, 거절되거나 생성·검증에 실패하면
`start-mini.bat`가 LAN 서비스를 시작하지 않습니다. 서버 실행 시에도 현재 API 포트에 대한
규칙을 확인하므로 포트를 변경한 경우에도 같은 확인 절차를 거칩니다.

- 추론 API는 기본 `0.0.0.0:8080` 바인딩이며, 같은 서브넷의 다른 PC에서 `http://서버IP:8080`으로 사용합니다.
- 웹 관리 대시보드 **8766은 계속 `127.0.0.1` 전용**이며 방화벽에서 열지 않습니다.
- 대시보드 **LAN 연결 허용**을 끄면 추론 API도 `127.0.0.1` 전용으로 돌아갑니다.
- **인증 미구현:** 기본 llama-server API는 별도 인증이 없으므로 신뢰할 수 있는 네트워크에서만 사용하세요. LocalSubnet 범위가 신뢰할 수 없는 공유망을 포함한다면 사용하지 마세요.
- Domain/Private 프로필이 아니거나, 원격 PC가 다른 서브넷이거나, 조직 GPO에서 로컬 인바운드 규칙을 막으면 접속되지 않을 수 있습니다. 관리자 또는 네트워크팀 승인 정책이 우선합니다.
- 다른 PC의 Claude Code 연결 시 `ANTHROPIC_BASE_URL=http://서버IP:8080`으로 변경합니다. 토큰 문자열은 이 설정만으로 API 인증이 활성화되는 것이 아닙니다.
