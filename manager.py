"""Local-only Windows llama.cpp manager. Python 3.12, standard library UI."""
from __future__ import annotations
import ctypes, fnmatch, hashlib, json, os, re, secrets, shutil, socket, subprocess, sys, tempfile, threading, time, traceback, urllib.request, urllib.error, webbrowser, zipfile
from collections import deque
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = Path(__file__).resolve().parent
GIB = 1024**3
WIN = os.name == 'nt'
BASE = json.loads((HERE/'config.json').read_text(encoding='utf-8'))
TOKEN = secrets.token_urlsafe(32)
LOCK = threading.RLock()
TASKS = {}
SETTINGS = {}
LOGS = {k: deque(maxlen=900) for k in ('download','build','server','benchmark','tune','system')}
CREATE = subprocess.CREATE_NO_WINDOW if WIN else 0


def auto_workspace():
    roots = [Path(HERE.anchor)]
    if WIN:
        mask = ctypes.windll.kernel32.GetLogicalDrives()
        roots = [Path(f'{chr(65+i)}:/') for i in range(26) if mask & (1<<i) and ctypes.windll.kernel32.GetDriveTypeW(f'{chr(65+i)}:\\') == 3]
    candidates = []
    for r in roots:
        try: candidates.append((shutil.disk_usage(r).free,r))
        except OSError: pass
    # Avoid writing source tree; prefer already established D:\AI when present.
    for _,r in candidates:
        if (r/'AI'/'llama.cpp').is_dir() or (r/'AI'/'models').is_dir(): return r/'AI'
    if WIN and candidates:
        r=max(candidates,key=lambda x:x[0])[1]
        return r/'AI' if r != Path(os.environ.get('SystemDrive','C:')+'/') else Path.home()/'AI'
    return Path.home()/'AI'


def absolute(p): return Path(os.path.expandvars(str(p))).expanduser().resolve()
WORK=absolute(BASE['workspace']) if BASE['workspace'] else auto_workspace()
# State belongs outside git; use LOCALAPPDATA for path preferences, not credentials.
STATE=Path(os.environ.get('LOCALAPPDATA',str(Path.home()/'.local'/'state')))/'InklingManager'
STATE.mkdir(parents=True,exist_ok=True)
PREF=STATE/'settings.json'
KEYS=STATE/'api-keys.txt'
BENCH_HISTORY=STATE/'benchmark-history.json'
try: SETTINGS=json.loads(PREF.read_text(encoding='utf-8'))
except (OSError, ValueError): SETTINGS={}


def paths(d=None):
    d={**SETTINGS,**(d or {})}
    w=absolute(d.get('workspace') or BASE['workspace'] or WORK)
    l=absolute(d.get('llama_dir') or BASE['llama_dir'] or w/'llama.cpp')
    m=absolute(d.get('model_dir') or BASE['model_dir'] or w/'models')
    x=d.get('server_exe') or BASE['server_exe']
    return w,l,m,absolute(x) if x else None


def log(kind,msg):
    msg=str(msg).rstrip('\r\n')
    if not msg: return
    with LOCK: LOGS[kind].append(time.strftime('%H:%M:%S')+' '+msg)


def run_capture(args, timeout=10, env=None):
    try:
        p=subprocess.run([str(a) for a in args],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout,env=env,creationflags=CREATE)
        return p.returncode,p.stdout+'\n'+p.stderr
    except (OSError,subprocess.TimeoutExpired) as e: return -1,str(e)


def find_tool(name):
    x=shutil.which(name)
    if x:return x
    pf=Path(os.environ.get('ProgramFiles','C:/Program Files'))
    pf86=Path(os.environ.get('ProgramFiles(x86)','C:/Program Files (x86)'))
    known={'git':[pf/'Git/cmd/git.exe'],'cmake':[pf/'CMake/bin/cmake.exe'], 'nvidia-smi':[pf/'NVIDIA Corporation/NVSMI/nvidia-smi.exe',Path(os.environ.get('SystemRoot','C:/Windows'))/'System32/nvidia-smi.exe'], 'nvcc':[], 'vswhere':[pf86/'Microsoft Visual Studio/Installer/vswhere.exe']}
    if name=='nvcc':
        if os.environ.get('CUDA_PATH'): known[name].append(Path(os.environ['CUDA_PATH'])/'bin/nvcc.exe')
        known[name] += sorted((pf/'NVIDIA GPU Computing Toolkit/CUDA').glob('v*/bin/nvcc.exe'),key=lambda p:tuple(map(int,re.findall(r'\d+',p.parent.parent.name))),reverse=True)
    for p in known.get(name,[]):
        if p.is_file():return str(p)
    return None


def cuda_candidates():
    pf=Path(os.environ.get('ProgramFiles','C:/Program Files'))
    roots=[Path(x) for k,x in os.environ.items() if x and (k=='CUDA_PATH' or k.startswith('CUDA_PATH_V'))]
    roots+=list((pf/'NVIDIA GPU Computing Toolkit/CUDA').glob('v*'))
    tool=shutil.which('nvcc')
    if tool:roots.append(Path(tool).parent.parent)
    found={}
    for root in roots:
        try:root=root.resolve()
        except OSError:continue
        nv=root/'bin'/('nvcc.exe' if WIN else 'nvcc')
        if not nv.is_file():continue
        rc,out=run_capture([nv,'--version'],8)
        match=re.search(r'release\s+(\d+)\.(\d+)',out)
        if rc==0 and match:found[str(root)]={'root':str(root),'nvcc':str(nv),'version':'.'.join(match.groups()),'order':tuple(map(int,match.groups()))}
    return sorted(found.values(),key=lambda x:x['order'],reverse=True)


def cuda_toolkit(d=None):
    explicit=(d or {}).get('cuda_root','') or ''
    if explicit:
        root=absolute(explicit); nv=root/'bin'/('nvcc.exe' if WIN else 'nvcc')
        if not nv.is_file():raise ValueError('지정한 CUDA Toolkit의 bin/nvcc를 찾지 못했습니다: '+str(root))
        rc,out=run_capture([nv,'--version'],8);match=re.search(r'release\s+(\d+)\.(\d+)',out)
        if rc or not match:raise ValueError('지정한 nvcc 실행 실패: '+out[-600:])
        return {'root':str(root),'nvcc':str(nv),'version':'.'.join(match.groups())}
    found=cuda_candidates()
    return found[0] if found else None


def build_hint(message):
    x=message.lower()
    if 'unknown projector type' in x:
        return 'Vision projector 형식 불일치: Inkling 2의 언어 모델은 inkling2, upstream projector는 inkling2v입니다. Vision 해제 및 --no-mmproj 확인, 모델 파일이 mmproj가 아닌 첫 shard인지 확인하세요.'
    if any(t in x for t in ('no cuda toolset','could not find nvcc','could not find cudatoolkit','cuda toolkit not found','failed to find nvcc','cuda compiler identification is unknown')):
        return 'CUDA Toolkit 경로와 Visual Studio CUDA Build Customizations를 확인하세요. 공식 바이너리 설치도 가능합니다.'
    if any(t in x for t in ('unsupported microsoft visual studio','unsupported compiler','unsupported host compiler','host compiler targets','compiler is not supported')):
        return 'CUDA Toolkit과 MSVC 호환성 문제일 수 있습니다. 호환되는 CUDA/MSVC 버전을 설치하세요. 강제 우회 플래그는 사용하지 않습니다.'
    if any(t in x for t in ('does not match the generator','does not match the toolset','cmakecache.txt','does not match the source')):
        return '이전 CMake 설정과 충돌할 수 있습니다. 새 빌드 폴더 옵션으로 재시도하세요.'
    if any(t in x for t in ('out of memory','cuda error: out of memory','failed to allocate')):
        return 'VRAM/RAM 할당 실패: 컨텍스트, ubatch, Expert cache를 줄이거나 CPU MoE 층을 늘린 뒤 예상치를 확인하세요.'
    if any(t in x for t in ('unknown argument','unknown model architecture','unsupported model architecture')):
        return '모델 아키텍처 또는 옵션을 지원하는 최신 llama.cpp 빌드인지 확인하세요.'
    if any(t in x for t in ('no kernel image','invalid device function','not compiled with cuda','cuda driver version is insufficient','cublas_status_arch_mismatch')):
        return 'CUDA 커널/드라이버 호환성을 확인하세요. 실행 파일 버전과 nvidia-smi 확인, H100 arch 90 빌드 또는 호환 공식 바이너리 권장.'
    if any(t in x for t in ('invalid magic','unexpected end of file','failed to read','tensor data is not within','missing tensor','failed to open gguf','invalid gguf')):
        return 'GGUF 파일/분할 shard 또는 모델 변환 호환성 확인. 다운로드 완료 여부, 첫 shard, 최신 엔진을 확인하세요.'
    if any(t in x for t in ('flash attention','flash_attn','ggml_assert','assertion failed')):
        return 'Attention/커널 관련 오류일 수 있습니다. 8K/F16 KV/Expert 캐시 0으로 먼저 테스트하고 정확한 에러 줄을 확인하세요.'
    if any(t in x for t in ('paging file','page file','1455','bad_alloc','not enough memory')):
        return 'Windows RAM/커밋 한도 또는 페이지 파일 부족 가능성. 여유 RAM 및 페이지 파일 설정을 확인하세요.'
    return ''

def vs_install():
    if not WIN:return None
    exe=find_tool('vswhere')
    if not exe:return None
    rc,out=run_capture([exe,'-latest','-products','*','-requires','Microsoft.VisualStudio.Component.VC.Tools.x86.x64','-format','json','-utf8'])
    try:
        data=json.loads(out.strip())
        if rc==0 and data:
            p=Path(data[0]['installationPath']); major=int(data[0]['installationVersion'].split('.')[0]); return p,major
    except (ValueError,KeyError,IndexError):pass
    return None


def find_server(d=None):
    _,l,_,x=paths(d)
    if x:return x if x.is_file() else None
    for p in sorted(l.glob('build-cuda-*/bin/Release/llama-server.exe'),key=lambda a:a.stat().st_mtime,reverse=True):
        if p.is_file():return p
    for p in (l/'build/bin/Release/llama-server.exe',l/'build/bin/llama-server.exe',l/'build/bin/llama-server',l/'llama-server.exe'):
        if p.is_file():return p
    w,_,_,_=paths(d)
    for p in sorted((w/'runtime'/'llama.cpp').glob('*/**/llama-server.exe'),key=lambda a:a.stat().st_mtime,reverse=True):
        if p.is_file():return p
    x=shutil.which('llama-server');return Path(x) if x else None


def ram():
    if WIN:
        class MS(ctypes.Structure):
            _fields_=[('dwLength',ctypes.c_ulong),('dwMemoryLoad',ctypes.c_ulong)]+[(v,ctypes.c_ulonglong) for v in ['ullTotalPhys','ullAvailPhys','ullTotalPageFile','ullAvailPageFile','ullTotalVirtual','ullAvailVirtual','ullAvailExtendedVirtual']]
        s=MS();s.dwLength=ctypes.sizeof(s)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):return s.ullTotalPhys/GIB,s.ullAvailPhys/GIB
    else:
        try:
            v={x.split(':')[0]:int(x.split()[1])*1024/GIB for x in Path('/proc/meminfo').read_text().splitlines()};return v['MemTotal'],v['MemAvailable']
        except Exception:pass
    return None,None


def gpus():
    exe=find_tool('nvidia-smi'); arr=[]
    if not exe:return arr
    rc,out=run_capture([exe,'--query-gpu=index,name,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu,power.draw,driver_version','--format=csv,noheader,nounits'],4)
    if rc:return arr
    for ln in out.strip().splitlines():
        v=[x.strip() for x in ln.split(',')]
        try:
            if len(v)!=9:continue
            def f(x):
                try:return float(x)
                except ValueError:return None
            arr.append(dict(index=int(v[0]),name=v[1],total=float(v[2])/1024,free=float(v[3])/1024,used=float(v[4])/1024,util=f(v[5]),temp=f(v[6]),power=f(v[7]),driver=v[8]))
        except ValueError:continue
    return arr


def disk(path):
    while not path.exists() and path!=path.parent:path=path.parent
    try:s=shutil.disk_usage(path);return {'total':round(s.total/GIB,1),'free':round(s.free/GIB,1)}
    except OSError:return {}


def diagnostics(d=None):
    w,l,m,x=paths(d); vs=vs_install(); gs=gpus(); rt,rf=ram()
    links={'git':'https://git-scm.com/download/win','cmake':'https://cmake.org/download/','nvcc':'https://developer.nvidia.com/cuda-downloads','nvidia-smi':'https://www.nvidia.com/Download/index.aspx','vs':'https://visualstudio.microsoft.com/downloads/','hf':'https://huggingface.co/docs/huggingface_hub/installation','server':'https://github.com/ggml-org/llama.cpp/releases'}
    items=[]
    for n in ('git','cmake','nvcc','nvidia-smi'):
        p=find_tool(n); rc,out=run_capture([p,'--version'],5) if p else (-1,'')
        items.append({'name':n,'ok':bool(p) and (rc==0 or (n=='nvidia-smi' and bool(gs))),'detail':str(p or '미설치/PATH 없음')+' '+out.strip()[:250],'link':links[n]})
    rc,out=run_capture([sys.executable,'-c','import huggingface_hub; print(huggingface_hub.__version__)'])
    items += [{'name':'Visual Studio C++','ok':bool(vs) if WIN else True,'detail':str(vs or 'C++ 워크로드를 설치하세요'),'link':links['vs']},{'name':'huggingface_hub','ok':rc==0,'detail':out.strip()[:200],'link':links['hf']},{'name':'llama-server','ok':bool(find_server(d)),'detail':str(find_server(d) or '빌드 또는 경로 지정 필요'),'link':links['server']}]
    try: cuda=cuda_toolkit(d); cuda_error=''
    except ValueError as exc: cuda=None; cuda_error=str(exc)
    available=cuda_candidates()
    items.append({'name':'선택된 CUDA Toolkit','ok':bool(cuda),'detail':str(cuda_error or (cuda and (cuda['version']+' | '+cuda['root'])) or 'Toolkit 없음 (공식 바이너리 사용 가능)'),'link':links['nvcc']})
    return {'cuda_candidates':available,'items':items,'gpu':gs,'ram_total':rt,'ram_free':rf,'disk':disk(m),'paths':{'workspace':str(w),'llama_dir':str(l),'model_dir':str(m),'server_exe':str(x or find_server(d) or '')},'python':sys.version.split()[0]}


def profile(d):
    p=BASE['model_profiles'].get(d.get('profile')); q=p and p['quants'].get(d.get('quant'))
    if not p or not q:raise ValueError('모델/양자화 선택이 올바르지 않습니다.')
    return p,q


def repo_path(d):
    p,_=profile(d); _,_,m,_=paths(d)
    return m/p['repo'].replace('/','--')


def model_files(d):
    p,q=profile(d); base=repo_path(d); pattern=q['pattern']; result=[]
    if base.exists():
        result=[x for x in base.glob(pattern) if x.is_file() and x.suffix.lower()=='.gguf' and not x.name.startswith('mmproj')]
    # support older downloads and flat file layout; do not select arbitrary other quant
    _,_,m,_=paths(d)
    for r in (m/'Inkling2',m/'Inkling-2-Small-GGUF',m):
        if r==base or not r.exists():continue
        result += [x for x in r.glob(pattern) if x.is_file() and x.suffix.lower()=='.gguf' and not x.name.startswith('mmproj')]
    return sorted(set(result))


def resolve_model(d):
    if d.get('model_path'): return absolute(d['model_path'])
    files=model_files(d)
    first=[x for x in files if re.search(r'-00001-of-\d+\.gguf$',x.name,re.I)]
    single=[x for x in files if not re.search(r'-\d{5}-of-\d{5}\.gguf$',x.name,re.I)]
    return (first+single+[None])[0]


def check_shards(f):
    if not f or not f.is_file():raise ValueError('모델 파일을 찾지 못했습니다. 다운로드하거나 첫 GGUF 파일 경로를 지정하세요.')
    if f.suffix.lower()!='.gguf':raise ValueError('GGUF 파일을 지정하세요.')
    with f.open('rb') as h:
        if h.read(4)!=b'GGUF':raise ValueError('GGUF 헤더가 올바르지 않습니다: '+str(f))
    m=re.search(r'-(\d{5})-of-(\d{5})\.gguf$',f.name,re.I)
    if not m:return [f]
    if int(m[1])!=1:raise ValueError('분할 GGUF의 첫 번째 파일을 선택하세요.')
    fs=[f.with_name(f.name[:m.start()]+f'-{i:05d}-of-{int(m[2]):05d}.gguf') for i in range(1,int(m[2])+1)]
    for a in fs:
        if not a.is_file():raise ValueError('누락된 shard: '+a.name)
        with a.open('rb') as h:
            if h.read(4)!=b'GGUF':raise ValueError('손상된 shard 헤더: '+a.name)
    return fs


def num(d,k,lo,hi,integer=True):
    v=float(d[k]);
    if not lo<=v<=hi or (integer and v!=int(v)):raise ValueError(f'{k}: {lo}~{hi} 범위 필요')
    return int(v) if integer else v


def validate(d):
    p,q=profile(d); d={**BASE['defaults'],**d}
    for k,lo,hi in [('context',512,p['max_context']),('pool',512,33554432),('parallel',1,128),('cpu_layers',0,p['layers']),('expert_cache',0,1048576),('ram_cache',0,4194304),('runtime_reserve',1,1024),('ram_reserve',1,4096),('batch',1,32768),('ubatch',1,32768),('threads',1,1024),('port',1024,65535)]:d[k]=num(d,k,lo,hi)
    d['cpu_share']=num(d,'cpu_share',0,1,False)
    if not d['unified']:d['pool']=d['context']*d['parallel']
    elif d['pool']<d['context']:raise ValueError('pool은 사용자당 최대 context 이상이어야 합니다.')
    if d['ubatch']>d['batch']:raise ValueError('ubatch는 batch 이하여야 합니다.')
    if d['mode'] not in ('gpu','hybrid') or d['kv_location'] not in ('gpu','cpu') or d['kv_type'] not in ('f16','q8_0','q4_0'):raise ValueError('지원하지 않는 모드입니다.')
    if d['host'] not in ('127.0.0.1','0.0.0.0'):raise ValueError('지원하는 호스트: 127.0.0.1 / 0.0.0.0')
    if d['reasoning'] not in ('none','minimal','low','medium','high','xhigh','max'):raise ValueError('reasoning 설정 오류')
    if d['mode']=='gpu':d['cpu_layers']=0;d['expert_cache']=0;d['cpu_share']=0
    if d['cpu_layers']==0 and d['expert_cache']:raise ValueError('Expert 캐시는 CPU Expert가 있어야 합니다.')
    if d['cpu_share'] and not d['expert_cache']:raise ValueError('CPU share는 Expert 캐시가 있어야 합니다.')
    return d,p,q


def selected_gpus(d,all_g):
    ids=d.get('gpu_ids','').strip()
    if not ids:return all_g
    ls=[int(x) for x in ids.split(',')]; by={g['index']:g for g in all_g}
    if len(set(ls))!=len(ls) or any(i not in by for i in ls):raise ValueError('GPU 번호가 존재하지 않거나 중복됩니다.')
    return [by[i] for i in ls]


def estimate(raw,hardware=None,weight_override=None):
    d,p,q=validate(raw); all_g,rt,rf=hardware if hardware is not None else (gpus(),*ram()); gs=selected_gpus(d,all_g); warnings=[]; danger=[]
    f=resolve_model(d); weight=q['gb']*1e9/GIB
    if weight_override is not None:weight=weight_override
    elif f and f.exists():
        try:weight=sum(x.stat().st_size for x in check_shards(f))/GIB
        except ValueError as e:warnings.append(str(e))
    cpu=weight*p.get('expert_fraction_estimate',.9)*d['cpu_layers']/p['layers']
    # K and V per scalar including GGML block scale overhead, no recurrent/checkpoint extras.
    scalar={'f16':2,'q8_0':34/32,'q4_0':18/32}[d['kv_type']]
    full=p['full_layers']*d['pool']; sliding=p['sliding_layers']*min(d['pool'],p['window']*d['parallel'])
    kv=2*p['kv_heads']*p['head_dim']*scalar*(full+sliding)/GIB
    # SWA implementation may allocate full length instead; expose worst case.
    worst=2*p['kv_heads']*p['head_dim']*scalar*p['layers']*d['pool']/GIB
    expert=d['expert_cache']/1024
    gpu=weight-cpu+expert+(kv if d['kv_location']=='gpu' else 0)+d['runtime_reserve']+(2 if d['vision'] else 0)
    # mmap may also keep file pages in RAM, reclaimable; this is mandatory-ish budget, not physical RSS guarantee.
    host=cpu+(kv if d['kv_location']=='cpu' else 0)+d['ram_cache']/1024+d['ram_reserve']
    vfree=sum(g['free'] for g in gs); vtotal=sum(g['total'] for g in gs)
    if not gs:danger.append('NVIDIA GPU 정보를 읽지 못했습니다. CUDA/드라이버 및 GPU 번호를 확인하세요.')
    elif gpu>vfree:danger.append(f'예상 VRAM {gpu:.1f} GiB > 현재 여유 {vfree:.1f} GiB')
    if rf is not None and host>rf:danger.append(f'예상 RAM 예산 {host:.1f} GiB > 현재 여유 {rf:.1f} GiB (RAM 캐시 상한 포함)')
    if gs:
        split=d.get('split','').strip(); ratios=[float(x) for x in split.split(',')] if split else [g['free'] for g in gs]
        if len(ratios)!=len(gs) or any(x<=0 for x in ratios):raise ValueError('tensor split은 GPU 수와 같은 개수의 양수여야 합니다.')
        per=[gpu*x/sum(ratios) for x in ratios]
        if any(x>g['free'] for x,g in zip(per,gs)):danger.append('GPU별 추정 배치가 한 장의 여유 VRAM을 초과합니다.')
    else:per=[];ratios=[]
    if expert>cpu and expert>0:danger.append('Expert 캐시 예산이 CPU 상주 Expert 가중치 추정치를 초과합니다. 캐시를 줄이세요.')
    if d['kv_location']=='gpu' and gpu-kv+worst>vfree and gs:warnings.append('SWA가 전체 컨텍스트 버퍼를 할당하는 구현이라면 VRAM이 부족합니다.')
    if d['kv_location']=='cpu' and rf is not None and host-kv+worst>rf:warnings.append('SWA가 전체 버퍼를 할당하면 RAM이 부족할 수 있습니다.')
    if d['kv_location']=='cpu':warnings.append('활성 KV 및 일부 Attention 연산이 CPU에 남을 수 있어 장문 생성 속도가 크게 떨어질 수 있습니다.')
    if d['expert_cache']:warnings.append('동적 Expert 캐시는 소규모 decode 배치에 유리하며 긴 prefill/동시 요청에서는 효과가 달라집니다.')
    if d['unified'] and d['pool']<d['context']*d['parallel']:warnings.append('공유 pool이 모든 슬롯 최대 길이의 합보다 작습니다. 동시 장문 요청은 제한/실패할 수 있습니다.')
    if d['kv_type']!='f16':warnings.append('양자화 KV 및 Flash Attention의 모델/빌드 호환성을 실행 시 검증해야 합니다.')
    if d['host']=='0.0.0.0' and not d['auth']:danger.append('인증 없이 사내 네트워크에 노출합니다. 승인된 격리망에서만 사용하세요.')
    if d['ram_cache']:warnings.append('RAM prompt cache는 유휴/재사용 캐시입니다. 활성 KV 전체를 대체하지 않습니다. 다중 사용자 격리를 검증하세요.')
    if d['context']>=262144:warnings.append('긴 컨텍스트는 prefill 시간, 추가 상태 및 체크포인트 메모리를 크게 증가시킬 수 있습니다.')
    warnings.append('가중치 배치/Expert 비율 및 SWA는 추정치입니다. 프로젝트 파일, CUDA 버퍼, mmap, GPU별 레이어 불균형에 따라 실측이 달라집니다.')
    return {'weight':weight,'gpu_weight':weight-cpu,'cpu_weight':cpu,'kv':kv,'kv_worst':worst,'gpu':gpu,'host_ram':host,'vfree':vfree,'vtotal':vtotal,'ram_free':rf,'ram_total':rt,'gpu_allocation':per,'ratios':ratios,'gpu_ids':[g['index'] for g in gs],'pool':d['pool'],'warnings':warnings,'danger':danger,'model':str(f or ''),'archive_free':disk(repo_path(d)).get('free')}


def placement_plan(raw):
    d,p,q=validate(raw); hardware=(gpus(),*ram()); base=estimate(d,hardware); candidates=[]
    if not base['gpu_ids']:return {'candidates':[], 'message':'NVIDIA GPU를 확인할 수 없습니다.'}
    # The operator selected model/context/slots. Do not silently alter those.
    # Static resident experts and several dynamic-cache allocations are independent candidates.
    selected=selected_gpus(d,hardware[0])
    caches=[0,8192,16384,24576] if p.get('expert_fraction_estimate',0)>0 else [0]
    for cache in caches:
        layers=range(p['layers']+1) if p.get('expert_fraction_estimate',0)>0 else [0]
        for n in layers:
            if cache and not n:continue
            trial={**d,'mode':'hybrid' if n else 'gpu','cpu_layers':n,'expert_cache':cache,'cpu_share':0.5 if cache else 0}
            e=estimate(trial,hardware,base['weight'])
            # Reserve additional per-device margin beyond the user's runtime reserve.
            memory_dangers=[x for x in e['danger'] if '네트워크' not in x]
            if memory_dangers or any(g['free']-v<2 for g,v in zip(selected,e['gpu_allocation'])):continue
            if e['ram_free'] is None or e['ram_free']-e['host_ram']<4:continue
            candidate={'name':'정적 GPU 우선' if not cache else f'동적 캐시 {cache//1024} GiB',
                       'settings':{k:trial[k] for k in ('mode','cpu_layers','expert_cache','cpu_share')},
                       'gpu':e['gpu'],'ram':e['host_ram'],'gpu_weight':e['gpu_weight'],'kv':e['kv'],'kv_worst':e['kv_worst'],
                       'swa_risk':any('SWA' in x for x in e['warnings'])}
            candidates.append(candidate);break
    return {'candidates':candidates,'message':'추정 예산을 만족하는 후보입니다. 실측 전 속도 순위나 로딩 성공을 보장하지 않습니다.' if candidates else '현재 모델/컨텍스트에는 여유 예산을 만족하는 후보가 없습니다. KV 정밀도, 컨텍스트, RAM 캐시, 양자화 선택을 확인하세요.'}

# Recovery options are argv entries, not shell commands. Paths, network and auth stay managed.
RECOVERY_VALUE = {'--flash-attn': {'on','off','auto'}, '--fit': {'on','off'},
                  '--cache-type-k': {'f16','q8_0','q4_0'}, '--cache-type-v': {'f16','q8_0','q4_0'}}
RECOVERY_SWITCH = {'--no-mmap','--mmap','--no-warmup','--warmup','--no-kv-unified','--kv-unified',
                   '--no-kv-offload','--no-cont-batching','--cont-batching','--jinja','--metrics',
                   '--cache-idle-slots','--swa-full','--no-op-offload','--no-repack','--no-mmproj'}
RECOVERY_REMOVE_VALUE = {'--cache-ram','--kv-unified-per-slot','--chat-template-kwargs','--moe-cache-mib','--moe-cache-cpu-share','--n-cpu-moe','--mmproj'}
RECOVERY_PAIRS = {'--no-mmap':'--mmap','--mmap':'--no-mmap','--no-warmup':'--warmup','--warmup':'--no-warmup',
                  '--no-kv-unified':'--kv-unified','--kv-unified':'--no-kv-unified',
                  '--no-cont-batching':'--cont-batching','--cont-batching':'--no-cont-batching'}

def recovery_options(raw):
    value=raw.get('recovery_options','')
    if not isinstance(value,str) or len(value)>4096:raise ValueError('복구 옵션은 4096자 이하 JSON 텍스트여야 합니다.')
    try: opts=json.loads(value) if value.strip() else {}
    except ValueError as exc:raise ValueError('복구 옵션 JSON 오류: '+str(exc)) from exc
    if not isinstance(opts,dict):raise ValueError('복구 옵션은 JSON 객체여야 합니다.')
    for k,v in opts.items():
        if k in RECOVERY_VALUE:
            if v is not None and (not isinstance(v,str) or v not in RECOVERY_VALUE[k]):raise ValueError('지원하지 않는 복구 값: '+k)
        elif k in RECOVERY_SWITCH:
            if v is not None and v is not True:raise ValueError('스위치 값은 true 또는 null: '+k)
        elif k in RECOVERY_REMOVE_VALUE:
            if v is not None:raise ValueError('이 옵션은 제거(null)만 가능. 값은 일반 설정으로 변경하세요: '+k)
        else:raise ValueError('지원하지 않는 복구 옵션: '+str(k))
    if any(opts.get(k) is True and opts.get(v) is True for k,v in RECOVERY_PAIRS.items()):raise ValueError('서로 반대되는 스위치가 동시에 설정되었습니다.')
    return opts

def apply_recovery(args,opts):
    args=list(args)
    def remove(k):
        while k in args[1:]:
            i=args.index(k,1);count=2 if k in RECOVERY_VALUE or k in RECOVERY_REMOVE_VALUE else 1
            del args[i:i+count]
    for k,v in opts.items():
        remove(k)
        if v is not None:
            if k in RECOVERY_PAIRS:remove(RECOVERY_PAIRS[k])
            args.append(k)
            if v is not True:args.append(v)
    return args

def recommend_setup(raw):
    """Estimate-only wizard; never starts a server."""
    p,q=profile(raw)
    purpose=raw.get('wizard_purpose','coding')
    location=raw.get('wizard_kv','gpu')
    strategy=raw.get('wizard_strategy','auto')
    if purpose not in ('safe','coding','long','team') or location not in ('gpu','cpu') or strategy not in ('auto','gpu','static','dynamic'):
        raise ValueError('간편 설정 선택을 확인하세요.')
    if task_busy('server') or task_busy('tune'):
        raise ValueError('여유 메모리 계산 전 실행/튜닝 중인 서버를 먼저 중지하세요.')
    gs=gpus(); rt,rf=ram()
    if not gs or rf is None:
        raise ValueError('GPU/RAM 자동 감지 실패. 환경 진단을 확인하세요.')
    desired=num({'v':raw.get('wizard_context',32768)},'v',512,p['max_context'])
    slots=num({'v':raw.get('wizard_slots',4)},'v',1,128) if purpose=='team' else 1
    reserve=max(8,min(32,round(sum(g['free'] for g in gs)*0.08)))
    host_reserve=max(8,min(32,round(rf*0.10)))
    d={**raw,'context':desired,'parallel':slots,'unified':False,'pool':desired*slots,
       'mode':'hybrid','cpu_layers':0,'expert_cache':0,'cpu_share':0,'expert_verbose':False,
       'kv_location':location,'kv_type':'f16' if purpose=='safe' or location=='cpu' else ('q8_0' if desired>=131072 else 'f16'),
       'runtime_reserve':reserve,'ram_reserve':host_reserve,'ram_cache':0,'batch':256 if purpose=='safe' else 512,
       'ubatch':128,'threads':max(1,(os.cpu_count() or 2)//2),'vision':bool(raw.get('vision',False)),'gpu_ids':'','split':''}
    if purpose=='safe' and strategy=='auto':strategy='static'
    result=placement_plan(d)
    static=[x for x in result['candidates'] if x['settings']['expert_cache']==0]
    dynamic=[x for x in result['candidates'] if x['settings']['expert_cache']>0]
    if strategy=='gpu': candidates=[x for x in static if x['settings']['cpu_layers']==0]
    elif strategy=='static':candidates=static
    elif strategy=='dynamic':candidates=dynamic
    else:candidates=static or dynamic
    if not candidates:
        return {'ok':False,'message':'현재 선택 조건에 맞는 메모리 배치가 없습니다. 컨텍스트/슬롯, 양자화 또는 KV 위치를 변경하세요. 요청값을 자동으로 낮추지 않습니다.'}
    chosen=candidates[0]
    d.update(chosen['settings'])
    keys=('context','parallel','unified','pool','mode','cpu_layers','expert_cache','cpu_share','expert_verbose','kv_location','kv_type','runtime_reserve','ram_reserve','ram_cache','batch','ubatch','threads','vision','gpu_ids','split')
    return {'ok':True,'settings':{k:d[k] for k in keys},'candidate':chosen,'message':'현재 여유 메모리 기반 추천값입니다. 실측 성공/속도를 보장하지 않습니다. 기존 접속/인증/복구 옵션 유지.'}

def command_preview(raw):
    d,p,q=validate(raw); e=estimate(d); exe=find_server(d); f=resolve_model(d)
    args=[str(exe or 'llama-server.exe'),'-m',str(f or '<GGUF first shard>'),'--alias',str(d['alias']),'-ngl','99','--split-mode','layer','-c',str(d['pool']),'-np',str(d['parallel']),'-b',str(d['batch']),'-ub',str(d['ubatch']),'-t',str(d['threads']),'--host',d['host'],'--port',str(d['port']),'--jinja','--metrics','--cont-batching','--fit','off']
    if e['ratios']:args+=['--tensor-split',','.join(f'{x:.4g}' for x in e['ratios'])]
    if d['unified']:args+=['--kv-unified','--kv-unified-per-slot',str(d['context'])]
    else:args+=['--no-kv-unified']
    if d['kv_location']=='cpu':args+=['--no-kv-offload']
    args+=['--cache-type-k',d['kv_type'],'--cache-type-v',d['kv_type'],'--flash-attn','on','--cache-ram',str(d['ram_cache'])]
    if d['ram_cache'] and d['unified']:args+=['--cache-idle-slots']
    if d['cpu_layers']:args+=['--n-cpu-moe',str(d['cpu_layers'])]
    if d['expert_cache']:
        args+=['--moe-cache-mib',str(d['expert_cache']),'--moe-cache-cpu-share',str(d['cpu_share'])]
        if d.get('expert_verbose'):args+=['-lv','4']
    if p.get('reasoning_template'):args+=['--temp','1.0','--top-p','1.0','--top-k','0','--min-p','0.0','--chat-template-kwargs',json.dumps({'reasoning_effort':d['reasoning']})]
    if d['vision']:
        projector=p.get('projector'); candidates=[repo_path(d)/projector,f.parent/projector,f.parent.parent/projector] if f and projector else ([repo_path(d)/projector] if projector else [])
        proj=next((x for x in candidates if x.is_file()),None)
        if not proj:raise ValueError('Vision projector 파일을 찾지 못했습니다. Vision 다운로드 옵션을 선택하세요.')
        args+=['--mmproj',str(proj)]
    else:args+=['--no-mmproj']
    if d['host']=='0.0.0.0' and not d['auth']:args+=['--no-slots']
    if d['auth']:args+=['--api-key-file',str(KEYS)]
    opts=recovery_options(d)
    if opts:
        args=apply_recovery(args,opts)
        if '--no-mmproj' in args and '--mmproj' in args:
            raise ValueError('Vision 체크를 해제하거나 --mmproj를 제거하세요.')
        e['danger'].append('복구 옵션 적용 중: 메모리 예상값과 GUI 설정이 최종 명령과 다를 수 있습니다. 실제 명령과 메모리를 확인하세요.')
    return d,e,args


def supports(exe,args,env=None):
    rc,h=run_capture([exe,'--help'],30,env=env)
    if rc!=0:raise ValueError('llama-server --help 실패: '+h[-600:])
    missing=[]
    for a in args[1:]:
        if a.startswith('--') and not re.search(r'(?<![\w-])'+re.escape(a)+r'(?![\w-])',h):missing.append(a)
    if missing:raise ValueError('현재 llama.cpp 빌드가 지원하지 않는 옵션: '+', '.join(missing)+'. 최신 빌드로 업데이트하거나 해당 설정을 끄세요.')


def task_busy(kind):
    t=TASKS.get(kind);return bool(t and t['active'])


def step(kind,args,cwd=None,env=None):
    log(kind,'> '+subprocess.list2cmdline([str(x) for x in args]))
    with LOCK:
        t=TASKS[kind]
        if t['cancel']:raise RuntimeError('사용자가 중지했습니다.')
        p=subprocess.Popen([str(x) for x in args],cwd=str(cwd) if cwd else None,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1,creationflags=CREATE)
        t['proc']=p
    try:
        for line in p.stdout:log(kind,line)
        rc=p.wait()
    finally:
        if p.stdout:p.stdout.close()
    with LOCK:TASKS[kind]['proc']=None
    if rc:raise RuntimeError(f'프로세스 종료 코드 {rc}. 로그에서 원인을 확인하세요.')


def files_overlap(download_meta,server_meta):
    # Conservative on missing metadata. Read-only inspect / HF install do not modify GGUFs.
    if download_meta.get('read_only'):return False
    if not download_meta.get('dest') or not download_meta.get('patterns') or not server_meta.get('files'):return True
    base=Path(download_meta['dest']).resolve()
    for f in server_meta['files']:
        try: rel=Path(f).resolve().relative_to(base).as_posix()
        except ValueError:continue
        if any(fnmatch.fnmatchcase(rel.casefold(),pattern.casefold()) for pattern in download_meta['patterns']):return True
    return False


def start_task(kind,func,meta=None):
    with LOCK:
        if task_busy(kind):raise ValueError('이미 실행 중입니다.')
        if (kind=='server' and task_busy('build')) or (kind=='build' and task_busy('server')):raise ValueError('빌드와 서버 실행을 동시에 할 수 없습니다.')
        if kind=='server' and task_busy('download') and files_overlap(TASKS['download']['meta'],meta or {}):raise ValueError('실행할 GGUF 또는 projector가 현재 다운로드 대상과 겹칩니다.')
        if kind=='download' and task_busy('server') and files_overlap(meta or {},TASKS['server']['meta']):raise ValueError('다운로드 대상이 실행 중인 GGUF 또는 projector와 겹칩니다.')
        if (kind=='tune' and any(task_busy(k) for k in ('build','download','server','benchmark'))) or (kind in ('build','download','server','benchmark') and task_busy('tune')):raise ValueError('자동 튜닝은 다른 모델/서버 작업과 동시에 실행할 수 없습니다.')
        LOGS[kind].clear()
        TASKS[kind]={'active':True,'state':'running','started':time.time(),'proc':None,'cancel':False,'meta':meta or {}}
    def worker():
        try:func(); state='stopped' if TASKS[kind]['cancel'] else 'done';log(kind,'완료' if state=='done' else '중지됨')
        except Exception as e:
            state='stopped' if TASKS[kind]['cancel'] else 'error';log(kind,str(e));
            if state=='error' and kind in ('build','server','tune'):
                with LOCK: lines=list(LOGS[kind])
                for line in reversed(lines):
                    hint=build_hint(line)
                    if hint:log(kind,'자동 진단: '+hint);break
            log(kind,'진단 탭과 README의 문제 해결 항목을 확인하세요.')
        finally:
            with LOCK:TASKS[kind].update(active=False,state=state,proc=None)
    threading.Thread(target=worker,daemon=True).start()


def stop_task(kind):
    with LOCK:
        if not task_busy(kind):return
        t=TASKS[kind];t['cancel']=True;p=t['proc']
    if p and p.poll() is None:
        if WIN:run_capture(['taskkill','/PID',str(p.pid),'/T','/F'],10)
        else:p.terminate()


def download(raw,inspect=False):
    d=raw;p,q=profile(d);base=repo_path(d);base.mkdir(parents=True,exist_ok=True)
    patterns=[q['pattern']]+([p['projector']] if d['vision'] and p.get('projector') else [])
    # API (not a version-dependent CLI module entry point); reruns reuse local metadata and partial cache.
    script='''import json,sys\nfrom huggingface_hub import HfApi,snapshot_download\nfrom fnmatch import fnmatch\nfrom pathlib import Path\nimport shutil\na=json.loads(sys.argv[1]); info=HfApi().model_info(a['repo'],revision=a['revision'],files_metadata=True); files=[s for s in info.siblings if any(fnmatch(s.rfilename,g) for g in a['patterns'])]; assert files, 'No files matched the configured pattern'; print('Revision:',info.sha,flush=True); print('\\n'.join(f'{s.rfilename} | {s.size} bytes' for s in files),flush=True); print('Total bytes:',sum(s.size or 0 for s in files),flush=True)\nif not a['inspect']:\n    dst=Path(a['dest']); need=sum(s.size or 0 for s in files if not ((dst/s.rfilename).is_file() and (dst/s.rfilename).stat().st_size==s.size)); free=shutil.disk_usage(dst).free; print('Remaining file budget (conservative):',need,'Free bytes:',free,flush=True)\n    if need>free: raise RuntimeError('Not enough disk space for remaining full files (partial cache not subtracted)')\n    snapshot_download(repo_id=a['repo'],revision=info.sha,local_dir=a['dest'],allow_patterns=a['patterns'],max_workers=2)\n'''
    a=dict(repo=p['repo'],revision=p.get('revision','main'),patterns=patterns,dest=str(base),inspect=inspect)
    def job():
        rc,_=run_capture([sys.executable,'-c','import huggingface_hub'])
        if rc:raise RuntimeError('huggingface_hub 없음. 환경 탭에서 설치하세요.')
        step('download',[sys.executable,'-u','-c',script,json.dumps(a,ensure_ascii=False)])
    start_task('download',job,{'dest':str(base),'patterns':patterns,'read_only':inspect,'profile':d['profile'],'quant':d['quant']})


def install_hf():start_task('download',lambda:step('download',[sys.executable,'-m','pip','install','--upgrade','--only-binary=:all:','huggingface_hub']),{'read_only':True})


def set_task_meta(kind,**items):
    with LOCK:TASKS[kind]['meta'].update(items)


def cuda_env(raw=None):
    env=os.environ.copy()
    # A portable runtime does not require a toolkit. Only use auto-detected one if present.
    try:tk=cuda_toolkit(raw)
    except ValueError:tk=None
    if tk:
        env['CUDA_PATH']=tk['root'];env['PATH']=str(Path(tk['nvcc']).parent)+os.pathsep+env.get('PATH','')
    return env


def release_pair(releases,major,tag=''):
    for r in releases:
        t=r.get('tag_name','')
        if not re.fullmatch(r'b\d+',t) or r.get('draft') or (tag and t!=tag) or (not tag and r.get('prerelease')):continue
        assets={a['name']:a for a in r.get('assets',[]) if 'name' in a}
        for name,asset in assets.items():
            m=re.fullmatch(r'llama-'+re.escape(t)+r'-bin-win-cuda-('+str(major)+r'\.\d+)-x64\.zip',name)
            if not m:continue
            ver=m[1]
            runtime=assets.get(f'cudart-llama-bin-win-cuda-{ver}-x64.zip') or assets.get(f'cudart-llama-{t}-bin-win-cuda-{ver}-x64.zip')
            if runtime:return t,ver,[asset,runtime]
    raise RuntimeError('공식 Windows x64 CUDA 바이너리와 같은 버전의 DLL ZIP 쌍을 찾지 못했습니다. Releases 페이지에서 확인하세요.')


def checked_asset(asset,release,dest):
    url=asset.get('browser_download_url','');name=asset['name']
    if url!=f'https://github.com/ggml-org/llama.cpp/releases/download/{release}/{name}' or not re.fullmatch(r'[\w.-]+\.zip',name):raise RuntimeError('공식 release asset URL/이름 검증 실패')
    size=int(asset.get('size',0)); digest=asset.get('digest') or ''; h=hashlib.sha256(); done=0; next_log=0
    if size<=0:raise RuntimeError('릴리스 파일 크기를 확인하지 못했습니다.')
    req=urllib.request.Request(url,headers={'User-Agent':'EasyLLMManager/0.2'})
    with urllib.request.urlopen(req,timeout=45) as response,dest.open('wb') as f:
        while chunk:=response.read(1024*1024):
            if TASKS['build']['cancel']:raise RuntimeError('다운로드 중지됨')
            f.write(chunk);h.update(chunk);done+=len(chunk)
            if done>size:raise RuntimeError('릴리스 파일 크기 불일치')
            if done>=next_log:log('build',f'{name}: {done/1024**2:.0f} / {size/1024**2:.0f} MiB');next_log=done+32*1024**2
    if done!=size or (digest and digest.lower()!='sha256:'+h.hexdigest()):raise RuntimeError('릴리스 파일 크기 또는 SHA-256 불일치')
    log('build',f'SHA-256 {name}: {h.hexdigest()}'+(' (GitHub digest 확인)' if digest else ' (GitHub digest 미제공)'))


def extract_release(src,dest):
    with zipfile.ZipFile(src) as z:
        if sum(i.file_size for i in z.infolist())>8*GIB:raise RuntimeError('ZIP 압축 해제 용량 한도 초과')
        for info in z.infolist():
            n=info.filename.rstrip('/');parts=n.split('/')
            if not n or '\\' in n or ':' in n or any(x in ('','..','.') for x in parts) or (info.external_attr>>16)&0o170000==0o120000:raise RuntimeError('안전하지 않은 ZIP 경로')
            target=(dest/n).resolve()
            if not target.is_relative_to(dest.resolve()):raise RuntimeError('ZIP 경로 검증 실패')
            if TASKS['build']['cancel']:raise RuntimeError('압축 해제 중지됨')
            z.extract(info,dest)


def install_binary(raw):
    if not WIN:raise ValueError('이 기능은 Windows x64용입니다.')
    major=str(raw.get('runtime_cuda','12'));tag=str(raw.get('runtime_tag','')).strip()
    if major not in ('12','13') or (tag and not re.fullmatch(r'b\d+',tag)):raise ValueError('CUDA 12/13 및 릴리스 태그 b12345 형식만 허용됩니다.')
    w,_,_,_=paths(raw); root=w/'runtime'/'llama.cpp'
    def job():
        root.mkdir(parents=True,exist_ok=True)
        base='https://api.github.com/repos/ggml-org/llama.cpp/releases'
        url=base+('/tags/'+tag if tag else '?per_page=20')
        req=urllib.request.Request(url,headers={'User-Agent':'EasyLLMManager/0.2','Accept':'application/vnd.github+json'})
        with urllib.request.urlopen(req,timeout=30) as response: rs=json.load(response)
        release,ver,assets=release_pair([rs] if isinstance(rs,dict) else rs,major,tag)
        target=root/f'{release}-cuda-{ver}'
        if target.exists():raise RuntimeError('이미 설치된 릴리스 폴더입니다. 기존 서버 경로를 직접 지정하세요: '+str(target))
        need=3*sum(int(x['size']) for x in assets)+512*1024**2
        if shutil.disk_usage(root).free<need:raise RuntimeError('런타임 ZIP 및 압축 해제 공간이 부족합니다.')
        log('build',f'공식 릴리스 {release} / CUDA {ver} (드라이버와 모델 호환성 별도 확인)')
        with tempfile.TemporaryDirectory(prefix='install-',dir=root) as temp:
            stage=Path(temp); payload=stage/'files';payload.mkdir()
            for asset in assets:
                src=stage/asset['name'];checked_asset(asset,release,src);extract_release(src,payload)
            exes=list(payload.rglob('llama-server.exe'))
            if len(exes)!=1:raise RuntimeError('릴리스에서 llama-server.exe를 유일하게 찾지 못했습니다.')
            exe=exes[0]
            for dll in payload.rglob('*.dll'):
                if dll.parent!=exe.parent and dll.name.lower().startswith(('cudart','cublas')) and not (exe.parent/dll.name).exists():shutil.copy2(dll,exe.parent/dll.name)
            step('build',[exe,'--version'],cwd=exe.parent)
            rc,devices=run_capture([exe,'--list-devices'],30)
            log('build','디바이스 검사: '+devices[-3000:])
            if rc or not re.search(r'CUDA\d|NVIDIA',devices,re.I):log('build','GPU 확인 실패/미확인: 드라이버와 CUDA DLL 호환성을 검사하세요. 설치 파일은 유지합니다.')
            if TASKS['build']['cancel']:raise RuntimeError('설치 중지됨')
            rel=exe.relative_to(payload);payload.rename(target)
        set_task_meta('build',exe=str(target/rel));log('build','설치 완료. 빌드 화면의 실행 파일 적용 버튼을 누르세요.')
    start_task('build',job)

def build(raw):
    w,l,m,x=paths(raw); jobs=num({'jobs':raw.get('build_jobs',BASE['build_jobs'])},'jobs',1,128); arch=str(raw.get('cuda_arch') or BASE['cuda_arch']); ref=str(raw.get('llama_ref') or BASE['llama_ref']); update=bool(raw.get('update_source'))
    if not re.fullmatch(r'\d+(?:[a-z])?(?:;\d+(?:[a-z])?)*',arch):raise ValueError('CUDA arch 예: 90, 89, 90;100')
    if ref.startswith('-'):raise ValueError('잘못된 git ref')
    def job():
        git=find_tool('git'); cm=find_tool('cmake'); tk=cuda_toolkit(raw); vs=vs_install()
        if not all((git,cm,tk)) or (WIN and not vs):raise RuntimeError('Git/CMake/CUDA Toolkit/MSVC C++ 환경을 먼저 설치하세요. Toolkit이 없으면 공식 바이너리 설치를 선택할 수 있습니다.')
        nv=tk['nvcc'];log('build',f"CUDA Toolkit {tk['version']} | {tk['root']}")
        l.parent.mkdir(parents=True,exist_ok=True)
        if not (l/'.git').exists():
            if l.exists() and any(l.iterdir()):raise RuntimeError('llama.cpp 경로에 Git 저장소가 아닌 파일이 있습니다.')
            step('build',[git,'clone',BASE['llama_repo'],str(l)])
        elif update:
            rc,out=run_capture([git,'-C',str(l),'status','--porcelain'])
            if rc or out.strip():raise RuntimeError('소스 변경사항이 있어 자동 업데이트를 중단합니다. 직접 확인하세요.')
            step('build',[git,'-C',str(l),'fetch','--tags','origin'])
            if not ref:step('build',[git,'-C',str(l),'pull','--ff-only'])
        if ref:step('build',[git,'-C',str(l),'checkout','--detach',ref])
        env=os.environ.copy();env['PATH']=str(Path(nv).parent)+os.pathsep+env.get('PATH','');env['CUDA_PATH']=tk['root'];env['CUDACXX']=nv
        bdir=l/('build-cuda-'+tk['version']+'-vs'+str(vs[1] if vs else 0)+'-sm'+arch.replace(';','_'))
        if raw.get('fresh_build'):bdir=bdir.with_name(bdir.name+'-'+str(time.time_ns()))
        args=[cm,'-S',str(l),'-B',str(bdir),'-DGGML_CUDA=ON','-DCMAKE_CUDA_ARCHITECTURES='+arch,'-DCUDAToolkit_ROOT='+tk['root'],'-DLLAMA_BUILD_TESTS=OFF','-DCMAKE_BUILD_TYPE=Release']
        if WIN:
            vsroot,major=vs; gen={17:'Visual Studio 17 2022',18:'Visual Studio 18 2026'}.get(major)
            if not gen:raise RuntimeError(f'지원되지 않는 Visual Studio 버전 {major}; CMake 생성기를 직접 확인하세요.')
            integrated=list(vsroot.glob('MSBuild/Microsoft/VC/*/BuildCustomizations/CUDA '+tk['version']+'.props'))
            standalone=Path(tk['root'])/'extras/visual_studio_integration/MSBuildExtensions'/('CUDA '+tk['version']+'.props')
            toolkit= tk['root'] if not integrated and standalone.is_file() else tk['version']
            args+=['-G',gen,'-A','x64','-DCMAKE_GENERATOR_INSTANCE='+str(vsroot),'-T','cuda='+toolkit]
            if not integrated and not standalone.is_file():log('build','Visual Studio CUDA .props를 찾지 못했습니다. CUDA Visual Studio Integration을 확인하세요.')
        log('build','격리된 CMake 빌드 폴더: '+str(bdir))
        step('build',args,env=env);step('build',[cm,'--build',str(bdir),'--config','Release','--parallel',str(jobs),'--target','llama-server'],env=env)
        exe=next((a for a in (bdir/'bin/Release/llama-server.exe',bdir/'bin/llama-server.exe',bdir/'bin/llama-server') if a.is_file()),None)
        if not exe:raise RuntimeError('빌드 후 실행 파일을 찾지 못했습니다.')
        step('build',[str(exe),'--version'],cwd=exe.parent,env=env);set_task_meta('build',exe=str(exe))
        log('build','빌드 완료. 실행 파일 적용 버튼을 누르세요.')
    start_task('build',job)


def launch(raw):
    d,e,args=command_preview(raw)
    if task_busy('benchmark'):raise ValueError('벤치마크 작업이 끝난 뒤 서버를 시작하세요.')
    if e['danger'] and not raw.get('force'):return {'confirm':True,'danger':e['danger']}
    exe=find_server(d);f=resolve_model(d)
    if not exe:raise ValueError('llama-server를 먼저 빌드하거나 실행 파일 경로를 지정하세요.')
    files=[str(a.resolve()) for a in check_shards(f)]
    if '--mmproj' in args:files.append(str(Path(args[args.index('--mmproj')+1]).resolve()))
    if d['auth'] and not read_keys():raise ValueError('API 키를 먼저 생성하세요.')
    if not e['gpu_ids']:raise ValueError('사용할 NVIDIA GPU가 없습니다.')
    with socket.socket() as s:
        try:s.bind((d['host'],d['port']))
        except OSError:raise ValueError('서버 포트가 이미 사용 중이거나 열 수 없습니다.')
    env=cuda_env(d);env['PATH']=str(exe.parent)+os.pathsep+env.get('PATH','')
    supports(str(exe),args,env)
    env['CUDA_VISIBLE_DEVICES']=','.join(map(str,e['gpu_ids']));
    # Do not inherit unexpected API keys if the UI requests no authentication.
    for k in ('LLAMA_API_KEY','LLAMA_ARG_API_KEY_FILE','LLAMA_ARG_MMPROJ','LLAMA_ARG_MMPROJ_URL','LLAMA_ARG_MMPROJ_AUTO'):env.pop(k,None)
    start_task('server',lambda:step('server',args,cwd=exe.parent,env=env),{'port':d['port'],'auth':d['auth'],'alias':d['alias'],'host':d['host'],'context':d['context'],'parallel':d['parallel'],'profile':d['profile'],'quant':d['quant'],'cpu_layers':d['cpu_layers'],'expert_cache':d['expert_cache'],'cpu_share':d['cpu_share'],'kv_type':d['kv_type'],'kv_location':d['kv_location'],'files':files,'recovery_options':d.get('recovery_options','')})
    return {'started':True}


def read_keys():
    try:return [x.strip() for x in KEYS.read_text(encoding='utf-8').splitlines() if x.strip() and not x.startswith('#')]
    except OSError:return []


def key_action(action,index=None):
    if task_busy('server'):raise ValueError('키 변경 전 서버를 중지하세요. 변경 후 다시 시작해야 합니다.')
    with LOCK:
        keys=read_keys(); new=None
        if action=='add':new=secrets.token_urlsafe(32);keys.append(new)
        elif action=='remove':keys.pop(int(index))
        else:raise ValueError('키 작업 오류')
        KEYS.write_text('\n'.join(keys)+'\n',encoding='utf-8')
        if not WIN:KEYS.chmod(0o600)
    return {'new':new,'keys':masked_keys()}


def masked_keys():return [{'index':i,'mask':x[:6]+'…'+x[-4:]} for i,x in enumerate(read_keys())]


def local_api(port,path,auth=False):
    headers={}; ks=read_keys() if auth else []
    if ks:headers['Authorization']='Bearer '+ks[0]
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(urllib.request.Request(f'http://127.0.0.1:{port}{path}',headers=headers),timeout=1.5) as r:return r.read(300000).decode('utf-8','replace')
    except (OSError,urllib.error.URLError):return ''


def cache_stats(lines):
    # Only report actual counters emitted by supported engine builds, never fabricate a rate.
    hits=misses=cpu=samples=0;uploaded=0.0
    for line in lines:
        m=re.search(r'llama_moe_cache: ubatch .*?: hits = (\d+), misses = (\d+)(?: \((\d+) on the CPU\))?, hit rate = [0-9.]+%, uploaded = ([0-9.]+) MiB',line)
        if m:hits+=int(m[1]);misses+=int(m[2]);cpu+=int(m[3] or 0);uploaded+=float(m[4]);samples+=1
    return {'hits':hits,'misses':misses,'cpu':cpu,'uploaded_mib':uploaded,'samples':samples,'hit_rate':100*hits/(hits+misses) if hits+misses else None}


def api_json(port,path,body=None,auth=False,timeout=5):
    headers={'Content-Type':'application/json'}
    if auth:
        keys=read_keys()
        if not keys:raise RuntimeError('API 키가 없습니다.')
        headers['Authorization']='Bearer '+keys[0]
    req=urllib.request.Request(f'http://127.0.0.1:{int(port)}{path}',data=json.dumps(body).encode() if body is not None else None,headers=headers)
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=timeout) as response:
        return json.loads(response.read(4*1024**2))


def bench_counts(raw,ctx):
    prompt=num({'n':raw.get('bench_prompt',512)},'n',32,1048576);out=num({'n':raw.get('bench_output',128)},'n',16,1024)
    if prompt+out+32>ctx:raise ValueError('벤치마크 입력+출력+32 토큰이 실행 중인 슬롯 컨텍스트보다 큽니다.')
    return prompt,out


def measure_server(port,prompt_tokens,out_tokens,auth=False,seed=123):
    # Synthetic sample only; exact prompt token count from server tokenizer.
    sample='The robot recorded the experiment, analyzed the error, and described the next step.\n'
    ids=api_json(port,'/tokenize',{'content':sample,'add_special':False},auth,30).get('tokens',[])
    if not ids or not all(isinstance(x,int) for x in ids):raise RuntimeError('서버 tokenizer 응답이 올바르지 않습니다.')
    prompt=(ids*((prompt_tokens+len(ids)-1)//len(ids)))[:prompt_tokens]
    begin=time.monotonic();response=api_json(port,'/completion',{'prompt':prompt,'n_predict':out_tokens,'ignore_eos':True,'cache_prompt':False,'temperature':0.0,'seed':seed,'stream':False},auth,14400)
    t=response.get('timings',{});result={k:t[k] for k in ('prompt_n','prompt_ms','prompt_per_second','predicted_n','predicted_ms','predicted_per_second') if isinstance(t.get(k),(int,float))}
    if result.get('predicted_n',0)<=0 or result.get('predicted_per_second',0)<=0:raise RuntimeError('서버가 생성 timing 데이터를 반환하지 않았습니다.')
    result['wall_seconds']=round(time.monotonic()-begin,3);result['requested_prompt']=prompt_tokens;result['requested_output']=out_tokens
    return result


def history():
    try:
        v=json.loads(BENCH_HISTORY.read_text(encoding='utf-8'));return v if isinstance(v,list) else []
    except (OSError,ValueError):return []


def record_bench(kind,meta,result):
    safe={k:meta[k] for k in ('profile','quant','context','parallel','cpu_layers','expert_cache','cpu_share','kv_type','kv_location') if k in meta}
    item={'date':time.strftime('%Y-%m-%d %H:%M:%S'),'kind':kind,'settings':safe,**result}
    with LOCK:
        recent=(history()+[item])[-50:];tmp=BENCH_HISTORY.with_suffix('.tmp');tmp.write_text(json.dumps(recent,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(BENCH_HISTORY)
    return item


def benchmark(raw):
    with LOCK:
        t=TASKS.get('server',{});meta=dict(t.get('meta',{}))
        if not t.get('active') or not meta:raise ValueError('관리 중인 서버를 먼저 실행하세요.')
    if api_json(meta['port'],'/health',auth=meta['auth']).get('status')!='ok':raise ValueError('서버가 아직 준비되지 않았습니다.')
    if meta['parallel']!=1:raise ValueError('기존 사용자 캐시 및 부하에 영향을 주므로 단일 슬롯 서버에서만 벤치마크를 수행하세요.')
    n,m=bench_counts(raw,meta['context'])
    def job():
        log('benchmark',f'합성 입력 {n} tokens, 생성 {m} tokens (cache_prompt=false)')
        result=measure_server(meta['port'],n,m,meta['auth']);item=record_bench('manual',meta,result);set_task_meta('benchmark',result=item)
        log('benchmark',json.dumps(result,ensure_ascii=False))
    start_task('benchmark',job)


def tune(raw):
    if any(task_busy(k) for k in ('build','download','server','benchmark','tune')):raise ValueError('다른 작업을 중지한 뒤 튜닝하세요.')
    d,_,_=validate(raw)
    if recovery_options(d):raise ValueError('자동 튜닝 전에 복구 옵션을 비워 주세요.')
    if d['parallel']!=1 or d['unified']:raise ValueError('자동 튜닝은 단일 슬롯/비공유 KV 전용입니다.')
    if d['mode']!='hybrid' and not raw.get('allow_gpu_tune'):raise ValueError('GPU + CPU MoE 모드에서 실행하세요.')
    f=resolve_model(d);check_shards(f);exe=find_server(d)
    if not exe:raise ValueError('llama-server 실행 파일이 필요합니다.')
    n,m=bench_counts(raw,d['context'])
    d={**d,'host':'127.0.0.1','auth':False,'ram_cache':0}
    plan=placement_plan(d);choices=plan['candidates']
    if not choices:raise ValueError(plan['message'])
    # Compare static vs several cache sizes; also test one no-CPU-share candidate.
    variants=[dict(c['settings']) for c in choices]
    if variants[-1]['expert_cache']:variants.append({**variants[-1],'cpu_share':0.0})
    # Keep requested context; no automatic quant/context reduction. Test only synthetic requests.
    def job():
        results=[]; best=None
        for i,variant in enumerate(variants):
            if TASKS['tune']['cancel']:break
            trial={**d,**variant}
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));trial['port']=sock.getsockname()[1]
            _,e,args=command_preview(trial)
            label=f"{i+1}/{len(variants)}: CPU {trial['cpu_layers']}층, cache {trial['expert_cache']} MiB, CPU share {trial['cpu_share']}"
            log('tune',label)
            if e['danger']:
                log('tune','메모리 재검사 실패, 후보 건너뜀: '+'; '.join(e['danger']));continue
            env=cuda_env(d);env['PATH']=str(exe.parent)+os.pathsep+env.get('PATH','');env['CUDA_VISIBLE_DEVICES']=','.join(map(str,e['gpu_ids']))
            for k in ('LLAMA_API_KEY','LLAMA_ARG_API_KEY_FILE','LLAMA_ARG_MMPROJ','LLAMA_ARG_MMPROJ_URL','LLAMA_ARG_MMPROJ_AUTO'):env.pop(k,None)
            try:supports(str(exe),args,env)
            except ValueError as exc:log('tune',str(exc));continue
            proc=None;reader=None
            try:
                with LOCK:
                    if TASKS['tune']['cancel']:break
                    proc=subprocess.Popen(args,cwd=exe.parent,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',creationflags=CREATE);TASKS['tune']['proc']=proc
                def drain():
                    for line in proc.stdout:log('tune',line)
                reader=threading.Thread(target=drain,daemon=True);reader.start()
                deadline=time.monotonic()+3600
                while True:
                    if TASKS['tune']['cancel'] or proc.poll() is not None:raise RuntimeError('서버가 중지되었거나 모델 로딩에 실패했습니다.')
                    if time.monotonic()>deadline:raise RuntimeError('모델 로딩 시간 한도 초과')
                    try:
                        if api_json(trial['port'],'/health').get('status')=='ok':break
                    except (OSError,ValueError):pass
                    time.sleep(2)
                first=measure_server(trial['port'],n,m)
                if TASKS['tune']['cancel']:break
                second=measure_server(trial['port'],n,m,seed=124)
                packed={**second,'cold_decode_tps':first['predicted_per_second']}
                item=record_bench('tune',trial,packed);results.append(item)
                if best is None or item['predicted_per_second']>best['predicted_per_second']:best=item
                set_task_meta('tune',results=results,best=best)
                log('tune',f"측정 완료 | cold {first['predicted_per_second']:.2f} / warm {second['predicted_per_second']:.2f} t/s")
            except Exception as exc:
                log('tune','후보 실패 (다음 안전 후보로 이동): '+str(exc))
            finally:
                if proc and proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                if reader:reader.join(timeout=5)
                if proc and proc.stdout:proc.stdout.close()
                with LOCK:TASKS['tune']['proc']=None
            if not TASKS['tune']['cancel']:time.sleep(2)
        if not best and not TASKS['tune']['cancel']:raise RuntimeError('성공한 튜닝 후보가 없습니다. 로그를 확인하세요.')
        if best:log('tune','추천은 테스트한 후보와 입력 길이 기준이며 사용자가 적용하기 전에는 설정을 변경하지 않습니다.')
    start_task('tune',job)

def diagnostic_command(raw):
    cmd=str(raw.get('diagnostic','')).strip()
    choices={'nvidia-smi':('nvidia-smi',[]),'nvidia-smi -q':('nvidia-smi',['-q']),
             'nvcc --version':('nvcc',['--version']), 'llama-server --version':('server',['--version']),
             'llama-server --help':('server',['--help']), 'llama-server --list-devices':('server',['--list-devices'])}
    if cmd not in choices:raise ValueError('허용된 진단 명령만 입력하세요: '+', '.join(choices))
    tool,flags=choices[cmd]
    if tool=='server':exe=find_server(raw)
    elif tool=='nvcc':
        tk=cuda_toolkit(raw);exe=tk['nvcc'] if tk else None
    else:exe=find_tool(tool)
    if not exe:raise ValueError('실행 파일을 찾지 못했습니다: '+tool)
    env=cuda_env(raw);env['PATH']=str(Path(exe).parent)+os.pathsep+env.get('PATH','')
    rc,out=run_capture([str(exe),*flags],30,env=env)
    return {'code':rc,'output':out[-120000:],'command':subprocess.list2cmdline([str(exe),*flags])}

def error_excerpt(lines):
    bad=[]
    for ln in lines:
        if re.search(r'error|failed|fatal|assert|exception|out of memory|invalid|자동 진단|종료 코드',ln,re.I) and not ln.startswith('> '):bad.append(ln)
    return bad[-15:] if bad else lines[-8:]

def status():
    with LOCK:
        ts={k:{x:v for x,v in t.items() if x not in ('proc','cancel')}|{'pid':t['proc'].pid if t['proc'] else None} for k,t in TASKS.items()}; logs={k:list(v) for k,v in LOGS.items()}
    t=ts.get('server',{}); meta=t.get('meta',{}); health='';metrics='';slots=''
    if t.get('active') and meta:
        health=local_api(meta['port'],'/health');
        if health:
            metrics=local_api(meta['port'],'/metrics',meta['auth']);slots=local_api(meta['port'],'/slots',meta['auth'])
    rt,rf=ram()
    # slots can contain prompt internals depending on build; only report selected counters.
    slim=[]
    try:
        for s in json.loads(slots):
            if isinstance(s,dict):slim.append({k:s[k] for k in ('id','is_processing','n_ctx','n_past','n_decoded','n_remaining') if k in s})
    except (ValueError,TypeError):pass
    return {'tasks':ts,'logs':logs,'gpu':gpus(),'ram_total':rt,'ram_free':rf,'health':health,'metrics':metrics,'slots':slim,'cache_stats':cache_stats(logs['server']),'history':history()[-10:],'server_errors':error_excerpt(logs['server']) if t.get('state')=='error' else []}


def save_prefs(raw):
    global SETTINGS
    allowed=set(BASE['defaults'])|{'workspace','llama_dir','model_dir','server_exe','build_jobs','cuda_arch','llama_ref','cuda_root','runtime_cuda','runtime_tag'}
    prefs={k:v for k,v in raw.items() if k in allowed};validate({**BASE['defaults'],**prefs});recovery_options(prefs);
    with LOCK:
        tmp=PREF.with_suffix('.tmp');tmp.write_text(json.dumps(prefs,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(PREF);SETTINGS=prefs
    return {'ok':True}

HTML = r'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Inkling Manager</title><style>
:root{color-scheme:dark;--bg:#10151d;--panel:#19222e;--line:#344253;--fg:#edf2f7;--muted:#a5b5c7;--accent:#68d6b4;--warn:#f4be72}*{box-sizing:border-box}body{background:var(--bg);color:var(--fg);font:14px/1.55 'Segoe UI',system-ui,sans-serif;margin:0}header{padding:22px max(24px,calc((100vw - 1260px)/2));background:#151e29;border-bottom:1px solid var(--line)}h1{margin:0;font-size:25px;letter-spacing:-.5px}h2{font-size:18px;margin:0 0 14px}h3{font-size:15px;margin:15px 0 8px}p{margin:8px 0;color:var(--muted)}main{max-width:1260px;margin:auto;padding:18px 24px 50px}nav{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 18px}button,a.btn{border:1px solid var(--line);border-radius:7px;background:#253445;color:var(--fg);padding:9px 13px;font:inherit;cursor:pointer;text-decoration:none}button:hover{border-color:var(--accent)}button.active,button.primary{background:var(--accent);color:#0b211b;border-color:var(--accent);font-weight:650}button:disabled{opacity:.4;cursor:not-allowed}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(235px,1fr));gap:12px}.cols{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);gap:14px}.panel{border:1px solid var(--line);background:var(--panel);border-radius:10px;padding:18px;margin-bottom:14px}.field{display:flex;flex-direction:column;gap:5px;margin:0 0 9px;min-width:0}.field>span{color:#c4d0df;font-size:13px}.field small{color:var(--muted)}input,select{width:100%;padding:9px 10px;background:#111923;color:var(--fg);border:1px solid var(--line);border-radius:6px;font:inherit;min-width:0}input[type=checkbox]{width:auto;margin-right:8px}label.check{display:flex;align-items:center;padding:8px 0}input:focus,select:focus{outline:1px solid var(--accent)}.buttons{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}.good{color:var(--accent)}.bad{color:#ff9992}.warn{color:var(--warn)}.muted{color:var(--muted)}.metric{background:#121b25;padding:13px;border-radius:8px;border:1px solid #2b3949}.metric b{display:block;font-size:23px;line-height:1.4;font-variant-numeric:tabular-nums}.bar{height:9px;background:#354254;border-radius:6px;overflow:hidden;margin:7px 0}.bar i{display:block;height:100%;background:var(--accent)}pre{background:#0c131c;border-radius:7px;padding:13px;white-space:pre-wrap;overflow-wrap:anywhere;max-height:360px;overflow:auto;font:12px/1.55 Consolas,monospace;margin:8px 0}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid var(--line);padding:8px;text-align:left;overflow-wrap:anywhere}th{color:var(--muted)}a{color:#9cc9ff}.hidden{display:none!important}#toast{position:fixed;bottom:18px;right:18px;max-width:min(600px,95vw);background:#2c4252;border:1px solid #597a8d;border-radius:8px;padding:12px 16px;z-index:9;white-space:pre-wrap}#toast:empty{display:none}.lead{font-size:13px;color:var(--muted)}@media(max-width:850px){.cols{grid-template-columns:1fr}main{padding:12px}.panel{padding:13px}} </style></head><body>
<header><h1>Easy LLM Manager v0.2.2 <span class="lead"> / Windows local inference</span></h1><p>모델 다운로드부터 빌드, GPU 배치, API 서버 운영까지. 관리 화면은 이 PC의 localhost에서만 접근 가능.</p></header><main><nav id="nav"></nav><div class="grid" id="summary"></div>
<section id="env" class="tab"><div class="cols"><div class="panel"><h2>환경 진단</h2><p>먼저 GPU와 필수 도구를 확인하세요. 설치 직후 PATH가 갱신되지 않았다면 이 도구를 다시 실행하세요.</p><div class="buttons"><button class="primary" onclick="diagnose()">환경 다시 검사</button><button onclick="act('install_hf',{})">HF 다운로드 도구 설치</button></div><div id="diagnosis"></div></div><div class="panel"><h2>저장 경로</h2><p>기존 AI 폴더 또는 여유 공간이 큰 고정 디스크를 탐색합니다. 필요하면 직접 수정하세요.</p><div id="pathFields"></div><div class="buttons"><button onclick="save()">경로/세팅 저장</button></div><p>모델, 빌드 결과는 작업 폴더에 저장합니다. 개인 설정과 API 키는 사용자 전용 데이터 폴더에 저장합니다.</p></div></div></section>
<section id="download" class="tab hidden"><div class="cols"><div class="panel"><h2>모델 다운로드</h2><p>다운로드 선택은 서버 설정과 독립적입니다. 실행 중인 동일 파일은 보호됩니다.</p><div id="modelFields"></div><div id="modelSpec"></div><div class="buttons"><button onclick="act('inspect',downloadForm())">원격 파일/실제 용량 조회</button><button class="primary" onclick="downloadModel()">다운로드 / 이어받기</button><button onclick="stop('download')">중지</button></div><p>같은 repo·revision·폴더로 다시 실행하면 완료 파일 및 HF가 보존한 부분 캐시를 재사용합니다. 파일을 임의로 이동하지 마세요.</p></div><div class="panel"><h2>다운로드 로그</h2><pre id="downloadLog"></pre><p>서버 구동 전 첫 shard와 모든 분할 파일의 존재/헤더를 검사합니다.</p></div></div></section>
<section id="build" class="tab hidden"><div class="cols"><div class="panel"><h2>llama.cpp CUDA 빌드</h2><p>기존 소스를 재사용하고 없다면 공식 Git 저장소에서 clone합니다. Visual Studio C++ 빌드 도구와 CUDA Toolkit이 필요합니다.</p><div id="buildFields"></div><h3>공식 사전 컴파일 바이너리 (컴파일 대안)</h3><div id="runtimeFields"></div><div class="buttons"><button onclick="installBinary()">공식 CUDA 바이너리 설치</button><button onclick="useInstalled()">설치/빌드 결과 실행 파일 적용</button></div><p>GitHub 공식 릴리스만 사용합니다. CUDA 12/13은 드라이버와 호환되는 버전을 선택하세요.</p><label class="check"><input type="checkbox" id="fresh_build">새 빌드 폴더 사용 (기존 파일 유지)</label><label class="check"><input type="checkbox" id="update_source">기존 소스 업데이트 (변경사항이 없을 때만)</label><div class="buttons"><button class="primary" onclick="act('build')">컴파일 시작</button><button onclick="stop('build')">중지</button></div><p>H100은 CUDA arch 90. Git ref를 지정하면 해당 버전으로 빌드합니다. 최초 빌드는 오래 걸릴 수 있습니다.</p></div><div class="panel"><h2>빌드 로그</h2><pre id="buildLog"></pre><p>오류 시 마지막 로그와 환경 진단을 확인하세요. 기존 CMake 생성기와 충돌한다면 소스를 보존하고 build 폴더만 이름을 변경한 후 다시 빌드하세요.</p></div></div></section>
<section id="run" class="tab hidden"><div class="cols"><div class="panel"><h2>추론 서버 설정</h2><div id="runModel"></div><div id="wizard"></div><details id="advancedDetails"><summary style="cursor:pointer;padding:12px 0;font-weight:650">고급 설정 · 모든 숫자 직접 변경</summary><div class="buttons"><button onclick="preset('safe')">첫 구동 (2 GPU)</button><button onclick="preset('long')">1M / GPU KV</button><button onclick="preset('host')">1M / RAM KV</button><button onclick="preset('team')">15인 공유 KV</button><button onclick="preset('q4long')">Q4 1M / 2 GPU</button></div><p>프리셋은 출발점일 뿐이며 GPU 장수와 선택 모델에 따라 조정하세요. 먼저 예상치를 계산하세요.</p><h3>모델 · 배치</h3><div class="grid" id="placement"></div><h3>컨텍스트 · 캐시</h3><div class="grid" id="contexts"></div><h3>엔진 · 네트워크</h3><div class="grid" id="advanced"></div><h3>오류 복구 옵션 (고급)</h3><p>실행 명령에 옵션을 추가/제거/변경합니다. 예: --flash-attn off 또는 --no-mmap. null은 제거, true는 스위치 추가입니다. 일반 설정에서 바꿀 수 있는 값은 그곳에서 변경하세요.</p><textarea id="recovery" rows="4" style="width:100%;background:#111923;color:inherit;padding:9px" placeholder='{"--flash-attn":"off","--no-warmup":true,"--no-kv-unified":null}'></textarea><div class="buttons"><button onclick="clearRecovery()">복구 옵션 초기화</button><button onclick="preset('rescue')">8K 안전 구동</button></div></details><div class="buttons"><button onclick="save()">세팅 저장</button><button onclick="preview()">예상 자원 / 명령어</button><button onclick="plan()">자동 메모리 배치 추천</button><button class="primary" onclick="startServer()">서버 시작</button><button onclick="stop('server')">서버 중지</button></div></div><div><div class="panel"><h2>실행 전 메모리 예측</h2><div id="estimate"><p>설정을 선택한 후 예상 자원을 계산하세요.</p></div><p>GiB 단위. 실제 llama.cpp가 할당하는 메모리는 다를 수 있습니다.</p><h3>자동 배치 후보</h3><div id="plans"><p>선택한 모델/컨텍스트를 유지한 채 후보를 계산합니다.</p></div><h3>생성 명령어</h3><pre id="command"></pre></div><div class="panel"><h2>접속</h2><div id="connection"></div><p>로컬 브라우저에서 먼저 테스트한 후 사내 IP를 사용하세요. Claude Code 확장은 Anthropic Messages API 호환 여부를 별도 확인하세요.</p></div></div></div></section>
<section id="monitor" class="tab hidden"><div class="panel"><h2>실시간 GPU · 서버 상태</h2><div id="gpuCards" class="grid"></div><div id="taskCards" class="grid"></div><div class="buttons"><button onclick="stop('server')">관리 중인 서버 중지</button><button onclick="poll()">즉시 갱신</button></div><div id="health"></div></div><div class="cols"><div class="panel"><h2>서버 로그 (최근 900줄)</h2><h3>최근 실행 요청 오류</h3><pre id="actionError">—</pre><h3>실패 로그 핵심 (자동 추출)</h3><pre id="serverErrors">—</pre><h3>로컬 진단 명령</h3><p>서버 외부로 전송하지 않습니다. 허용된 진단 명령만 실행합니다.</p><input id="diagCommand" list="diagList" value="llama-server --version"><datalist id="diagList"><option value="llama-server --version"><option value="llama-server --help"><option value="llama-server --list-devices"><option value="nvidia-smi"><option value="nvidia-smi -q"><option value="nvcc --version"></datalist><div class="buttons"><button onclick="runDiagnostic()">진단 실행</button></div><pre id="diagOutput"></pre><label class="check"><input type="checkbox" id="follow" checked>자동 스크롤</label><pre id="serverLog" style="max-height:560px"></pre></div><div class="panel"><h2>API /metrics · /slots</h2><div id="rates" class="grid"></div><pre id="metrics"></pre><pre id="slots"></pre></div></div></section>
<section id="tuning" class="tab hidden"><div class="panel"><h2>실측 벤치마크 · 자동 튜닝</h2><p>합성 토큰만 사용합니다. 단일 슬롯에서 테스트하세요. 자동 튜닝은 서버를 중지한 상태에서 여러 후보를 직접 로딩하고 실패 시 다음 후보로 이동합니다. 1M 설정은 매우 오래 걸릴 수 있습니다. 입력 길이를 짧게 시작하세요.</p><div class="grid"><label class="field"><span>입력 토큰 수</span><input type="number" id="bench_prompt" value="512" min="32" max="1048576"></label><label class="field"><span>생성 토큰 수</span><input type="number" id="bench_output" value="128" min="16" max="1024"></label></div><div class="buttons"><button onclick="runBench(false)">실행 중 서버 벤치마크</button><button class="primary" onclick="runBench(true)">자동 튜닝 (서버 중지 후)</button><button onclick="stop('tune')">튜닝 중지</button><button onclick="applyBest()">가장 빠른 실측 후보 적용</button></div><div id="tuneBest"></div><h3>최근 측정 (최대 10개)</h3><div id="history"></div><h3>벤치마크 로그</h3><pre id="benchmarkLog"></pre><h3>튜닝 로그</h3><pre id="tuneLog"></pre></div></section><section id="auth" class="tab hidden"><div class="panel"><h2>API 접근 키</h2><p>기본값은 인증 없음 + localhost. 사내 네트워크에서 서비스할 때만 승인된 범위에 노출하고 API 키 사용을 권장합니다. 키 활성화는 실행 설정에서 별도로 선택하세요. 키 추가/회수 후 재시작이 필요합니다.</p><div class="buttons"><button class="primary" onclick="key('add')">새 API 키 생성</button></div><div id="newKey"></div><div id="keyList"></div><p>키는 사용자 전용 디렉터리의 api-keys.txt에 저장되며 Git에 포함되지 않습니다. 사용자별 요청량 할당/회계는 제공하지 않습니다.</p></div></section></main><div id="toast" role="status"></div>
<script>
const csrf=__TOKEN__; let config, form={}, prev=null, polling=false, current='env', plans=[], dl={}; const $=x=>document.getElementById(x); const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); const fmt=x=>x==null?'—':Number(x).toFixed(1); let toastTimer;
function toast(x){$('toast').textContent=x;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').textContent='',10000)}
async function api(path,body){const opt={headers:{'X-Manager-Token':csrf}};if(body!==undefined){opt.method='POST';opt.headers['Content-Type']='application/json';opt.body=JSON.stringify(body)} const r=await fetch('/api/'+path,opt);let j=await r.json();if(!r.ok)throw Error(j.error||r.status);return j}
const tabs=[['env','1 환경'],['download','2 다운로드'],['build','3 컴파일'],['run','4 서버 실행'],['monitor','5 모니터링'],['tuning','6 성능 튜닝'],['auth','7 API 키']]; $('nav').innerHTML=tabs.map(([k,t])=>`<button id="nav_${k}" onclick="tab('${k}')">${t}</button>`).join('');function tab(k){current=k;for(const [x] of tabs){$(x).classList.toggle('hidden',x!==k);$('nav_'+x).classList.toggle('active',x===k)}if(k==='auth')keys();if(k==='run')connection()}
function field(k,label,type='text',choices=null,help=''){let v=form[k]??'';let inner=type==='check'?`<label class="check"><input data-k="${k}" type="checkbox" ${v?'checked':''}>${esc(label)}</label>`:type==='select'?`<label class="field"><span>${esc(label)}</span><select data-k="${k}">${choices.map(([x,t])=>`<option value="${esc(x)}" ${String(x)===String(v)?'selected':''}>${esc(t)}</option>`).join('')}</select>`:`<label class="field"><span>${esc(label)}</span><input data-k="${k}" type="${type}" ${type==='number'?'step="any"':''} value="${esc(v)}">`;return inner+(type==='check'?'':(help?`<small>${esc(help)}</small>`:'')+'</label>')}
function wire(){document.querySelectorAll('[data-k]').forEach(el=>el.onchange=()=>{const k=el.dataset.k;form[k]=el.type==='checkbox'?el.checked:el.type==='number'?Number(el.value):el.value;plans=[];if($('plans'))$('plans').textContent='설정 변경됨. 자동 배치 후보를 다시 계산하세요.';if(k==='profile'){form.quant=Object.keys(config.model_profiles[form.profile].quants)[0];form.model_path='';render()}else if(k==='quant'){form.model_path='';render()}else{document.querySelectorAll(`[data-k="${k}"]`).forEach(x=>{if(x!==el){if(x.type==='checkbox')x.checked=form[k];else x.value=form[k]}});connection()}})}
function downloadForm(){return {...form,...dl,model_path:''}}
function renderDownload(){const p=config.model_profiles[dl.profile],q=p.quants[dl.quant];$('modelFields').innerHTML=`<label class="field"><span>다운로드 모델 계열</span><select id="dlProfile">${Object.entries(config.model_profiles).map(([k,v])=>`<option value="${esc(k)}" ${k===dl.profile?'selected':''}>${esc(v.label)}</option>`).join('')}</select></label><label class="field"><span>다운로드 양자화</span><select id="dlQuant">${Object.entries(p.quants).map(([k,v])=>`<option value="${esc(k)}" ${k===dl.quant?'selected':''}>${esc(k)} · ${v.gb} GB</option>`).join('')}</select></label><label class="check"><input type="checkbox" id="dlVision" ${dl.vision?'checked':''}>Vision projector 포함</label>`;$('dlProfile').onchange=()=>{dl.profile=$('dlProfile').value;dl.quant=Object.keys(config.model_profiles[dl.profile].quants)[0];renderDownload()};$('dlQuant').onchange=()=>{dl.quant=$('dlQuant').value;renderDownload()};$('dlVision').onchange=()=>{dl.vision=$('dlVision').checked};$('modelSpec').innerHTML=`<div class="grid">${metric('파일 용량',q.gb+' GB',fmt(q.gb*1e9/1073741824)+' GiB')}${metric('원본 top-token 일치율',q.agreement==null?'—':q.agreement+'%','품질 벤치마크 점수 아님')}</div>`}
function clearRecovery(){form.recovery_options='';$('recovery').value='';preview()}
async function runDiagnostic(){await guard(async()=>{const r=await api('diagnostic_command',{...form,diagnostic:$('diagCommand').value});$('diagOutput').textContent=r.command+'\nexit '+r.code+'\n'+r.output})}
function models(){let p=config.model_profiles[form.profile];return field('profile','모델 계열','select',Object.entries(config.model_profiles).map(([k,v])=>[k,v.label]))+field('quant','양자화','select',Object.entries(p.quants).map(([k,v])=>[k,`${k} · ${v.gb} GB`]))+field('vision','Vision projector 포함','check')}
let wizardPrefs=null,wizardBusy=false;
function renderWizard(){
 const p=config.model_profiles[form.profile];
 if(!wizardPrefs)wizardPrefs={purpose:form.parallel>1?'team':'coding',kv:form.kv_location||'gpu',strategy:'auto',context:form.context||32768,slots:Math.max(2,form.parallel||4)};
 const contexts=[8192,16384,32768,65536,131072,262144,524288,1048576,p.max_context,Number(form.context)].filter(x=>x>=512&&x<=p.max_context).sort((a,b)=>a-b).filter((x,i,a)=>i===0||x!==a[i-1]);
 if(!contexts.includes(wizardPrefs.context))wizardPrefs.context=contexts.filter(x=>x<=wizardPrefs.context).pop()||contexts[0];
 const select=(id,caption,opts,value)=>`<label class="field"><span>${caption}</span><select id="${id}">${opts.map(([v,t])=>`<option value="${esc(v)}" ${String(v)===String(value)?'selected':''}>${esc(t)}</option>`).join('')}</select></label>`;
 const slotOptions=[2,4,8,15,32,wizardPrefs.slots].filter((x,i,a)=>a.indexOf(x)===i).sort((a,b)=>a-b);
 $('wizard').innerHTML=`<p>② 사용 목적</p>${select('wPurpose','사용 목적',[['safe','첫 실행 · 오류 확인'],['coding','개인 코딩 / 일반 대화'],['long','장문 연구 / 1M 목표'],['team','팀 공유 / 여러 사용자']],wizardPrefs.purpose)}<p>③ 컨텍스트 및 배치</p><div class="grid">${select('wContext','슬롯당 최대 컨텍스트',contexts.map(x=>[x,x>=1048576&&x%1048576===0?(x/1048576)+'M':x%1024===0?(x/1024)+'K':x+' tokens']),wizardPrefs.context)}${wizardPrefs.purpose==='team'?select('wSlots','동시 사용자 수',slotOptions.map(x=>[x,x+'명']),wizardPrefs.slots):''}${select('wKV','활성 KV 위치',[['gpu','GPU · 빠른 응답'],['cpu','RAM · VRAM 절약']],wizardPrefs.kv)}${select('wStrategy','가중치 배치',[['auto','자동 추천 (정적 배치 우선)'],['gpu','GPU 전체 상주만'],['static','정적 CPU Expert 분산'],['dynamic','동적 Expert 캐시 실험']],wizardPrefs.strategy)}</div><div class="buttons"><button class="primary" id="wizardApply" onclick="applyWizard()">하드웨어 기반 추천값 적용</button><button onclick="wizardStart()">추천값으로 실행</button></div><p id="wizardMessage">모델, 사용 목적, 컨텍스트를 선택하면 나머지 숫자를 자동 계산합니다. 추천값 적용 전에는 기존 실행 설정을 유지합니다.</p>`;
 const bind=(id,key,convert=x=>x)=>{let el=$(id);if(el)el.onchange=()=>{wizardPrefs[key]=convert(el.value);if(key==='purpose'){wizardPrefs.context=Math.min(p.max_context,{safe:8192,coding:32768,long:1048576,team:32768}[el.value]);wizardPrefs.strategy='auto';renderWizard()}else $('wizardMessage').textContent='선택 변경됨. 추천값 적용을 눌러주세요.'}};
 bind('wPurpose','purpose');bind('wContext','context',Number);bind('wSlots','slots',Number);bind('wKV','kv');bind('wStrategy','strategy');
 $('wizardApply').disabled=wizardBusy;
}
async function applyWizard(){
 if(wizardBusy)return;wizardBusy=true;$('wizardApply').disabled=true;$('wizardMessage').textContent='GPU/RAM 여유 공간으로 계산 중…';
 const signature=JSON.stringify([form,wizardPrefs]);
 try{let r=await api('recommend',{...form,wizard_purpose:wizardPrefs.purpose,wizard_kv:wizardPrefs.kv,wizard_strategy:wizardPrefs.strategy,wizard_context:wizardPrefs.context,wizard_slots:wizardPrefs.slots});
 if(signature!==JSON.stringify([form,wizardPrefs]))throw Error('계산 중 설정 변경됨. 다시 계산하세요.');
 if(!r.ok){$('wizardMessage').textContent=r.message;return false}
 Object.assign(form,r.settings);plans=[];$('plans').textContent='추천값 적용됨. 필요하면 추가 후보를 계산하세요.';render();
 $('wizardMessage').textContent=r.message+' CPU '+form.cpu_layers+'층 / Expert '+form.expert_cache+' MiB / '+form.context.toLocaleString()+' tokens / 여유 GPU '+form.runtime_reserve+' GiB / RAM '+form.ram_reserve+' GiB';return !!(await preview());
 }catch(e){$('wizardMessage').textContent=e.message;toast(e.message);return false}finally{wizardBusy=false;$('wizardApply').disabled=false}
}
async function wizardStart(){if(await applyWizard())await startServer()}
function render(){renderWizard();let p=config.model_profiles[form.profile],q=p.quants[form.quant];$('pathFields').innerHTML=field('workspace','작업 폴더')+field('llama_dir','llama.cpp 소스 폴더','text',null,'비워 두면 작업폴더/llama.cpp')+field('model_dir','모델 폴더','text',null,'비워 두면 작업폴더/models')+field('server_exe','기존 llama-server.exe 직접 지정','text',null,'비워 두면 자동 탐색');renderDownload();$('runModel').innerHTML=models()+field('model_path','첫 GGUF 파일 직접 지정 (선택)','text',null,'자동 감지에 실패했거나 기존 다운로드를 사용하려면 지정');$('runtimeFields').innerHTML=field('runtime_cuda','공식 바이너리 CUDA','select',[['12','CUDA 12 (구형 드라이버 우선 확인)'],['13','CUDA 13 (신형 드라이버)']])+field('runtime_tag','공식 릴리스 태그 (선택)','text',null,'비워 두면 최근 안정 릴리스 중 DLL 쌍 탐색');$('buildFields').innerHTML=field('cuda_root','CUDA Toolkit 폴더 (선택)','text',null,'비워 두면 설치된 최신 nvcc 탐색; v12.2 등 전체 경로 지정 가능')+field('cuda_arch','CUDA Architecture')+field('build_jobs','병렬 빌드 작업 수','number')+field('llama_ref','Git ref / 태그 / 커밋 (선택)','text',null,'비워 두면 현재 소스 유지, 최초 clone은 기본 브랜치');$('placement').innerHTML=field('mode','가중치 모드','select',[['gpu','GPU 전체 상주'],['hybrid','GPU + CPU MoE']])+field('cpu_layers','CPU MoE 레이어 수','number',null,'앞 N개 레이어 Expert만 RAM에 저장')+field('expert_cache','동적 GPU Expert cache (MiB)','number',null,'0 = 끔 / 최신 빌드 기능')+field('cpu_share','캐시 miss CPU 연산 비율 (0~1)','number')+field('expert_verbose','Expert 캐시 상세 로그 (디버그)','check')+field('gpu_ids','GPU 번호 (예: 0,1)','text',null,'비워 두면 모든 GPU')+field('split','GPU 배치 비율 (예: 1,1)','text',null,'비워 두면 여유 VRAM 비례');$('contexts').innerHTML=field('context','슬롯당 최대 컨텍스트','number')+field('parallel','동시 슬롯 수','number')+field('unified','공유 KV 풀 사용','check')+field('pool','전체 공유 KV 풀 토큰','number',null,'공유 모드에서만 사용')+field('kv_location','활성 KV 위치','select',[['gpu','GPU (권장)'],['cpu','RAM / CPU (느릴 수 있음)']])+field('kv_type','KV 정밀도','select',[['f16','FP16 (호환성 우선)'],['q8_0','Q8_0'],['q4_0','Q4_0 (실험)']])+field('ram_cache','RAM prompt cache 상한 (MiB)','number',null,'0 = 끔; 유휴 슬롯/프롬프트 보관')+field('runtime_reserve','GPU 기타 여유 공간 (GiB)','number')+field('ram_reserve','RAM 기타 여유 공간 (GiB)','number');$('advanced').innerHTML=field('batch','logical batch','number')+field('ubatch','physical ubatch','number')+field('threads','CPU 연산 스레드','number')+field('reasoning','Reasoning effort','select',['none','minimal','low','medium','high','xhigh','max'].map(x=>[x,x]))+field('alias','API 모델 이름')+field('host','접속 범위','select',[['127.0.0.1','이 서버에서만'],['0.0.0.0','사내망 접속 (승인 필요)']])+field('port','API 포트','number')+field('auth','API 키 인증 사용','check');$('recovery').value=form.recovery_options||'';$('recovery').oninput=()=>{form.recovery_options=$('recovery').value;plans=[]};wire();connection()}
function preset(k){wizardPrefs=null;plans=[];$('plans').textContent='프리셋 변경됨. 자동 배치 후보를 다시 계산하세요.';if(k==='rescue'){let cpu={'UD-IQ1_S':8,'UD-IQ1_M':10,'UD-IQ2_XXS':12,'UD-Q2_K_XL':14,'UD-IQ3_XXS':18,'UD-Q3_K_XL':24,'UD-Q4_K_XL':32}[form.quant]??Math.min(8,config.model_profiles[form.profile].layers);Object.assign(form,{mode:'hybrid',cpu_layers:form.profile==='inkling2'?cpu:Math.min(cpu,config.model_profiles[form.profile].layers),expert_cache:0,cpu_share:0,expert_verbose:false,context:8192,pool:8192,parallel:1,unified:false,kv_location:'gpu',kv_type:'f16',ram_cache:0,batch:256,ubatch:128,recovery_options:''})}if(k==='safe')Object.assign(form,{mode:'hybrid',cpu_layers:8,expert_cache:0,cpu_share:0,context:32768,pool:32768,parallel:1,unified:false,kv_location:'gpu',kv_type:'f16',ram_cache:0,batch:512,ubatch:128});if(k==='long')Object.assign(form,{context:1048576,pool:1048576,parallel:1,unified:false,kv_location:'gpu',kv_type:'q8_0',ram_cache:0});if(k==='host')Object.assign(form,{mode:'gpu',context:1048576,pool:1048576,parallel:1,unified:false,kv_location:'cpu',kv_type:'f16',ram_cache:0});if(k==='team')Object.assign(form,{mode:'gpu',parallel:15,context:1048576,pool:1572864,unified:true,kv_location:'gpu',kv_type:'q8_0',ram_cache:65536});if(k==='q4long'){Object.assign(form,{profile:'inkling2',quant:'UD-Q4_K_XL',model_path:'',mode:'hybrid',cpu_layers:32,expert_cache:8192,cpu_share:0.5,context:1048576,pool:1048576,parallel:1,unified:false,kv_location:'gpu',kv_type:'q8_0',ram_cache:0,batch:512,ubatch:128});}render();preview()}
async function guard(fn){try{return await fn()}catch(e){toast(e.message);$('actionError').textContent=e.message;return null}}
async function act(name,data=form){return guard(async()=>{let r=await api(name,{...data,update_source:$('update_source').checked,fresh_build:$('fresh_build').checked});toast('요청 접수: '+name);poll();return r})}
async function stop(kind){if(confirm(kind+' 작업을 중지할까요?'))await act('stop',{kind})}
async function save(){await guard(async()=>{await api('save',form);toast('사용자 설정 저장 완료')})}
async function downloadModel(){let q=config.model_profiles[dl.profile].quants[dl.quant];if(confirm(`${dl.quant} 약 ${q.gb} GB를 다운로드/이어받기 할까요? 실제 크기는 로그에서 확인됩니다.`))await act('download',downloadForm())}
async function diagnose(){await guard(async()=>{let r=await api('diagnose',form);$('diagnosis').innerHTML=`<p>Python ${esc(r.python)} · 모델 디스크 여유 ${fmt(r.disk.free)} GiB</p><table><tr><th>항목</th><th>상태 / 경로</th></tr>${r.items.map(x=>`<tr><td>${esc(x.name)}</td><td><span class="${x.ok?'good':'bad'}">${x.ok?'정상':'확인 필요'}</span> <a href="${esc(x.link)}" target="_blank" rel="noreferrer">설치/안내</a><div class="muted">${esc(x.detail)}</div></td></tr>`).join('')}</table><pre>${esc(JSON.stringify(r.paths,null,2))}</pre>`;toast('환경 검사 완료')})}
function metric(k,v,h=''){return `<div class="metric"><small>${esc(k)}</small><b>${esc(v)}</b><small>${esc(h)}</small></div>`}
async function preview(){return guard(async()=>{let r=await api('preview',form),e=r.estimate;$('estimate').innerHTML=`<div class="grid">${metric('예상 VRAM',fmt(e.gpu)+' GiB','여유 '+fmt(e.vfree))}${metric('예상 RAM 예산',fmt(e.host_ram)+' GiB','여유 '+fmt(e.ram_free))}${metric('GPU 상주 가중치',fmt(e.gpu_weight)+' GiB','CPU '+fmt(e.cpu_weight))}${metric('활성 KV',fmt(e.kv)+' GiB','SWA 미절약 시 '+fmt(e.kv_worst))}${metric('KV pool',e.pool.toLocaleString(),'tokens')}${metric('전체 가중치',fmt(e.weight)+' GiB','디스크 여유 '+fmt(e.archive_free))}</div><p>GPU별 추정: ${e.gpu_allocation.map((x,i)=>`GPU ${e.gpu_ids[i]}: ${fmt(x)} GiB`).join(' / ')||'—'}</p><p>모델: ${esc(e.model||'아직 다운로드 안 됨 / 경로 미지정')}</p>${e.danger.map(x=>`<p class="bad">⚠ ${esc(x)}</p>`).join('')}${e.warnings.map(x=>`<p class="warn">• ${esc(x)}</p>`).join('')}`;$('command').textContent=r.command;return r})}
async function installBinary(){if(confirm('공식 llama.cpp GitHub 릴리스의 Windows CUDA 실행 파일 및 DLL을 다운로드하고 실행 파일 버전을 확인할까요? GPU 드라이버 호환성은 별도 확인이 필요합니다.'))await act('install_binary')}
function useInstalled(){let x=prev?.tasks?.build?.meta?.exe;if(!x){toast('완료된 빌드/설치 실행 파일이 없습니다.');return}form.server_exe=x;render();save();toast('실행 파일 적용: '+x)}
async function plan(){await guard(async()=>{const r=await api('plan',form);plans=r.candidates;renderPlans(r.message)})}
function renderPlans(msg){$('plans').innerHTML='<p>'+esc(msg)+'</p>'+ (plans.length?'<table><tr><th>후보</th><th>CPU 층 / VRAM / RAM</th><th></th></tr>'+plans.map((p,i)=>`<tr><td>${esc(p.name)}${p.swa_risk?' · SWA 위험':''}</td><td>${p.settings.cpu_layers}층 / ${fmt(p.gpu)} / ${fmt(p.ram)} GiB</td><td><button onclick="applyPlan(${i})">적용</button></td></tr>`).join('')+'</table>':'')}
function applyPlan(i){if(!plans[i])return;Object.assign(form,plans[i].settings);render();preview()}
async function runBench(tuning){const prompt=Number($('bench_prompt').value),out=Number($('bench_output').value);if(!Number.isInteger(prompt)||!Number.isInteger(out)){toast('정수 토큰 수를 입력하세요.');return}if(!confirm(tuning?'현재 실행 설정을 기준으로 여러 후보를 로딩해 실제 속도를 측정할까요? 다른 사용자 작업이 없어야 하며 장시간 걸릴 수 있습니다.':'실행 중 서버에 합성 테스트 요청을 보냅니다. 다른 작업이 없을 때만 진행하세요.'))return;await act(tuning?'tune':'benchmark',{...form,bench_prompt:prompt,bench_output:out});poll()}
function applyBest(){const b=prev?.tasks?.tune?.meta?.best;if(!b){toast('성공한 자동 튜닝 결과가 없습니다.');return}const a=b.settings;if(a.profile!==form.profile||a.quant!==form.quant||a.context!==form.context||a.kv_type!==form.kv_type||a.kv_location!==form.kv_location||a.parallel!==form.parallel){toast('모델/컨텍스트/슬롯/KV 설정이 측정 조건과 다릅니다. 동일하게 맞춘 후 적용하세요.');return}Object.assign(form,{mode:a.cpu_layers?'hybrid':'gpu',cpu_layers:a.cpu_layers,expert_cache:a.expert_cache,cpu_share:a.cpu_share});render();tab('run');preview();toast('실측 후보 적용. 예상치 확인 후 시작하세요.')}
function benchView(s){let h=s.history||[];let best=s.tasks.tune?.meta?.best;$('tuneBest').textContent=best?`가장 빠른 측정: ${fmt(best.predicted_per_second)} t/s (warm), CPU ${best.settings.cpu_layers}층 / ${best.settings.expert_cache} MiB / share ${best.settings.cpu_share}`:'아직 성공한 튜닝 결과 없음';$('history').innerHTML=h.length?'<table><tr><th>시각 / 모델</th><th>입력 · 출력</th><th>Prefill / Decode (t/s)</th><th>CPU 층 / 캐시 MiB / share</th></tr>'+h.slice().reverse().map(x=>`<tr><td>${esc(x.date)}<br>${esc(x.settings.quant)} (${esc(x.kind)})</td><td>${esc(x.requested_prompt)} / ${esc(x.requested_output)}</td><td>${fmt(x.prompt_per_second)} / ${fmt(x.predicted_per_second)}</td><td>${esc(x.settings.cpu_layers)} / ${esc(x.settings.expert_cache)} / ${esc(x.settings.cpu_share)}</td></tr>`).join('')+'</table>':'<p>아직 측정 결과가 없습니다.</p>';logs('benchmarkLog',s.logs.benchmark||[]);logs('tuneLog',s.logs.tune||[])}

async function startServer(){let pr=await preview();if(!pr)return;const dangers=pr.estimate.danger;if(dangers.length&&!confirm('위험 설정입니다. 그래도 진행할까요?\n\n'+dangers.join('\n')))return;await guard(async()=>{let r=await api('start',{...form,force:!!dangers.length});if(r.confirm){if(!confirm(r.danger.join('\n')+'\n그래도 진행할까요?'))return;await api('start',{...form,force:true})}toast('서버 실행 요청 완료');tab('monitor');poll()})}
function connection(meta=null){if(!config)return;let p=meta?.port||form.port,host=meta?.host||form.host,alias=meta?.alias||form.alias;let base=`http://127.0.0.1:${p}`;$('connection').innerHTML=`<a href="${base}" target="_blank" rel="noreferrer">llama.cpp 채팅 UI 열기</a><pre>OpenAI: ${base}/v1\nAnthropic: ${base}\nModel: ${esc(alias)}\n${host==='0.0.0.0'?'원격 노트북은 127.0.0.1 대신 실제 서버 IP 사용':'현재 로컬 전용'}</pre>`}
function logs(id,items,follow=true){const x=$(id),v=items.join('\n');if(x.textContent!==v){x.textContent=v;if(follow)x.scrollTop=x.scrollHeight}}
function rateMetrics(m){let out=[];for(const ln of m.split('\n')){if(ln.startsWith('#'))continue;const z=ln.match(/^([^\s{]+)(?:\{[^}]*\})?\s+([\d.eE+\-]+)$/);if(z&&/tokens_per_second|requests_processing|requests_deferred|kv_cache_usage|predicted_tokens_total|prompt_tokens_total/.test(z[1]))out.push(metric(z[1].replace(/^llamacpp:/,''),fmt(Number(z[2]))))}return out.slice(0,12).join('')}
async function poll(){if(polling||!config)return;polling=true;try{let s=await api('status');prev=s;let st=s.tasks.server;let total=s.gpu.reduce((a,g)=>a+g.total,0),used=s.gpu.reduce((a,g)=>a+g.used,0);$('summary').innerHTML=metric('GPU VRAM',fmt(used)+' / '+fmt(total)+' GiB',`${s.gpu.length} GPU`)+metric('RAM 여유',fmt(s.ram_free)+' GiB','전체 '+fmt(s.ram_total)+' GiB')+metric('추론 서버',st?.active?(s.health?'응답 중':'시작/로딩 중'):(st?.state||'미실행'),st?.pid?'PID '+st.pid:'');$('gpuCards').innerHTML=s.gpu.map(g=>`<div class="metric"><small>GPU ${g.index} · ${esc(g.name)}</small><b>${fmt(g.used)} / ${fmt(g.total)} GiB</b><div class="bar"><i style="width:${Math.min(100,g.used/g.total*100)}%"></i></div><small>GPU ${fmt(g.util)}% · ${fmt(g.temp)}°C · ${fmt(g.power)}W</small></div>`).join('')||'<p>GPU 미감지</p>';$('taskCards').innerHTML=['download','build','server','benchmark','tune'].map(k=>metric(k,s.tasks[k]?.state||'idle',s.tasks[k]?.pid?'PID '+s.tasks[k].pid:'')).join('');$('health').textContent=s.health||'서버 health 응답 없음';logs('downloadLog',s.logs.download);logs('buildLog',s.logs.build);logs('serverLog',s.logs.server,$('follow').checked);$('serverErrors').textContent=(s.server_errors||[]).join('\n')||'—';$('metrics').textContent=s.metrics||'metrics 응답 없음';$('slots').textContent=JSON.stringify(s.slots,null,2);$('rates').innerHTML=rateMetrics(s.metrics)+(s.cache_stats?.samples?metric('Expert cache hit (최근 로그)',fmt(s.cache_stats.hit_rate)+'%',s.cache_stats.samples+' samples'):metric('Expert cache hit','—','지원 로그가 있을 때만 표시'));benchView(s);if(st?.active)connection(st.meta)}catch(e){$('health').textContent=e.message}finally{polling=false}}
async function keys(){await guard(async()=>drawKeys((await api('keys')).keys))}function drawKeys(a){$('keyList').innerHTML=a.length?`<table><tr><th>키</th><th>관리</th></tr>${a.map(x=>`<tr><td>${esc(x.mask)}</td><td><button onclick="key('remove',${x.index})">회수</button></td></tr>`).join('')}</table>`:'<p>등록된 키 없음</p>'}async function key(action,index){if(action==='remove'&&!confirm('이 키를 회수할까요?'))return;await guard(async()=>{let r=await api('key',{action,index});drawKeys(r.keys);if(r.new){$('newKey').innerHTML='<p class="warn">새 키 (한 번만 표시 · 안전하게 보관)</p><pre></pre><button onclick="navigator.clipboard.writeText(this.previousElementSibling.textContent)">복사</button>';$('newKey').querySelector('pre').textContent=r.new}else $('newKey').innerHTML='';toast('키 변경 완료; 서버 재시작 후 적용')})}
(async()=>{try{config=await api('config');form=config.form;dl={profile:form.profile,quant:form.quant,vision:form.vision};render();tab('env');diagnose();poll();setInterval(poll,3000)}catch(e){toast(e.message)}})();
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def send(self,status,data,content='application/json; charset=utf-8'):
        if not isinstance(data,bytes):data=json.dumps(data,ensure_ascii=False).encode('utf-8')
        self.send_response(status);self.send_header('Content-Type',content);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('X-Frame-Options','DENY');self.send_header('Referrer-Policy','no-referrer');self.end_headers();self.wfile.write(data)
    def authorized(self):
        # Defend localhost admin against cross-site requests and DNS rebinding.
        expected=f'127.0.0.1:{self.server.server_port}'
        host=self.headers.get('Host','');origin=self.headers.get('Origin')
        return host==expected and (not origin or origin=='http://'+expected)
    def do_GET(self):
        try:
            if not self.authorized():return self.send(403,{'error':'local manager only'})
            if self.path=='/':return self.send(200,HTML.replace('__TOKEN__',json.dumps(TOKEN)).encode('utf-8'),'text/html; charset=utf-8')
            if self.headers.get('X-Manager-Token')!=TOKEN:return self.send(403,{'error':'invalid manager session token'})
            if self.path=='/api/config':
                w,l,m,x=paths();form={**BASE['defaults'],'workspace':str(w),'llama_dir':BASE['llama_dir'],'model_dir':BASE['model_dir'],'server_exe':BASE['server_exe'],'build_jobs':BASE['build_jobs'],'cuda_arch':BASE['cuda_arch'],'llama_ref':BASE['llama_ref'],'cuda_root':BASE.get('cuda_root',''),'runtime_cuda':BASE.get('runtime_cuda','12'),'runtime_tag':BASE.get('runtime_tag',''),**SETTINGS};return self.send(200,{'form':form,'model_profiles':BASE['model_profiles']})
            if self.path=='/api/status':return self.send(200,status())
            if self.path=='/api/keys':return self.send(200,{'keys':masked_keys()})
            return self.send(404,{'error':'not found'})
        except (BrokenPipeError,ConnectionResetError):pass
        except Exception as e:self.send(500,{'error':str(e)})
    def do_POST(self):
        try:
            if not self.authorized() or self.headers.get('X-Manager-Token')!=TOKEN:return self.send(403,{'error':'invalid manager session'})
            if self.headers.get_content_type()!='application/json':return self.send(415,{'error':'JSON required'})
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<200000:return self.send(413,{'error':'invalid request size'})
            raw=json.loads(self.rfile.read(length));d={**BASE['defaults'],**SETTINGS,**raw}; action=self.path.removeprefix('/api/');result={'ok':True}
            if action=='diagnose':result=diagnostics(d)
            elif action=='diagnostic_command':result=diagnostic_command(d)
            elif action=='save':result=save_prefs(d)
            elif action=='preview':
                _,e,args=command_preview(d);result={'estimate':e,'command':subprocess.list2cmdline(args)}
            elif action=='download':download(d)
            elif action=='inspect':download(d,True)
            elif action=='install_hf':install_hf()
            elif action=='build':build(d)
            elif action=='install_binary':install_binary(d)
            elif action=='plan':result=placement_plan(d)
            elif action=='recommend':result=recommend_setup(d)
            elif action=='benchmark':benchmark(d)
            elif action=='tune':tune(d)
            elif action=='start':result=launch(d)
            elif action=='stop':
                if raw.get('kind') not in ('download','server','build','tune'):raise ValueError('invalid task')
                stop_task(raw['kind'])
            elif action=='key':result=key_action(raw['action'],raw.get('index'))
            else:return self.send(404,{'error':'not found'})
            self.send(200,result)
        except (BrokenPipeError,ConnectionResetError):pass
        except Exception as e:self.send(400,{'error':str(e)})


def main():
    if sys.version_info<(3,10):raise SystemExit('Python 3.10+ required; 3.12 recommended.')
    # Local service, independent from configurable inference server port.
    try:srv=ThreadingHTTPServer(('127.0.0.1',8765),Handler)
    except OSError:srv=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    url=f'http://127.0.0.1:{srv.server_port}/'
    print('Inkling Manager:',url,'\nKeep this terminal open. Ctrl+C stops managed tasks.\nWorkspace:',WORK,flush=True)
    if '--no-browser' not in sys.argv:threading.Timer(.5,lambda:webbrowser.open(url)).start()
    try:srv.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        for k in ('tune','server','download','build','benchmark'):stop_task(k)
        srv.server_close()

if __name__=='__main__':main()
