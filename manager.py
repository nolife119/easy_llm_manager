"""Local-only Windows llama.cpp manager. Python 3.12, standard library UI."""
from __future__ import annotations
import base64, zlib
import speed
import ctypes, fnmatch, hashlib, json, os, re, secrets, shutil, socket, struct, subprocess, sys, tempfile, threading, time, traceback, urllib.request, urllib.error, webbrowser, zipfile
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
LAST_READY_FILE=STATE/'last-ready.json'
try: LAST_READY=json.loads(LAST_READY_FILE.read_text(encoding='utf-8'))
except (OSError,ValueError): LAST_READY={}
if not isinstance(LAST_READY,dict): LAST_READY={}
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
    if 'key not found in model' in x or 'error loading model hyperparameters' in x:
        return 'GGUF 메타데이터와 엔진 변환 스키마 불일치 가능성이 큽니다. 누락 키를 임의 추가하지 말고 기존 GGUF와 호환되는 체크아웃을 별도 빌드/선택하세요.'
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


def source_state(d=None):
    _,l,_,_=paths(d);git=find_tool('git')
    if not git or not l.is_dir():return {'path':str(l),'head':'','branch':'','dirty':None}
    rc,h=run_capture([git,'-C',str(l),'rev-parse','HEAD'],10)
    if rc or not re.fullmatch(r'[0-9a-f]{40,64}',h.strip()):return {'path':str(l),'head':'','branch':'','dirty':None}
    _,b=run_capture([git,'-C',str(l),'rev-parse','--abbrev-ref','HEAD'],10)
    rc,status=run_capture([git,'-C',str(l),'status','--porcelain','--untracked-files=no'],10)
    return {'path':str(l),'head':h.strip(),'branch':b.strip(),'dirty':bool(status.strip()) if rc==0 else None}

def engine_inventory(d=None):
    w,l,_,x=paths(d);source=source_state(d);found=[];seen=set()
    paths_to_check=[x] if x else []
    paths_to_check += [l/'build/bin/Release/llama-server.exe',l/'build/bin/llama-server.exe',l/'build/bin/llama-server',l/'llama-server.exe']
    paths_to_check += list(l.glob('build-cuda-*/bin/Release/llama-server.exe'))+list(l.glob('build-cuda-*/bin/llama-server'))+list(l.glob('build-cuda-*/bin/llama-server.exe'))
    paths_to_check += list((w/'runtime'/'llama.cpp').glob('*/**/llama-server.exe'))
    for f in paths_to_check:
        if not f or not f.is_file() or str(f.resolve()) in seen:continue
        seen.add(str(f.resolve()));manifest={}
        for parent in (f.parent,*list(f.parents)[:4]):
            m=parent/'easy-llm-build.json'
            if m.is_file():
                try:manifest=json.loads(m.read_text(encoding='utf-8'))
                except (OSError,ValueError):pass
                break
        found.append({'path':str(f),'commit':manifest.get('commit',''),'kind':manifest.get('kind','unknown'),'matches':bool(source['head'] and manifest.get('commit')==source['head'] and manifest.get('clean') and source['dirty'] is False)})
    selected=find_server(d)
    version=''
    features={}
    if selected:
        env=cuda_env(d);env['PATH']=str(selected.parent)+os.pathsep+env.get('PATH','')
        rc,out=run_capture([selected,'--version'],15,env=env)
        version=out.strip()[-1500:]
        try:
            h=engine_help(str(selected),env)
            features={flag:help_has(h,flag) for flag in ('--n-cpu-moe','--moe-cache-mib','--moe-cache-cpu-share','--no-mmproj','--cache-type-k','--kv-unified','--chat-template-kwargs')}
        except (ValueError,OSError):pass
    return {'source':source,'engines':found,'selected':str(selected or ''),'version':version,'features':features}

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


def gguf_metadata(path):
    """Read bounded GGUF metadata without loading tensor data or rewriting files."""
    sizes={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    formats={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
    end=min(path.stat().st_size,512*1024**2)
    with path.open('rb') as f:
        def read(n):
            if n<0 or f.tell()+n>end:raise ValueError('GGUF metadata 범위가 잘못되었거나 검사 한도를 초과했습니다.')
            data=f.read(n)
            if len(data)!=n:raise ValueError('GGUF metadata EOF')
            return data
        def skip(n):
            if n<0 or f.tell()+n>end:raise ValueError('GGUF metadata 범위 오류')
            f.seek(n,1)
        def uint(fmt):return struct.unpack('<'+fmt,read(struct.calcsize('<'+fmt)))[0]
        def string(keep):
            n=uint('Q')
            if keep:
                if n>1048576:raise ValueError('GGUF metadata 문자열 검사 한도 초과')
                return read(n).decode('utf-8','replace')
            skip(n)
        if read(4)!=b'GGUF':raise ValueError('GGUF header 오류')
        version=uint('I')
        if version not in (2,3):raise ValueError('지원하지 않는 GGUF metadata 버전: '+str(version))
        tensors,count=uint('Q'),uint('Q')
        if count>100000:raise ValueError('GGUF metadata 키 검사 한도 초과')
        out={}
        for _ in range(count):
            key=string(True);typ=uint('I')
            keep=key in ('general.architecture','general.name','general.file_type','split.no','split.count','split.tensors.count') or key.startswith(('inkling2.attention.','inkling2.block_count','inkling2.context_length','clip.projector_type'))
            if typ in sizes:
                if keep:out[key]=uint(formats[typ])
                else:skip(sizes[typ])
            elif typ==8:
                val=string(keep)
                if keep:out[key]=val
            elif typ==9:
                item,n=uint('I'),uint('Q')
                if n>10000000:raise ValueError('GGUF metadata 배열 검사 한도 초과')
                if item in sizes:skip(n*sizes[item])
                elif item==8:
                    for __ in range(n):string(False)
                else:raise ValueError('지원하지 않는 GGUF 배열 타입')
                if keep:out[key]='array('+str(n)+')'
            else:raise ValueError('지원하지 않는 GGUF 타입: '+str(typ))
        return {'version':version,'tensor_count':tensors,'metadata':out}

def inspect_local_model(raw):
    f=resolve_model(raw);files=check_shards(f); result=gguf_metadata(f)
    md=result['metadata'];notes=[]
    arch=md.get('general.architecture')
    if raw.get('profile')=='inkling2' and arch!='inkling2':notes.append('선택한 Inkling 2 모델과 GGUF architecture가 다릅니다. mmproj 또는 다른 모델 파일인지 확인하세요.')
    if arch=='inkling2' and 'inkling2.attention.relative_length' not in md:notes.append('relative_length 없음: 기존 GGUF와 최신 엔진의 스키마 차이일 수 있습니다. 현재 체크아웃 호환 엔진으로 시험하세요. 값을 임의로 추가하지 않습니다.')
    if md.get('split.count') is not None and md['split.count']!=len(files):notes.append('GGUF split.count와 실제 분할 파일 수가 다릅니다.')
    return {**result,'path':str(f),'shards':len(files),'gib':sum(x.stat().st_size for x in files)/GIB,'notes':notes}

def num(d,k,lo,hi,integer=True):
    v=float(d[k]);
    if not lo<=v<=hi or (integer and v!=int(v)):raise ValueError(f'{k}: {lo}~{hi} 범위 필요')
    return int(v) if integer else v


def validate(d):
    p,q=profile(d); d={**BASE['defaults'],**d}
    d.update(speed.validated(d))
    for k,lo,hi in [('context',512,p['max_context']),('pool',512,33554432),('parallel',1,128),('cpu_layers',0,p['layers']),('expert_cache',0,1048576),('ram_cache',0,4194304),('runtime_reserve',1,1024),('ram_reserve',1,4096),('batch',1,32768),('ubatch',1,32768),('threads',1,1024),('port',1024,65535)]:d[k]=num(d,k,lo,hi)
    d['cpu_share']=num(d,'cpu_share',0,1,False)
    if not d['unified']:d['pool']=d['context']*d['parallel']
    elif d['pool']<d['context']:raise ValueError('pool은 사용자당 최대 context 이상이어야 합니다.')
    if d['ubatch']>d['batch']:raise ValueError('ubatch는 batch 이하여야 합니다.')
    if d['mode'] not in ('gpu','hybrid') or d['kv_location'] not in ('gpu','cpu') or d['kv_type'] not in ('f16','q8_0','q4_0'):raise ValueError('지원하지 않는 모드입니다.')
    if d['host'] not in ('127.0.0.1','0.0.0.0'):raise ValueError('지원하는 호스트: 127.0.0.1 / 0.0.0.0')
    if d['reasoning'] not in ('none','minimal','low','medium','high','xhigh','max'):raise ValueError('reasoning 설정 오류')
    if d.get('gpu_fast') and recovery_options(d).get('--no-kv-offload') is True:raise ValueError('GPU 최대 모드와 CPU KV 복구 옵션이 충돌합니다.')
    if d.get('gpu_fast') and (d['mode']!='gpu' or d['kv_location']!='gpu' or d['parallel']!=1 or d['unified'] or d['ram_cache']):raise ValueError('GPU 최대 모드는 가중치/KV GPU, 단일 슬롯, RAM prompt cache 0 전용입니다.')
    if d.get('load_mode','auto') not in ('auto','mmap','none'):raise ValueError('모델 로딩 방식 오류')
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


def estimate(raw,hardware=None,weight_override=None,fast=False):
    d,p,q=validate(raw); all_g,rt,rf=hardware if hardware is not None else (gpus(),*ram()); gs=selected_gpus(d,all_g); warnings=[]; danger=[]
    f=None if fast else resolve_model(d); weight=q['gb']*1e9/GIB
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
    if recovery_options(d).get('--swa-full') is True:kv=worst
    expert=d['expert_cache']/1024
    gpu=weight-cpu+expert+(kv if d['kv_location']=='gpu' else 0)+d['runtime_reserve']+(2 if d['vision'] else 0)
    # mmap may also keep file pages in RAM, reclaimable; this is mandatory-ish budget, not physical RSS guarantee.
    host=cpu+(kv if d['kv_location']=='cpu' else 0)+d['ram_cache']/1024+d['ram_reserve']
    load_mode=d.get('load_mode','auto')
    # Windows mapped GPU pages may remain resident. Keep separate peak-risk budgets, not a fabricated exact RSS.
    mapped_extra=weight-cpu if WIN and load_mode in ('auto','mmap') and recovery_options(d).get('--no-mmap') is not True else 0
    host_peak=host+mapped_extra
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
    if d['host']=='0.0.0.0' and not d['auth']:danger.append('사내망 접속에는 API 키가 필수입니다. 인증을 켜세요.')
    if d['ram_cache']:warnings.append('RAM prompt cache는 유휴/재사용 캐시입니다. 활성 KV 전체를 대체하지 않습니다. 다중 사용자 격리를 검증하세요.')
    if mapped_extra and rf is not None and host_peak>rf:warnings.append(f'Windows mmap 상주 위험: RAM 필요 추정 {host:.1f} GiB, 파일 페이지 포함 시 {host_peak:.1f} GiB > 여유 {rf:.1f} GiB. 직접 읽기 방식도 별도 검증하세요.')
    if d['context']>=262144:warnings.append('긴 컨텍스트는 prefill 시간, 추가 상태 및 체크포인트 메모리를 크게 증가시킬 수 있습니다.')
    warnings.append('가중치 배치/Expert 비율 및 SWA는 추정치입니다. 프로젝트 파일, CUDA 버퍼, mmap, GPU별 레이어 불균형에 따라 실측이 달라집니다.')
    return {'weight':weight,'gpu_weight':weight-cpu,'cpu_weight':cpu,'kv':kv,'kv_worst':worst,'gpu':gpu,'host_ram':host,'host_peak_risk':host_peak,'mapped_extra_risk':mapped_extra,'load_mode':load_mode,'vfree':vfree,'vtotal':vtotal,'ram_free':rf,'ram_total':rt,'gpu_allocation':per,'ratios':ratios,'gpu_ids':[g['index'] for g in gs],'pool':d['pool'],'warnings':warnings,'danger':danger,'model':str(f or ''),'archive_free':None if fast else disk(repo_path(d)).get('free')}


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

SMART_KEYS=('profile','quant','model_path','context','parallel','unified','pool','mode','cpu_layers','expert_cache','cpu_share','expert_verbose','kv_location','kv_type','runtime_reserve','ram_reserve','ram_cache','batch','ubatch','threads','vision','gpu_ids','split','load_mode','gpu_fast')
def smart_settings(d):
    return {k:d.get(k,BASE['defaults'].get(k)) for k in SMART_KEYS}

def smart_search(base,p,weight,hardware,strategy,precision,policy,caps):
    """Bounded deterministic search; fixed model/context/slots/vision/KV placement."""
    gs=selected_gpus(base,hardware[0]);free=sum(g['free'] for g in gs);rf=hardware[2]
    if not gs or rf is None:return [],None,0
    moe=p.get('expert_fraction_estimate',0)>0 and caps['cpu']
    if strategy in ('static','dynamic') and not moe:return [],None,0
    if strategy=='dynamic' and not caps['dynamic']:return [],None,0
    types=[precision] if precision!='auto' else ['f16','q8_0','q4_0']
    types=[t for t in types if t=='f16' or caps['quant_kv']]
    if not types:return [],None,0
    normal=max(6,min(24,round(free*.08)))
    tight=max(4,2*len(gs)+2)
    levels=list(dict.fromkeys([(normal,512,256),(max(tight,round(normal*.75)),512,128),
            (max(tight,round(normal*.5)),256,128),(tight,128,64)]))
    if base['purpose']=='safe':levels=[(normal,256,128),(max(tight,round(normal*.75)),128,64)]
    if policy=='safe':levels=levels[:2]
    hosts=sorted({max(8,min(32,round(rf*.10))),max(4,min(12,round(rf*.04)))},reverse=True)
    if policy=='safe':hosts=hosts[:1]
    layers=[0] if strategy=='gpu' or not moe else list(range(p['layers']+1))
    if strategy in ('static','dynamic'):layers=[n for n in layers if n>0]
    caches=[0] if strategy in ('gpu','static','auto') else [4096,8192,16384,24576]
    tested=0;best=[];nearest=None
    for n in layers:
        for cache in caches:
            if cache and (not n or cache/1024>weight*p.get('expert_fraction_estimate',0)*n/p['layers']):continue
            for kt in types:
                for ri,(reserve,batch,ub) in enumerate(levels):
                    for hi,host in enumerate(hosts):
                        d={**base,'mode':'hybrid' if n else 'gpu','cpu_layers':n,'expert_cache':cache,'cpu_share':.5 if cache else 0,
                           'kv_type':kt,'runtime_reserve':reserve,'ram_reserve':host,'batch':batch,'ubatch':ub}
                        e=estimate(d,hardware,weight,fast=True);tested+=1
                        gaps=[g['free']-v for g,v in zip(gs,e['gpu_allocation'])]
                        vg=e['vfree']-e['gpu'];rg=rf-e['host_ram']
                        extra=max(0,e['kv_worst']-e['kv'])
                        swa_risk=(vg if d['kv_location']=='gpu' else rg)<extra-1e-6
                        deficit=max(0,-vg,max((-x for x in gaps),default=0))+max(0,-rg)
                        item={'settings':smart_settings(d),'gpu':e['gpu'],'ram':e['host_ram'],'weight':weight,
                              'gpu_weight':e['gpu_weight'],'cpu_weight':e['cpu_weight'],'kv':e['kv'],'kv_worst':e['kv_worst'],
                              'gpu_free':e['vfree'],'ram_free':rf,'gpu_gap':vg,'ram_gap':rg,'swa_risk':swa_risk,
                              'deficit':deficit,'per_gpu':[{'id':g['index'],'free':g['free'],'need':v,'gap':g['free']-v} for g,v in zip(gs,e['gpu_allocation'])],
                              'warnings':e['warnings'],'danger':e['danger'],'host_peak_risk':e['host_peak_risk'],'mapped_extra_risk':e['mapped_extra_risk'],'runtime_reserve':reserve,'ram_reserve':host,
                              'vision_reserve':2 if d['vision'] else 0,'rank':[]}
                        # Prefer avoiding uncertain full-SWA allocation failures, CPU offload, low KV precision, and then tight buffers.
                        rank=(int(swa_risk),int(e['host_peak_risk']>rf),n,types.index(kt),ri,hi,-cache)
                        item['rank']=list(rank)
                        key=(deficit,)+rank
                        if nearest is None or key<nearest[0]:nearest=(key,item)
                        memory_errors=[x for x in e['danger'] if 'API 키' not in x]
                        if deficit>1e-6 or memory_errors:continue
                        best.append((rank,item))
    best.sort(key=lambda x:x[0])
    # Return meaningfully distinct fallbacks, including smaller batches and precision.
    out=[];seen=set()
    for _,item in best:
        d=item['settings'];key=(d['cpu_layers'],d['expert_cache'],d['kv_type'],d['batch'],d['ubatch'])
        if key in seen:continue
        seen.add(key);out.append(item)
        if len(out)>=6:break
    if out:
        speed=min((item for _,item in best),key=lambda x:(x['settings']['cpu_layers'],x['swa_risk'],x['rank']))
        if speed not in out:out.append(speed)
    if out and not any(x['settings']['kv_type']=='f16' for x in out):
        f16=next((item for _,item in best if item['settings']['kv_type']=='f16'),None)
        if f16:out.append(f16)
    return out,(nearest[1] if nearest else None),tested

def gpu_context_search(base,p,weight,hardware,precision,policy,caps):
    """Max context is an explicit objective, not a silent relaxation of fixed requests."""
    limit=p['max_context']
    ladder=sorted({limit,*[x*1024 for x in (4,8,16,24,32,48,64,96,128,160,192,224,256,320,384,448,512,640,768,896,1024) if x*1024<=limit]},reverse=True)
    kinds=[precision] if precision!='auto' else ['f16','q8_0','q4_0']
    kinds=[k for k in kinds if k=='f16' or caps['quant_kv']]
    levels=[];nearest=None;count=0
    # Locate each precision's tight estimated ceiling before building the fallback ladder.
    for k in kinds:
        lo,hi=1,max(1,limit//4096);upper=0
        while lo<=hi:
            mid=(lo+hi)//2;ctx=min(limit,mid*4096)
            probe={**base,'purpose':'coding','context':ctx,'pool':ctx,'parallel':1,'unified':False,'mode':'gpu',
                   'cpu_layers':0,'expert_cache':0,'cpu_share':0,'kv_location':'gpu','ram_cache':0,'gpu_fast':True}
            fits,_,n=smart_search(probe,p,weight,hardware,'gpu',k,policy,caps);count+=n
            if fits:upper=ctx;lo=mid+1
            else:hi=mid-1
        if upper:ladder.append(upper)
    ladder=sorted(set(ladder),reverse=True)
    for ctx in ladder:
        fixed={**base,'purpose':'coding','context':ctx,'pool':ctx,'parallel':1,'unified':False,'mode':'gpu',
               'cpu_layers':0,'expert_cache':0,'cpu_share':0,'kv_location':'gpu','ram_cache':0,'gpu_fast':True}
        for k in kinds:
            candidates,near,n=smart_search(fixed,p,weight,hardware,'gpu',k,policy,caps);count+=n
            if near and (nearest is None or near['deficit']<nearest['deficit']):nearest=near
            if candidates:
                # The best modeled configuration at each length/precision, plus a small-batch fallback.
                levels.append(candidates[0])
                smaller=min(candidates,key=lambda x:(x['settings']['ubatch'],x['settings']['batch'],x['rank']))
                if smaller!=candidates[0]:levels.append(smaller)
    if not levels:return [],nearest,count
    korder={'f16':0,'q8_0':1,'q4_0':2}
    def rank(x):
        d=x['settings']
        return (-d['context'],int(x['swa_risk']),korder[d['kv_type']],x['rank'])
    levels.sort(key=rank)
    # Default to the greatest modeled length; separately expose conservative and F16 alternatives.
    top=levels[0];out=[top]
    def add(x):
        if x is not None and x not in out:out.append(x)
    add(next((x for x in levels if not x['swa_risk']),None))
    for k in kinds:
        add(next((x for x in levels if x['settings']['kv_type']==k),None))
    for fraction in (.75,.5,.25):
        add(next((x for x in levels if x['settings']['context']<=top['settings']['context']*fraction and x['settings']['kv_type']==top['settings']['kv_type']),None))
    add(next((x for x in levels if x['settings']['kv_type']=='f16' and x['settings']['context']<=32768),None))
    add(next((x for x in levels if x['settings']['context']==top['settings']['context'] and x['settings']['ubatch']<top['settings']['ubatch']),None))
    out.sort(key=rank)
    return out,top,count

def recommend_setup(raw):
    """Search current free hardware; no automatic model download, launch or constraint relaxation."""
    p,q=profile(raw)
    purpose=raw.get('wizard_purpose','coding');location=raw.get('wizard_kv','gpu');strategy=raw.get('wizard_strategy','auto')
    precision=raw.get('wizard_precision','auto');policy=raw.get('wizard_policy','balanced')
    quant_mode=raw.get('wizard_quant_mode','fixed');target=num({'v':raw.get('wizard_target',10)},'v',1,1000,False)
    if purpose not in ('safe','coding','long','team','gpu_max') or location not in ('gpu','cpu') or strategy not in ('auto','gpu','static','dynamic') or precision not in ('auto','f16','q8_0','q4_0') or policy not in ('balanced','safe') or quant_mode not in ('fixed','quality','all'):
        raise ValueError('간편 설정 선택을 확인하세요.')
    if any(task_busy(k) for k in ('server','tune','benchmark','build')):raise ValueError('실행/튜닝/빌드 중인 작업을 중지하고 메모리가 반환된 뒤 계산하세요.')
    if purpose=='gpu_max' and task_busy('download'):raise ValueError('GPU 최대 길이 계산 전 모델 다운로드를 완료하거나 중지하세요. 파일 크기/가용 메모리 및 부하가 변할 수 있습니다.')
    opts=recovery_options(raw)
    conflicts=set(opts)&{'--cache-type-k','--cache-type-v','--no-kv-offload','--kv-unified','--no-kv-unified','--cache-ram','--n-cpu-moe','--moe-cache-mib','--moe-cache-cpu-share','--mmproj','--no-mmproj','--fit','--flash-attn'}
    if conflicts:raise ValueError('자동 최적화와 충돌하는 복구 옵션: '+', '.join(sorted(conflicts))+'. 복구 옵션을 초기화하거나 직접 실행하세요.')
    gs=gpus();rt,rf=ram();hardware=(gs,rt,rf)
    if not gs or rf is None:return {'ok':False,'message':'GPU/RAM 자동 감지 실패. 환경 진단을 확인하세요.'}
    maximize=purpose=='gpu_max'
    if maximize:
        location='gpu';strategy='gpu';quant_mode='fixed'
    desired=p['max_context'] if maximize else num({'v':raw.get('wizard_context',32768)},'v',512,p['max_context'])
    slots=num({'v':raw.get('wizard_slots',4)},'v',1,128) if purpose=='team' else 1
    d={**BASE['defaults'],**raw,'purpose':purpose,'context':desired,'parallel':slots,'unified':False,'pool':desired*slots,
       'mode':'gpu','cpu_layers':0,'expert_cache':0,'cpu_share':0,'expert_verbose':False,'kv_location':location,
       'kv_type':'f16','ram_cache':0,'runtime_reserve':8,'ram_reserve':8,'batch':256,'ubatch':128,
       'threads':max(1,(os.cpu_count() or 2)//2),'vision':bool(raw.get('vision',False)),'load_mode':raw.get('load_mode','auto'),'gpu_fast':maximize}
    # Keep explicitly selected GPUs and split; do not silently switch devices.
    selected=selected_gpus(d,gs)
    if not selected or any(g['free']<=0 for g in selected):return {'ok':False,'message':'선택 GPU에 여유 VRAM이 없습니다.'}
    exe=find_server(d)
    if not exe:return {'ok':False,'message':'llama-server 실행 파일을 환경에서 지정하세요.'}
    env=cuda_env(d);env['PATH']=str(exe.parent)+os.pathsep+env.get('PATH','')
    h=engine_help(str(exe),env)
    caps={'cpu':help_has(h,'--n-cpu-moe'),'dynamic':all(help_has(h,k) for k in ('--moe-cache-mib','--moe-cache-cpu-share','--n-cpu-moe')),
          'quant_kv':all(help_has(h,k) for k in ('--cache-type-k','--cache-type-v','--flash-attn')),
          'direct_load':help_has(h,'--load-mode') or help_has(h,'--no-mmap')}
    if d['vision'] and not help_has(h,'--mmproj'):return {'ok':False,'message':'선택 엔진은 Vision projector 옵션을 지원하지 않습니다.'}
    if strategy=='dynamic' and not caps['dynamic']:return {'ok':False,'message':'선택 엔진에 동적 Expert 캐시 옵션이 없습니다. 정적 또는 GPU 배치를 선택하세요.'}
    if strategy=='static' and not caps['cpu']:return {'ok':False,'message':'선택 엔진은 CPU Expert 분산 옵션을 지원하지 않습니다.'}
    if location=='cpu' and not help_has(h,'--no-kv-offload'):return {'ok':False,'message':'선택 엔진은 CPU KV 옵션을 지원하지 않습니다.'}
    if precision in ('q8_0','q4_0') and not caps['quant_kv']:return {'ok':False,'message':'선택 엔진은 양자화 KV 옵션을 지원하지 않습니다.'}
    # Explicit quality ordering uses weight size as proxy, not a measured quality/throughput ranking.
    quants=[d['quant']] if quant_mode=='fixed' else sorted(p['quants'],key=lambda k:p['quants'][k]['gb'],reverse=True)
    summaries=[];chosen=None;fallbacks=[];downloadable=[];nearest=None;tested=0
    for quant in quants:
        trial={**d,'quant':quant,'model_path':d.get('model_path','') if quant==d['quant'] else ''}
        f=resolve_model(trial);installed=False;weight=p['quants'][quant]['gb']*1e9/GIB;issue=''
        if f:
            try:
                shards=check_shards(f);weight=sum(x.stat().st_size for x in shards)/GIB;installed=True
            except (ValueError,OSError) as exc:issue=str(exc)
        projector=p.get('projector')
        if trial['vision']:
            paths_to_check=([repo_path(trial)/projector,f.parent/projector,f.parent.parent/projector] if f and projector else [repo_path(trial)/projector] if projector else [])
            if not any(x.is_file() for x in paths_to_check):installed=False;issue='Vision projector가 없습니다.'
        if maximize:candidates,near,count=gpu_context_search(trial,p,weight,hardware,precision,policy,caps)
        else:candidates,near,count=smart_search(trial,p,weight,hardware,strategy,precision,policy,caps)
        tested+=count
        if not maximize and WIN and caps['direct_load'] and trial['load_mode']=='auto' and weight<rf and not any(k in opts for k in ('--mmap','--no-mmap')):
            direct,near_direct,n_direct=smart_search({**trial,'load_mode':'none'},p,weight,hardware,strategy,precision,policy,caps);tested+=n_direct
            # Keep both loading modes. Direct reading may lower resident file pages, but peak RAM is unverified.
            if direct:candidates=sorted(candidates[:4]+direct[:4],key=lambda x:x['rank'])
            if near_direct and (near is None or near_direct['deficit']<near['deficit']):near=near_direct
        valid=bool(candidates)
        summary={'quant':quant,'installed':installed,'feasible':valid,'issue':issue,'weight':weight,
                 'deficit':near['deficit'] if near else None,'gpu_gap':near['gpu_gap'] if near else None,'ram_gap':near['ram_gap'] if near else None}
        summaries.append(summary)
        if near and (nearest is None or near['deficit']<nearest['deficit']):nearest=near
        if valid and not installed and quant_mode!='quality':
            downloadable.append({'quant':quant,'candidate':candidates[0],'issue':issue or '모델 다운로드 필요'})
        if installed and valid:
            if chosen is None:chosen=candidates[0];fallbacks=candidates[1:]
            else:fallbacks.append(candidates[0])
    result={'ok':bool(chosen),'tested':tested,'models':summaries,'target_tps':target,'downloadable':downloadable[:4],
            'candidate':chosen or nearest,'alternatives':fallbacks,'maximize_context':maximize,'hardware':{'gpus':selected,'ram_total':rt,'ram_free':rf},
            'message':'추정상 가능한 설정을 찾았습니다. 속도와 실제 초기화는 벤치마크 전까지 미검증입니다.' if chosen else '고정한 조건으로 설치된 모델의 실행 가능 후보를 찾지 못했습니다. 아래 부족량과 다운로드 후보를 확인하세요.'}
    if chosen:
        result['settings']=chosen['settings']
        if maximize:result['message']='GPU 전용 최대 컨텍스트 추정 후보입니다. 실제 최대 길이/속도는 순차 로딩과 벤치마크로 검증하세요.'
    return result


def command_preview(raw):
    d,p,q=validate(raw); e=estimate(d); exe=find_server(d); f=resolve_model(d)
    args=[str(exe or 'llama-server.exe'),'-m',str(f or '<GGUF first shard>'),'--alias',str(d['alias']),'-ngl','99','--split-mode',d['split_mode'],'-c',str(d['pool']),'-np',str(d['parallel']),'-b',str(d['batch']),'-ub',str(d['ubatch']),'-t',str(d['threads']),'--host',d['host'],'--port',str(d['port']),'--jinja','--metrics','--cont-batching','--fit','off']
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
    if d['host']=='0.0.0.0':args+=['--no-slots']
    if d['auth']:args+=['--api-key-file',str(KEYS)]
    opts=recovery_options(d)
    if opts:
        args=apply_recovery(args,opts)
        if '--no-mmproj' in args and '--mmproj' in args:
            raise ValueError('Vision 체크를 해제하거나 --mmproj를 제거하세요.')
        e['danger'].append('복구 옵션 적용 중: 메모리 예상값과 GUI 설정이 최종 명령과 다를 수 있습니다. 실제 명령과 메모리를 확인하세요.')
    if exe and exe.is_file():
        env=cuda_env(d);env['PATH']=str(exe.parent)+os.pathsep+env.get('PATH','')
        h=engine_help(str(exe),env)
        if d['spec_mode']=='draft-dspark':
            draft=absolute(d['draft_path'])
            if not draft.is_file() or draft.suffix.lower()!='.gguf' or draft==f:raise ValueError('별도의 호환 가능한 DSpark GGUF를 지정하세요.')
            d['draft_path']=str(draft)
            e['warnings'].append('DSpark 가중치/KV/검증 버퍼는 배치 추정에 포함되지 않습니다. 실제 초기화를 확인하세요.')
        if d['split_mode']!='layer':e['warnings'].append('변경된 GPU 분배 모드의 실제 GPU별 배치와 호환성은 추정할 수 없습니다.')
        args+=speed.options(d,h)
        if d.get('gpu_fast'):
            if help_has(h,'--kv-offload'):args+=['--kv-offload']
            for flag in ('--ctx-checkpoints',):
                if help_has(h,flag):args+=[flag,'0'];break
            # Checkpoints are disabled to reduce additional state; active KV remains in VRAM.
            if help_has(h,'--no-cache-idle-slots'):args+=['--no-cache-idle-slots']
        mode=d.get('load_mode','auto')
        if mode!='auto':
            if any(k in opts for k in ('--mmap','--no-mmap')):raise ValueError('모델 로딩 방식과 복구 mmap 옵션을 동시에 지정하지 마세요.')
            if help_has(h,'--no-mmap') and mode=='none':args+=['--no-mmap']
            elif help_has(h,'--mmap') and mode=='mmap':args+=['--mmap']
            elif help_has(h,'--load-mode'):args+=['--load-mode',mode]
            else:raise ValueError('선택 엔진에서 지정한 모델 로딩 방식 미지원')
        args,notes=compatible_args(d,args,h);e['warnings'].extend(notes)
        missing=[a for a in args[1:] if a.startswith('--') and not help_has(h,a)]
        if missing:e['warnings'].append('필수/명시 옵션 미지원 (실행 전 해결 필요): '+', '.join(missing))
    return d,e,args


def engine_help(exe,env=None):
    rc,h=run_capture([exe,'--help'],30,env=env)
    if rc!=0:raise ValueError('llama-server --help 실패: '+h[-600:])
    return h

def help_has(h,flag):
    return bool(re.search(r'(?<![\w-])'+re.escape(flag)+r'(?![\w-])',h))

def compatible_args(d,args,h):
    """Only remove unsupported optional/default features; never change model placement or security."""
    args=list(args);notes=[]
    optional={'--metrics':0,'--cont-batching':0,'--no-mmproj':0}
    if not d['unified']:optional['--no-kv-unified']=0
    if not d['ram_cache']:optional['--cache-ram']=1
    # Older builds have no fit switch; absence is not the same as explicit auto-fit.
    optional['--fit']=1
    if d['kv_type']=='f16':
        optional.update({'--cache-type-k':1,'--cache-type-v':1})
    if d.get('reasoning')=='high':optional['--chat-template-kwargs']=1
    explicit=recovery_options(d)
    for flag,n in optional.items():
        if flag in args and flag not in explicit and not help_has(h,flag):
            i=args.index(flag);del args[i:i+1+n];notes.append('미지원 부가 옵션 생략: '+flag)
    return args,notes

def supports(exe,args,env=None):
    h=engine_help(exe,env)
    missing=[]
    for a in args[1:]:
        if a.startswith('--') and not help_has(h,a):missing.append(a)
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
        if (kind=='benchmark' and task_busy('download')) or (kind=='download' and task_busy('benchmark')):
            raise ValueError('정확한 성능 측정을 위해 다운로드와 벤치마크를 동시에 실행할 수 없습니다.')
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

def build_current(raw):
    # Never fetch, reset, checkout or pull in this workflow.
    if not source_state(raw)['head']:raise ValueError('먼저 기존 llama.cpp 소스 폴더를 지정하세요.')
    build({**raw,'llama_ref':'','update_source':False,'fresh_build':True,'preserve_checkout':True})

def build(raw):
    w,l,m,x=paths(raw); jobs=num({'jobs':raw.get('build_jobs',BASE['build_jobs'])},'jobs',1,128); arch=str(raw.get('cuda_arch') or BASE['cuda_arch']); ref='' if raw.get('preserve_checkout') else str(raw.get('llama_ref') or BASE['llama_ref']); update=False if raw.get('preserve_checkout') else bool(raw.get('update_source'))
    if not re.fullmatch(r'\d+(?:[a-z])?(?:;\d+(?:[a-z])?)*',arch):raise ValueError('CUDA arch 예: 90, 89, 90;100')
    if ref.startswith('-'):raise ValueError('잘못된 git ref')
    def job():
        git=find_tool('git'); cm=find_tool('cmake'); tk=cuda_toolkit(raw); vs=vs_install()
        if not all((git,cm,tk)) or (WIN and not vs):raise RuntimeError('Git/CMake/CUDA Toolkit/MSVC C++ 환경을 먼저 설치하세요. Toolkit이 없으면 공식 바이너리 설치를 선택할 수 있습니다.')
        nv=tk['nvcc'];log('build',f"CUDA Toolkit {tk['version']} | {tk['root']}")
        l.parent.mkdir(parents=True,exist_ok=True)
        if not (l/'.git').exists():
            if raw.get('preserve_checkout'):raise RuntimeError('기존 Git 소스를 찾지 못했습니다. 경로를 확인하세요.')
            if l.exists() and any(l.iterdir()):raise RuntimeError('llama.cpp 경로에 Git 저장소가 아닌 파일이 있습니다.')
            step('build',[git,'clone',BASE['llama_repo'],str(l)])
        elif update:
            rc,out=run_capture([git,'-C',str(l),'status','--porcelain'])
            if rc or out.strip():raise RuntimeError('소스 변경사항이 있어 자동 업데이트를 중단합니다. 직접 확인하세요.')
            step('build',[git,'-C',str(l),'fetch','--tags','origin'])
            if not ref:step('build',[git,'-C',str(l),'pull','--ff-only'])
        if ref:step('build',[git,'-C',str(l),'checkout','--detach',ref])
        env=os.environ.copy();env['PATH']=str(Path(nv).parent)+os.pathsep+env.get('PATH','');env['CUDA_PATH']=tk['root'];env['CUDACXX']=nv
        source=source_state(raw)
        if not source['head']:raise RuntimeError('현재 Git 커밋을 확인하지 못했습니다.')
        bdir=l/('build-cuda-'+tk['version']+'-vs'+str(vs[1] if vs else 0)+'-sm'+arch.replace(';','_')+'-'+source['head'][:12])
        if source['dirty'] is not False:log('build','수정된 소스: 별도 폴더 사용')
        if raw.get('fresh_build') or source['dirty'] is not False:bdir=bdir.with_name(bdir.name+'-'+str(time.time_ns()))
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
        step('build',[str(exe),'--version'],cwd=exe.parent,env=env)
        current=source_state(raw)
        if current['head']!=source['head']:raise RuntimeError('빌드 중 소스 커밋이 변경되었습니다. 다시 빌드하세요.')
        (bdir/'easy-llm-build.json').write_text(json.dumps({'kind':'source','commit':source['head'],'clean':source['dirty'] is False and current['dirty'] is False,'cuda':tk['version']},indent=2),encoding='utf-8')
        set_task_meta('build',exe=str(exe),commit=source['head'],apply=bool(raw.get('preserve_checkout')))
        log('build','빌드 완료. 현재 체크아웃 빌드는 완료 후 GUI에서 자동 선택합니다. 기타 빌드는 실행 파일 적용 버튼을 누르세요.')
    start_task('build',job)


def clean_inference_env(env):
    # This manager owns inference flags. Preserve CUDA/driver/PATH, not inherited llama flags.
    for k in list(env):
        if k.upper().startswith('LLAMA_ARG_') or k.upper()=='LLAMA_API_KEY':env.pop(k,None)
    return env

def launch(raw):
    d,e,args=command_preview(raw)
    if d['host']=='0.0.0.0' and not d['auth']:raise ValueError('사내망 접속은 API 키 인증이 필수입니다. 키를 등록하고 다시 시작하세요.')
    if task_busy('benchmark'):raise ValueError('벤치마크 작업이 끝난 뒤 서버를 시작하세요.')
    if e['danger'] and not raw.get('force'):return {'confirm':True,'danger':e['danger']}
    exe=find_server(d);f=resolve_model(d)
    if not exe:raise ValueError('llama-server를 먼저 빌드하거나 실행 파일 경로를 지정하세요.')
    files=[str(a.resolve()) for a in check_shards(f)]
    try:md=gguf_metadata(f)['metadata']
    except (ValueError,OSError,struct.error) as exc:
        log('system','GGUF metadata 사전 검사 미완료: '+str(exc));md={}
    arch=md.get('general.architecture')
    if arch and d['profile']=='inkling2' and arch!='inkling2':raise ValueError('선택한 파일이 Inkling 2 언어 모델이 아닙니다: '+str(arch))
    if '--mmproj' in args:files.append(str(Path(args[args.index('--mmproj')+1]).resolve()))
    if d['spec_mode']=='draft-dspark':files+=list(map(str,check_shards(absolute(d['draft_path']))))
    if d['auth'] and not read_keys():raise ValueError('API 키를 먼저 생성하세요. API 키 탭에서 생성한 다음 실행하세요.')
    if not e['gpu_ids']:raise ValueError('사용할 NVIDIA GPU가 없습니다.')
    with socket.socket() as s:
        try:s.bind((d['host'],d['port']))
        except OSError:raise ValueError('서버 포트가 이미 사용 중이거나 열 수 없습니다.')
    env=speed.environment(clean_inference_env(cuda_env(d)),d);env['PATH']=str(exe.parent)+os.pathsep+env.get('PATH','')
    supports(str(exe),args,env)
    env['CUDA_VISIBLE_DEVICES']=','.join(map(str,e['gpu_ids']));
    # Do not inherit unexpected API keys if the UI requests no authentication.
    for k in ('LLAMA_API_KEY','LLAMA_ARG_API_KEY_FILE','LLAMA_ARG_MMPROJ','LLAMA_ARG_MMPROJ_URL','LLAMA_ARG_MMPROJ_AUTO'):env.pop(k,None)
    with LOCK:
        if d['auth'] and not read_keys():raise ValueError('등록된 API 키가 없습니다.')
        start_task('server',lambda:step('server',args,cwd=exe.parent,env=env),{'port':d['port'],'auth':d['auth'],'alias':d['alias'],'host':d['host'],'context':d['context'],'parallel':d['parallel'],'profile':d['profile'],'quant':d['quant'],'cpu_layers':d['cpu_layers'],'expert_cache':d['expert_cache'],'cpu_share':d['cpu_share'],'kv_type':d['kv_type'],'kv_location':d['kv_location'],'files':files,'recovery_options':d.get('recovery_options',''),'engine':str(exe),'vision':d['vision'],'batch':d['batch'],'ubatch':d['ubatch'],'settings':{k:d[k] for k in (*BASE['defaults'],'workspace','llama_dir','model_dir','server_exe','cuda_root') if k in d}})
    return {'started':True}


def read_keys():
    try:return [x.strip() for x in KEYS.read_text(encoding='utf-8').splitlines() if x.strip() and not x.startswith('#')]
    except OSError:return []


def key_action(action,index=None,expected=None):
    with LOCK:
        if task_busy('server') and TASKS['server'].get('meta',{}).get('auth'):
            raise ValueError('인증 서버 실행 중에는 키 변경이 잠깁니다. 서버를 중지하고 키를 변경한 뒤 다시 시작하세요.')
        keys=read_keys(); new=None
        if action=='add':new=secrets.token_urlsafe(32);keys.append(new)
        elif action=='remove':
            i=int(index)
            if not 0<=i<len(keys) or expected!=keys[i][:6]+'…'+keys[i][-4:]:
                raise ValueError('키 목록이 변경됐습니다. 새로고침 후 다시 시도하세요.')
            keys.pop(i)
        else:raise ValueError('키 작업 오류')
        temp=KEYS.with_suffix('.tmp')
        with os.fdopen(os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600),'w',encoding='utf-8') as f:
            f.write('\n'.join(keys)+'\n')
        if not WIN:temp.chmod(0o600)
        temp.replace(KEYS)
    return {'new':new,'keys':masked_keys()}

def local_addresses():
    # No internet discovery; local adapter addresses only.
    ips=set()
    if WIN:
        rc,out=run_capture(['ipconfig'],5)
        if rc==0:
            for line in out.splitlines():
                if 'IPv4' in line:
                    ips.update(re.findall(r'(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])',line))
    else:
        try:ips.update(x[4][0] for x in socket.getaddrinfo(socket.gethostname(),None,socket.AF_INET))
        except OSError:pass
    return sorted(x for x in ips if not x.startswith(('127.','169.254.','0.')))

def probe_access():
    with LOCK:
        t=TASKS.get('server',{})
        if not t.get('active'):raise ValueError('관리 중인 추론 서버를 먼저 실행하세요.')
        m=dict(t.get('meta',{}))
    port=m['port']; opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def probe(path,key=None):
        h={'Authorization':'Bearer '+key} if key else {}
        try:
            req=urllib.request.Request(f'http://127.0.0.1:{port}'+path,headers=h)
            with opener.open(req,timeout=3) as response:return response.status
        except urllib.error.HTTPError as exc:return exc.code
        except (OSError,urllib.error.URLError):return None
    keys=read_keys() if m.get('auth') else []
    return {'health':probe('/health'),'without_key':probe('/v1/models'),
            'with_key':probe('/v1/models',keys[0]) if keys else None,'auth':bool(m.get('auth'))}



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


def measure_server(port,prompt_tokens,out_tokens,auth=False,seed=123,timeout=14400):
    # Synthetic sample only; exact prompt token count from server tokenizer.
    sample='The robot recorded the experiment, analyzed the error, and described the next step.\n'
    ids=api_json(port,'/tokenize',{'content':sample,'add_special':False},auth,30).get('tokens',[])
    if not ids or not all(isinstance(x,int) for x in ids):raise RuntimeError('서버 tokenizer 응답이 올바르지 않습니다.')
    prompt=(ids*((prompt_tokens+len(ids)-1)//len(ids)))[:prompt_tokens]
    begin=time.monotonic();response=api_json(port,'/completion',{'prompt':prompt,'n_predict':out_tokens,'ignore_eos':True,'cache_prompt':False,'temperature':0.0,'seed':seed,'stream':False},auth,timeout)
    t=response.get('timings',{});result={k:t[k] for k in ('prompt_n','prompt_ms','prompt_per_second','predicted_n','predicted_ms','predicted_per_second') if isinstance(t.get(k),(int,float))}
    if result.get('predicted_n',0)<=0 or result.get('predicted_per_second',0)<=0:raise RuntimeError('서버가 생성 timing 데이터를 반환하지 않았습니다.')
    result['wall_seconds']=round(time.monotonic()-begin,3);result['requested_prompt']=prompt_tokens;result['requested_output']=out_tokens
    return result


def history():
    try:
        v=json.loads(BENCH_HISTORY.read_text(encoding='utf-8'));return v if isinstance(v,list) else []
    except (OSError,ValueError):return []


def record_bench(kind,meta,result):
    safe={k:meta[k] for k in ('profile','quant','context','parallel','cpu_layers','expert_cache','cpu_share','kv_type','kv_location','vision','batch','ubatch','engine') if k in meta}
    item={'date':time.strftime('%Y-%m-%d %H:%M:%S'),'kind':kind,'settings':safe,**result}
    with LOCK:
        recent=(history()+[item])[-50:];tmp=BENCH_HISTORY.with_suffix('.tmp');tmp.write_text(json.dumps(recent,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(BENCH_HISTORY)
    return item


def vision_probe(port,auth,alias):
    # Synthetic PNG; no user file or remote image request.
    def chunk(kind,body):
        return struct.pack('>I',len(body))+kind+body+struct.pack('>I',zlib.crc32(kind+body)&0xffffffff)
    pixels=(b'\x00'+b'\xff\x00\x00'*224)*224
    png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',224,224,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(pixels))+chunk(b'IEND',b'')
    url='data:image/png;base64,'+base64.b64encode(png).decode('ascii')
    payload={'model':alias,'messages':[{'role':'user','content':[{'type':'text','text':'What is the dominant color? Answer in one word.'},{'type':'image_url','image_url':{'url':url}}]}],'max_tokens':64,'temperature':0,'stream':False}
    response=api_json(port,'/v1/chat/completions',payload,auth,180)
    content=response['choices'][0]['message'].get('content','')
    return {'accepted':True,'correct':bool(re.search(r'\bred\b|빨강|빨간',str(content),re.I)),'answer':str(content)[:200]}

def benchmark(raw):
    with LOCK:
        t=TASKS.get('server',{});meta=dict(t.get('meta',{}))
        if not t.get('active') or not meta:raise ValueError('관리 중인 서버를 먼저 실행하세요.')
    if api_json(meta['port'],'/health',auth=meta['auth']).get('status')!='ok':raise ValueError('서버가 아직 준비되지 않았습니다.')
    if meta['parallel']!=1:raise ValueError('기존 사용자 캐시 및 부하에 영향을 주므로 단일 슬롯 서버에서만 벤치마크를 수행하세요.')
    n,m=bench_counts(raw,meta['context'])
    target=num({'v':raw.get('goal_target',10)},'v',1,1000,False) if raw.get('goal_verify') else None
    do_vision=bool(raw.get('goal_vision')) and bool(meta.get('settings',{}).get('vision'))
    if task_busy('benchmark'):raise ValueError('이미 성능 측정 중입니다.')
    def job():
        log('benchmark',f'합성 입력 {n} tokens, 생성 {m} tokens (cache_prompt=false)')
        result=measure_server(meta['port'],n,m,meta['auth'])
        if target is not None:
            result.update(goal_target=target,goal_pass=result['predicted_per_second']>=target,server_started=t.get('started'),context_configured=meta['context'])
            if do_vision:
                try:result['vision_probe']=vision_probe(meta['port'],meta['auth'],meta['alias'])
                except Exception as exc:result['vision_probe']={'accepted':False,'error':str(exc)}
        item=record_bench('goal' if target is not None else 'manual',meta,result);set_task_meta('benchmark',result=item)
        log('benchmark',json.dumps(result,ensure_ascii=False))
    start_task('benchmark',job)


def tune(raw):
    if any(task_busy(k) for k in ('build','download','server','benchmark','tune')):raise ValueError('다른 작업을 중지한 뒤 튜닝하세요.')
    d,_,_=validate(raw)
    goal=bool(raw.get('goal_tune'))
    if recovery_options(d):raise ValueError('자동 튜닝 전에 복구 옵션을 비워 주세요.')
    if not goal and (d['parallel']!=1 or d['unified']):raise ValueError('자동 튜닝은 단일 슬롯/비공유 KV 전용입니다.')
    if d['mode']!='hybrid' and not raw.get('allow_gpu_tune') and not goal:raise ValueError('GPU + CPU MoE 모드에서 실행하세요.')
    exe=find_server(d)
    if not exe:raise ValueError('llama-server 실행 파일이 필요합니다.')
    if not goal:check_shards(resolve_model(d))
    d={**d,'host':'127.0.0.1','auth':False,'ram_cache':0}
    target=None
    if goal:
        plan=recommend_setup(raw)
        if not plan['ok']:raise ValueError(plan['message'])
        if plan['settings']['parallel']!=1:raise ValueError('실측 자동 탐색은 단일 슬롯 전용입니다.')
        target=plan['target_tps']
        n,m=bench_counts(raw,plan['settings']['context'])
        choices=[plan['candidate'],*plan['alternatives']]
        limit=num({'v':raw.get('trial_limit',6)},'v',1,18)
        distinct=[];seen=set()
        for c in choices:
            k=c['settings']['quant']
            if k not in seen:distinct.append(c);seen.add(k)
        if plan.get('maximize_context'):
            # Reserve trials for lower-context and F16 fallback instead of repeated near-identical variants.
            ordered=choices[:max(1,limit-2)]
            f16=next((c for c in reversed(choices) if c['settings']['kv_type']=='f16' and c not in ordered),None)
            lower=next((c for c in reversed(choices) if c not in ordered and c!=f16),None)
            ordered += [c for c in (lower,f16) if c is not None]
            variants=[dict(c['settings']) for c in ordered[:limit]]
            variants.sort(key=lambda x:x['context'],reverse=True)
        else:
            ordered=distinct+[c for c in choices if c not in distinct]
            variants=[dict(c['settings']) for c in ordered[:limit]]
            variants.sort(key=lambda x:BASE['model_profiles'][x['profile']]['quants'][x['quant']]['gb'],reverse=True)
    else:
        n,m=bench_counts(raw,d['context'])
        plan=placement_plan(d);choices=plan['candidates']
        if not choices:raise ValueError(plan['message'])
        variants=[dict(c['settings']) for c in choices]
        if variants[-1]['expert_cache']:variants.append({**variants[-1],'cpu_share':0.0})
    # Keep requested context; no automatic quant/context reduction. Test only synthetic requests.
    def job():
        results=[]; best=None
        for i,variant in enumerate(variants):
            if TASKS['tune']['cancel']:break
            trial={**d,**variant}
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));trial['port']=sock.getsockname()[1]
            try:_,e,args=command_preview(trial)
            except (ValueError,OSError) as exc:log('tune','후보 설정 실패: '+str(exc));continue
            label=f"{i+1}/{len(variants)}: {trial['quant']} / ctx {trial['context']} / {trial['kv_type']} / {trial.get('load_mode','auto')} / CPU {trial['cpu_layers']}층, cache {trial['expert_cache']} MiB, CPU share {trial['cpu_share']}"
            log('tune',label)
            if e['danger']:
                log('tune','메모리 재검사 실패, 후보 건너뜀: '+'; '.join(e['danger']));continue
            env=speed.environment(clean_inference_env(cuda_env(trial)),trial);env['PATH']=str(exe.parent)+os.pathsep+env.get('PATH','');env['CUDA_VISIBLE_DEVICES']=','.join(map(str,e['gpu_ids']))
            for k in ('LLAMA_API_KEY','LLAMA_ARG_API_KEY','LLAMA_ARG_API_KEY_FILE','LLAMA_ARG_MMPROJ','LLAMA_ARG_MMPROJ_URL','LLAMA_ARG_MMPROJ_AUTO'):env.pop(k,None)
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
                trial_n=min(n,trial['context']-m-32)
                first=measure_server(trial['port'],trial_n,m,timeout=300 if goal else 14400)
                if TASKS['tune']['cancel']:break
                second=measure_server(trial['port'],trial_n,m,seed=124,timeout=300 if goal else 14400)
                packed={**second,'cold_decode_tps':first['predicted_per_second']}
                item=record_bench('tune',trial,packed);results.append(item)
                if goal:
                    v=None
                    if trial['vision']:
                        try:v=vision_probe(trial['port'],False,trial['alias'])
                        except Exception as exc:v={'accepted':False,'error':str(exc)}
                    passed=min(first['predicted_per_second'],second['predicted_per_second'])>=target and min(first.get('predicted_n',0),second.get('predicted_n',0))>=m and (not trial['vision'] or bool(v and v.get('accepted') and v.get('correct')))
                    item={**item,'goal_pass':passed,'goal_target':target,'vision_probe':v,'full_settings':smart_settings(trial)}
                    results[-1]=item
                    if passed:best=item
                    log('tune','목표 통과' if passed else '목표 속도/비전 시험 미달')
                elif best is None or item['predicted_per_second']>best['predicted_per_second']:best=item
                set_task_meta('tune',results=results,best=best)
                log('tune',f"측정 완료 | cold {first['predicted_per_second']:.2f} / warm {second['predicted_per_second']:.2f} t/s")
            except Exception as exc:
                log('tune','후보 실패 (다음 후보로 이동): '+str(exc))
            finally:
                if proc and proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                if reader:reader.join(timeout=5)
                if proc and proc.stdout:proc.stdout.close()
                with LOCK:TASKS['tune']['proc']=None
            if goal and best:break
            if not TASKS['tune']['cancel']:time.sleep(2)
        if not best and not TASKS['tune']['cancel']:raise RuntimeError('시험한 후보 중 목표를 충족한 후보가 없습니다. 로그와 결과를 확인하세요.' if goal else '성공한 튜닝 후보가 없습니다. 로그를 확인하세요.')
        if best:log('tune','추천은 테스트한 후보와 입력 길이 기준이며 사용자가 적용하기 전에는 설정을 변경하지 않습니다.')
    start_task('tune',job)

def speed_info(raw):
    exe=find_server(raw)
    if not exe:raise ValueError('설치된 llama-server.exe를 지정하세요.')
    env=clean_inference_env(cuda_env(raw));env['PATH']=str(exe.parent)+os.pathsep+env.get('PATH','')
    h=engine_help(str(exe),env)
    rc,v=run_capture([exe,'--version'],15,env)
    return {'engine':str(exe),'version':v.strip()[-1200:],'capabilities':speed.capabilities(h),
            'variants':[name for name,_ in speed.variants(raw,h,bool(raw.get('experimental')),bool(raw.get('include_draft')))]}

def speed_sample(port,workload,n,m,auth=False):
    # Distinct synthetic code-edit and explanation tasks. Never read user files or send external requests.
    if workload=='code':
        lines=[f'def sensor_{i}(raw, offset, gain):\\n    value = (raw - offset) * gain\\n    return max(-100.0, min(100.0, value))\\n' for i in range(max(40,n//22))]
        text='\\n'.join(lines)+'\\n# Rewrite the sensor_0 function with a validity check, preserving the original calibration algorithm.\\ndef sensor_0_checked(raw, offset, gain):\\n'
    else:
        lines=[f'Test {i}: controller gain {i%17+1}, load {i*7%43}, measured residual {(i*i+17)%113}. Review measurement uncertainty and disturbances. ' for i in range(max(40,n//22))]
        text=''.join(lines)+'\\nExplain how to distinguish noise from a systematic error, design an independent experiment, and justify the conclusion.\\nAnswer:'
    tokens=api_json(port,'/tokenize',{'content':text,'add_special':False},auth,60).get('tokens',[])
    if not tokens or not all(type(x) is int for x in tokens):raise RuntimeError('tokenizer 응답 오류')
    prompt=tokens[-n:]
    begin=time.monotonic()
    r=api_json(port,'/completion',{'prompt':prompt,'n_predict':m,'ignore_eos':True,'cache_prompt':False,
                                 'temperature':0.0,'seed':123,'stream':False},auth,300)
    timing=r.get('timings',{}); out=r.get('content')
    if not isinstance(out,str) or not out:raise RuntimeError('출력 본문이 반환되지 않았습니다.')
    result={k:timing[k] for k in ('prompt_n','prompt_ms','prompt_per_second','predicted_n','predicted_ms','predicted_per_second') if isinstance(timing.get(k),(int,float))}
    if result.get('predicted_n',0)<m or result.get('predicted_per_second',0)<=0:raise RuntimeError('요청한 생성 길이 또는 timing을 충족하지 못했습니다.')
    result.update(output_hash=hashlib.sha256(out.encode('utf-8')).hexdigest(),requested_prompt=len(prompt),requested_output=m,wall_seconds=round(time.monotonic()-begin,3))
    for k in ('draft_n','draft_n_accepted'):
        if isinstance(timing.get(k),int):result[k]=timing[k]
    return result

def speed_tune(raw):
    if any(task_busy(k) for k in ('build','download','server','benchmark','tune')):raise ValueError('서버와 다른 작업을 중지한 후 가속 시험을 시작하세요.')
    d,_,_=validate(raw)
    if d['parallel']!=1 or d['unified'] or d['mode']!='gpu' or d['kv_location']!='gpu' or d['cpu_layers'] or d['ram_cache']:
        raise ValueError('가속 시험은 GPU 전용, 단일 슬롯, 비공유 KV, RAM 캐시 0인 설정만 지원합니다.')
    if recovery_options(d):raise ValueError('자동 가속 시험 전 복구 JSON을 비우세요. mmap은 모델 로딩 방식에서 변경할 수 있습니다.')
    n,m=bench_counts({'bench_prompt':raw.get('bench_prompt',4096),'bench_output':raw.get('bench_output',128)},d['context'])
    target=num({'v':raw.get('speed_target',50)},'v',1,1000,False)
    exe=find_server(d)
    if not exe:raise ValueError('실행 파일을 지정하세요.')
    check_shards(resolve_model(d))
    env=clean_inference_env(cuda_env(d));env['PATH']=str(exe.parent)+os.pathsep+env.get('PATH','')
    h=engine_help(str(exe),env)
    variants=speed.variants(d,h,bool(raw.get('experimental')),bool(raw.get('include_draft')))
    if len(variants)<2:raise ValueError('현재 엔진에서 확인된 가속 옵션이 없습니다. 엔진 검사 또는 실험 옵션을 확인하세요.')
    allow_estimate=raw.get('allow_estimate') is True
    def job():
        rows=[];baseline=None;best=None
        set_task_meta('tune',speed_mode=True,results=rows,best=None,total=len(variants),goal=target)
        for i,(label,variant) in enumerate(variants):
            if TASKS['tune']['cancel']:break
            trial={**variant,'host':'127.0.0.1','auth':False}
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));trial['port']=sock.getsockname()[1]
            set_task_meta('tune',current=label,index=i+1)
            log('tune',f'{i+1}/{len(variants)} {label} | {trial["quant"]} / {trial["context"]} / {trial["kv_type"]}')
            proc=None;reader=None
            try:
                _,e,args=command_preview(trial)
                if not e['gpu_ids'] or e['gpu_weight']>e['vfree']:
                    raise RuntimeError('GPU 가중치 자체가 여유 VRAM을 초과합니다.')
                if e['danger'] and not allow_estimate:raise RuntimeError('추정 메모리 부족: '+ '; '.join(e['danger']))
                if e['danger']:log('tune','사용자 승인 후 추정 경고를 무시하고 실제 로딩 검증: '+'; '.join(e['danger']))
                trial_env=speed.environment(clean_inference_env(cuda_env(trial)),trial)
                trial_env['PATH']=str(exe.parent)+os.pathsep+trial_env.get('PATH','')
                trial_env['CUDA_VISIBLE_DEVICES']=','.join(map(str,e['gpu_ids']))
                supports(str(exe),args,trial_env)
                with LOCK:
                    if TASKS['tune']['cancel']:break
                    proc=subprocess.Popen(args,cwd=exe.parent,env=trial_env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',creationflags=CREATE)
                    TASKS['tune']['proc']=proc
                def drain():
                    for line in proc.stdout:log('tune',line)
                reader=threading.Thread(target=drain,daemon=True);reader.start()
                deadline=time.monotonic()+1800
                while True:
                    if TASKS['tune']['cancel'] or proc.poll() is not None:raise RuntimeError('모델 로딩 실패 또는 작업 중지')
                    if time.monotonic()>deadline:raise RuntimeError('초기화 시간 한도 초과 (30분)')
                    try:
                        if api_json(trial['port'],'/health').get('status')=='ok':break
                    except (OSError,ValueError):pass
                    time.sleep(2)
                samples={}
                for k in ('code','general'):
                    if TASKS['tune']['cancel']:raise RuntimeError('중지됨')
                    samples[k]=speed_sample(trial['port'],k,n,m)
                a=speed.assess(samples,baseline,target)
                if i==0:baseline=samples
                row={'label':label,**a,'samples':samples,'settings':{k:variant[k] for k in BASE['defaults'] if k in variant},'status':'ok'}
                if a['match'] and (best is None or a['score']>best['score']):best=row
                rows.append(row)
                log('tune',f'{label}: 코드 {a["rates"]["code"]:.2f}, 일반 {a["rates"]["general"]:.2f} t/s | 기준 출력 '+('일치' if a['match'] else '불일치 (추천 제외)'))
            except Exception as exc:
                rows.append({'label':label,'status':'error','error':str(exc)})
                log('tune','실패: '+str(exc))
                if i==0:raise RuntimeError('기준 설정 실패: 기존 정상 설정을 먼저 복원하세요.') from exc
            finally:
                if proc and proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                if reader:reader.join(timeout=5)
                if proc and proc.stdout:proc.stdout.close()
                with LOCK:TASKS['tune']['proc']=None
                set_task_meta('tune',results=list(rows),best=best)
            if not TASKS['tune']['cancel']:time.sleep(2)
        if best:
            log('tune','시험 결과는 합성 입력/greedy 출력 기준입니다. 실제 Claude Code 속도 및 품질은 별도 검증하세요.')
    start_task('tune',job)

def local_models(raw):
    p,_=profile(raw);rows=[]
    for q in p['quants']:
        d={**raw,'quant':q,'model_path':raw.get('model_path','') if q==raw['quant'] else ''}
        f=resolve_model(d);installed=False;size=None;error=''
        if f:
            try:
                shards=check_shards(f);size=sum(x.stat().st_size for x in shards)/GIB;installed=True
            except (ValueError,OSError) as exc:error=str(exc)
        rows.append({'quant':q,'installed':installed,'gib':size,'path':str(f or ''),'error':error})
    return rows

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

def startup_failure(lines):
    msgtext='\n'.join(lines[-300:]).lower()
    rules=[
        ('kv_compat',r'quantized v cache|cache type.*not supported|flash attention.*not supported|flash.attention.*requires|unsupported.*cache type','KV 정밀도/Flash Attention 호환성 오류 후보. 동일 목표에서 F16 후보를 먼저 검토하세요.'),
        ('model_compat',r'unknown projector|unknown model architecture|key not found in model|error loading model hyperparameters|error loading model architecture','엔진과 모델 또는 projector의 호환성 오류 후보. 메모리 설정만 바꿔도 해결되지 않을 수 있습니다.'),
        ('memory',r'out of memory|failed to allocate|cannot allocate|cuda error.*memory|cuda_malloc.*fail|not enough memory','메모리 할당 실패 증거가 있습니다. 동일 조건의 낮은 KV/작은 batch/CPU Expert 후보를 검토하세요.'),
        ('context',r'failed to create context|failed to initialize context|failed to initialize the context','컨텍스트 초기화 실패. 정확한 원인은 이 문구 앞의 로그에서 확인해야 합니다.'),
    ]
    for key,pattern,msg in rules:
        if re.search(pattern,msgtext):return {'kind':key,'message':msg}
    return {'kind':'unknown','message':'오류 원인을 특정할 수 없습니다. 실패 직전 로그와 실행 명령을 확인하세요.'}

def error_excerpt(lines):
    bad=[]
    for ln in lines:
        if re.search(r'error|failed|fatal|assert|exception|out of memory|invalid|자동 진단|종료 코드',ln,re.I) and not ln.startswith('> '):bad.append(ln)
    return bad[-15:] if bad else lines[-8:]

def record_last_ready(t,health):
    global LAST_READY
    try:
        if json.loads(health).get('status')!='ok':return
    except (ValueError,TypeError,AttributeError):return
    with LOCK:
        cur=TASKS.get('server',{})
        meta=t.get('meta',{})
        if not cur.get('active') or cur.get('started')!=t.get('started') or not meta.get('settings'):return
        if LAST_READY.get('started')==t['started']:return
        LAST_READY={'started':t['started'],'settings':dict(meta['settings']),'engine':meta.get('engine','')}
        try:
            tmp=LAST_READY_FILE.with_suffix('.tmp')
            tmp.write_text(json.dumps(LAST_READY,ensure_ascii=False,indent=2),encoding='utf-8')
            tmp.replace(LAST_READY_FILE)
        except OSError:log('system','마지막 정상 구동 설정을 디스크에 저장하지 못했습니다. 현재 세션에만 보관합니다.')

def memory_detail(pid):
    if not WIN:return {}
    from ctypes import wintypes as W
    out={}
    class PMC(ctypes.Structure):
        _fields_=[('cb',W.DWORD),('PageFaultCount',W.DWORD)]+[(k,ctypes.c_size_t) for k in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]
    class PI(ctypes.Structure):
        _fields_=[('cb',W.DWORD)]+[(k,ctypes.c_size_t) for k in ('CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable','SystemCache','KernelTotal','KernelPaged','KernelNonpaged','PageSize')]+[(k,W.DWORD) for k in ('HandleCount','ProcessCount','ThreadCount')]
    try:
        psapi=ctypes.WinDLL('psapi',use_last_error=True);kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        psapi.GetPerformanceInfo.argtypes=[ctypes.POINTER(PI),W.DWORD];psapi.GetPerformanceInfo.restype=W.BOOL
        pi=PI();pi.cb=ctypes.sizeof(pi)
        if psapi.GetPerformanceInfo(ctypes.byref(pi),pi.cb):
            out.update(commit=pi.CommitTotal*pi.PageSize/GIB,commit_limit=pi.CommitLimit*pi.PageSize/GIB,system_cache=pi.SystemCache*pi.PageSize/GIB)
        if pid:
            kernel.OpenProcess.argtypes=[W.DWORD,W.BOOL,W.DWORD];kernel.OpenProcess.restype=W.HANDLE
            kernel.CloseHandle.argtypes=[W.HANDLE];kernel.CloseHandle.restype=W.BOOL
            psapi.GetProcessMemoryInfo.argtypes=[W.HANDLE,ctypes.POINTER(PMC),W.DWORD];psapi.GetProcessMemoryInfo.restype=W.BOOL
            handle=kernel.OpenProcess(0x0400|0x0010,False,pid)
            if handle:
                try:
                    m=PMC();m.cb=ctypes.sizeof(m)
                    if psapi.GetProcessMemoryInfo(handle,ctypes.byref(m),m.cb):
                        out.update(working_set=m.WorkingSetSize/GIB,peak_working_set=m.PeakWorkingSetSize/GIB,private_commit=m.PrivateUsage/GIB)
                finally:kernel.CloseHandle(handle)
    except (OSError,AttributeError,ValueError):pass
    return out

def status():
    with LOCK:
        ts={k:{x:v for x,v in t.items() if x not in ('proc','cancel')}|{'pid':t['proc'].pid if t['proc'] else None} for k,t in TASKS.items()}; logs={k:list(v) for k,v in LOGS.items()}
    t=ts.get('server',{}); meta=t.get('meta',{}); health='';metrics='';slots=''
    if t.get('active') and meta:
        health=local_api(meta['port'],'/health');
        if health:
            metrics=local_api(meta['port'],'/metrics',meta['auth'])
            if meta.get('host')!='0.0.0.0':slots=local_api(meta['port'],'/slots',meta['auth'])
    record_last_ready(t,health)
    rt,rf=ram()
    # slots can contain prompt internals depending on build; only report selected counters.
    slim=[]
    try:
        for s in json.loads(slots):
            if isinstance(s,dict):slim.append({k:s[k] for k in ('id','is_processing','n_ctx','n_past','n_decoded','n_remaining','n_remain','n_prompt_tokens','n_prompt_tokens_processed','n_prompt_tokens_cached','n_cache_tokens') if k in s})
    except (ValueError,TypeError):pass
    return {'tasks':ts,'logs':logs,'gpu':gpus(),'ram_total':rt,'ram_free':rf,'health':health,'metrics':metrics,'slots':slim,'cache_stats':cache_stats(logs['server']),'history':history()[-10:],'server_errors':error_excerpt(logs['server']) if t.get('state')=='error' else [],'sample_time':time.time(),'key_count':len(read_keys()),'last_ready':LAST_READY,'memory_detail':memory_detail(t.get('pid')),'startup_failure':startup_failure(logs['server']) if t.get('state')=='error' else None}


def save_prefs(raw):
    global SETTINGS
    allowed=set(BASE['defaults'])|{'workspace','llama_dir','model_dir','server_exe','build_jobs','cuda_arch','llama_ref','cuda_root','runtime_cuda','runtime_tag'}
    prefs={k:v for k,v in raw.items() if k in allowed};validate({**BASE['defaults'],**prefs});recovery_options(prefs);
    with LOCK:
        tmp=PREF.with_suffix('.tmp');tmp.write_text(json.dumps(prefs,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(PREF);SETTINGS=prefs
    return {'ok':True}

HTML = (HERE/'ui.html').read_text(encoding='utf-8')


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
                w,l,m,x=paths();form={**BASE['defaults'],'workspace':str(w),'llama_dir':BASE['llama_dir'],'model_dir':BASE['model_dir'],'server_exe':BASE['server_exe'],'build_jobs':BASE['build_jobs'],'cuda_arch':BASE['cuda_arch'],'llama_ref':BASE['llama_ref'],'cuda_root':BASE.get('cuda_root',''),'runtime_cuda':BASE.get('runtime_cuda','12'),'runtime_tag':BASE.get('runtime_tag',''),**SETTINGS};return self.send(200,{'form':form,'model_profiles':BASE['model_profiles'],'last_ready':LAST_READY,'version':BASE.get('version','0.5')})
            if self.path=='/api/status':return self.send(200,status())
            if self.path=='/api/keys':return self.send(200,{'keys':masked_keys(),'locked':bool(task_busy('server') and TASKS['server'].get('meta',{}).get('auth'))})
            if self.path=='/api/addresses':return self.send(200,{'addresses':local_addresses()})
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
            elif action=='build_current':build_current(d)
            elif action=='engines':result=engine_inventory(d)
            elif action=='inspect_local_model':result=inspect_local_model(d)
            elif action=='install_binary':install_binary(d)
            elif action=='plan':result=placement_plan(d)
            elif action=='recommend':result=recommend_setup(d)
            elif action=='benchmark':benchmark(d)
            elif action=='tune':tune(d)
            elif action=='speed_info':result=speed_info(d)
            elif action=='speed_tune':speed_tune(d)
            elif action=='local_models':result={'models':local_models(d)}
            elif action=='start':result=launch(d)
            elif action=='stop':
                if raw.get('kind') not in ('download','server','build','tune'):raise ValueError('invalid task')
                stop_task(raw['kind'])
            elif action=='key':result=key_action(raw['action'],raw.get('index'),raw.get('expected'))
            elif action=='probe_access':result=probe_access()
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
