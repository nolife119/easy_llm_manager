"""Local-only Windows llama.cpp manager. Python 3.12, standard library UI."""
from __future__ import annotations
import ctypes, fnmatch, json, os, re, secrets, shutil, socket, subprocess, sys, threading, time, traceback, urllib.request, urllib.error, webbrowser
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
LOGS = {k: deque(maxlen=900) for k in ('download','build','server','system')}
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
    for p in (l/'build/bin/Release/llama-server.exe',l/'build/bin/llama-server.exe',l/'build/bin/llama-server',l/'llama-server.exe'):
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
    return {'items':items,'gpu':gs,'ram_total':rt,'ram_free':rf,'disk':disk(m),'paths':{'workspace':str(w),'llama_dir':str(l),'model_dir':str(m),'server_exe':str(x or find_server(d) or '')},'python':sys.version.split()[0]}


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


def estimate(raw):
    d,p,q=validate(raw); gs=selected_gpus(d,gpus()); rt,rf=ram(); warnings=[]; danger=[]
    f=resolve_model(d); weight=q['gb']*1e9/GIB
    if f and f.exists():
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
    if d['expert_cache']:args+=['--moe-cache-mib',str(d['expert_cache']),'--moe-cache-cpu-share',str(d['cpu_share'])]
    if p.get('reasoning_template'):args+=['--temp','1.0','--top-p','1.0','--top-k','0','--min-p','0.0','--chat-template-kwargs',json.dumps({'reasoning_effort':d['reasoning']})]
    if d['vision']:
        projector=p.get('projector'); candidates=[repo_path(d)/projector,f.parent/projector,f.parent.parent/projector] if f and projector else ([repo_path(d)/projector] if projector else [])
        proj=next((x for x in candidates if x.is_file()),None)
        if not proj:raise ValueError('Vision projector 파일을 찾지 못했습니다. Vision 다운로드 옵션을 선택하세요.')
        args+=['--mmproj',str(proj)]
    if d['host']=='0.0.0.0' and not d['auth']:args+=['--no-slots']
    if d['auth']:args+=['--api-key-file',str(KEYS)]
    return d,e,args


def supports(exe,args):
    rc,h=run_capture([exe,'--help'],30)
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


def start_task(kind,func,meta=None):
    with LOCK:
        if task_busy(kind):raise ValueError('이미 실행 중입니다.')
        if (kind=='server' and task_busy('build')) or (kind=='build' and task_busy('server')):raise ValueError('빌드와 서버 실행을 동시에 할 수 없습니다.')
        TASKS[kind]={'active':True,'state':'running','started':time.time(),'proc':None,'cancel':False,'meta':meta or {}}
    def worker():
        try:func(); state='stopped' if TASKS[kind]['cancel'] else 'done';log(kind,'완료' if state=='done' else '중지됨')
        except Exception as e:
            state='stopped' if TASKS[kind]['cancel'] else 'error';log(kind,str(e));log(kind,'진단 탭과 README의 문제 해결 항목을 확인하세요.')
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
    if not inspect and task_busy('server'):raise ValueError('모델 파일 변경을 방지하기 위해 서버를 중지한 뒤 다운로드하세요.')
    d,p,q=validate(raw); base=repo_path(d); base.mkdir(parents=True,exist_ok=True)
    patterns=[q['pattern']]+([p['projector']] if d['vision'] and p.get('projector') else [])
    # API (not a version-dependent CLI module entry point); reruns reuse local metadata and partial cache.
    script='''import json,sys\nfrom huggingface_hub import HfApi,snapshot_download\nfrom fnmatch import fnmatch\nfrom pathlib import Path\nimport shutil\na=json.loads(sys.argv[1]); info=HfApi().model_info(a['repo'],revision=a['revision'],files_metadata=True); files=[s for s in info.siblings if any(fnmatch(s.rfilename,g) for g in a['patterns'])]; assert files, 'No files matched the configured pattern'; print('Revision:',info.sha,flush=True); print('\\n'.join(f'{s.rfilename} | {s.size} bytes' for s in files),flush=True); print('Total bytes:',sum(s.size or 0 for s in files),flush=True)\nif not a['inspect']:\n    dst=Path(a['dest']); need=sum(s.size or 0 for s in files if not ((dst/s.rfilename).is_file() and (dst/s.rfilename).stat().st_size==s.size)); free=shutil.disk_usage(dst).free; print('Remaining file budget (conservative):',need,'Free bytes:',free,flush=True)\n    if need>free: raise RuntimeError('Not enough disk space for remaining full files (partial cache not subtracted)')\n    snapshot_download(repo_id=a['repo'],revision=info.sha,local_dir=a['dest'],allow_patterns=a['patterns'],max_workers=2)\n'''
    a=dict(repo=p['repo'],revision=p.get('revision','main'),patterns=patterns,dest=str(base),inspect=inspect)
    def job():
        rc,_=run_capture([sys.executable,'-c','import huggingface_hub'])
        if rc:raise RuntimeError('huggingface_hub 없음. 환경 탭에서 설치하세요.')
        step('download',[sys.executable,'-u','-c',script,json.dumps(a,ensure_ascii=False)])
    start_task('download',job)


def install_hf():start_task('download',lambda:step('download',[sys.executable,'-m','pip','install','--upgrade','--only-binary=:all:','huggingface_hub']))


def build(raw):
    w,l,m,x=paths(raw); jobs=num({'jobs':raw.get('build_jobs',BASE['build_jobs'])},'jobs',1,128); arch=str(raw.get('cuda_arch') or BASE['cuda_arch']); ref=str(raw.get('llama_ref') or BASE['llama_ref']); update=bool(raw.get('update_source'))
    if not re.fullmatch(r'\d+(?:[a-z])?(?:;\d+(?:[a-z])?)*',arch):raise ValueError('CUDA arch 예: 90, 89, 90;100')
    if ref.startswith('-'):raise ValueError('잘못된 git ref')
    def job():
        git=find_tool('git'); cm=find_tool('cmake'); nv=find_tool('nvcc'); vs=vs_install()
        if not all((git,cm,nv)) or (WIN and not vs):raise RuntimeError('Git/CMake/CUDA/MSVC C++ 환경을 먼저 설치하세요.')
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
        env=os.environ.copy();env['PATH']=str(Path(nv).parent)+os.pathsep+env.get('PATH','');env['CUDA_PATH']=str(Path(nv).parent.parent)
        args=[cm,'-S',str(l),'-B',str(l/'build'),'-DGGML_CUDA=ON','-DCMAKE_CUDA_ARCHITECTURES='+arch,'-DLLAMA_BUILD_TESTS=OFF','-DCMAKE_BUILD_TYPE=Release']
        if WIN:
            vsroot,major=vs; gen={17:'Visual Studio 17 2022',18:'Visual Studio 18 2026'}.get(major)
            if not gen:raise RuntimeError(f'지원되지 않는 Visual Studio 버전 {major}; CMake 생성기를 직접 확인하세요.')
            args+=['-G',gen,'-A','x64','-DCMAKE_GENERATOR_INSTANCE='+str(vsroot)]
        step('build',args,env=env);step('build',[cm,'--build',str(l/'build'),'--config','Release','--parallel',str(jobs),'--target','llama-server'],env=env)
        exe=find_server(raw)
        if not exe:raise RuntimeError('빌드 후 실행 파일을 찾지 못했습니다.')
        step('build',[str(exe),'--version'])
    start_task('build',job)


def launch(raw):
    d,e,args=command_preview(raw)
    if task_busy('download'):raise ValueError('다운로드/파일 검사 작업이 끝난 뒤 서버를 시작하세요.')
    if e['danger'] and not raw.get('force'):return {'confirm':True,'danger':e['danger']}
    exe=find_server(d);f=resolve_model(d)
    if not exe:raise ValueError('llama-server를 먼저 빌드하거나 실행 파일 경로를 지정하세요.')
    check_shards(f)
    if d['auth'] and not read_keys():raise ValueError('API 키를 먼저 생성하세요.')
    if not e['gpu_ids']:raise ValueError('사용할 NVIDIA GPU가 없습니다.')
    with socket.socket() as s:
        try:s.bind((d['host'],d['port']))
        except OSError:raise ValueError('서버 포트가 이미 사용 중이거나 열 수 없습니다.')
    supports(str(exe),args)
    env=os.environ.copy();nv=find_tool('nvcc')
    if nv:env['PATH']=str(Path(nv).parent)+os.pathsep+env.get('PATH','')
    env['CUDA_VISIBLE_DEVICES']=','.join(map(str,e['gpu_ids']));
    # Do not inherit unexpected API keys if the UI requests no authentication.
    for k in ('LLAMA_API_KEY','LLAMA_ARG_API_KEY_FILE'):env.pop(k,None)
    start_task('server',lambda:step('server',args,cwd=exe.parent,env=env),{'port':d['port'],'auth':d['auth'],'alias':d['alias'],'host':d['host']})
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
    return {'tasks':ts,'logs':logs,'gpu':gpus(),'ram_total':rt,'ram_free':rf,'health':health,'metrics':metrics,'slots':slim}


def save_prefs(raw):
    global SETTINGS
    allowed=set(BASE['defaults'])|{'workspace','llama_dir','model_dir','server_exe','build_jobs','cuda_arch','llama_ref'}
    prefs={k:v for k,v in raw.items() if k in allowed};validate({**BASE['defaults'],**prefs});
    with LOCK:
        tmp=PREF.with_suffix('.tmp');tmp.write_text(json.dumps(prefs,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(PREF);SETTINGS=prefs
    return {'ok':True}

HTML = r'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Inkling Manager</title><style>
:root{color-scheme:dark;--bg:#10151d;--panel:#19222e;--line:#344253;--fg:#edf2f7;--muted:#a5b5c7;--accent:#68d6b4;--warn:#f4be72}*{box-sizing:border-box}body{background:var(--bg);color:var(--fg);font:14px/1.55 'Segoe UI',system-ui,sans-serif;margin:0}header{padding:22px max(24px,calc((100vw - 1260px)/2));background:#151e29;border-bottom:1px solid var(--line)}h1{margin:0;font-size:25px;letter-spacing:-.5px}h2{font-size:18px;margin:0 0 14px}h3{font-size:15px;margin:15px 0 8px}p{margin:8px 0;color:var(--muted)}main{max-width:1260px;margin:auto;padding:18px 24px 50px}nav{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 18px}button,a.btn{border:1px solid var(--line);border-radius:7px;background:#253445;color:var(--fg);padding:9px 13px;font:inherit;cursor:pointer;text-decoration:none}button:hover{border-color:var(--accent)}button.active,button.primary{background:var(--accent);color:#0b211b;border-color:var(--accent);font-weight:650}button:disabled{opacity:.4;cursor:not-allowed}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(235px,1fr));gap:12px}.cols{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);gap:14px}.panel{border:1px solid var(--line);background:var(--panel);border-radius:10px;padding:18px;margin-bottom:14px}.field{display:flex;flex-direction:column;gap:5px;margin:0 0 9px;min-width:0}.field>span{color:#c4d0df;font-size:13px}.field small{color:var(--muted)}input,select{width:100%;padding:9px 10px;background:#111923;color:var(--fg);border:1px solid var(--line);border-radius:6px;font:inherit;min-width:0}input[type=checkbox]{width:auto;margin-right:8px}label.check{display:flex;align-items:center;padding:8px 0}input:focus,select:focus{outline:1px solid var(--accent)}.buttons{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}.good{color:var(--accent)}.bad{color:#ff9992}.warn{color:var(--warn)}.muted{color:var(--muted)}.metric{background:#121b25;padding:13px;border-radius:8px;border:1px solid #2b3949}.metric b{display:block;font-size:23px;line-height:1.4;font-variant-numeric:tabular-nums}.bar{height:9px;background:#354254;border-radius:6px;overflow:hidden;margin:7px 0}.bar i{display:block;height:100%;background:var(--accent)}pre{background:#0c131c;border-radius:7px;padding:13px;white-space:pre-wrap;overflow-wrap:anywhere;max-height:360px;overflow:auto;font:12px/1.55 Consolas,monospace;margin:8px 0}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid var(--line);padding:8px;text-align:left;overflow-wrap:anywhere}th{color:var(--muted)}a{color:#9cc9ff}.hidden{display:none!important}#toast{position:fixed;bottom:18px;right:18px;max-width:min(600px,95vw);background:#2c4252;border:1px solid #597a8d;border-radius:8px;padding:12px 16px;z-index:9;white-space:pre-wrap}#toast:empty{display:none}.lead{font-size:13px;color:var(--muted)}@media(max-width:850px){.cols{grid-template-columns:1fr}main{padding:12px}.panel{padding:13px}} </style></head><body>
<header><h1>Inkling Manager <span class="lead"> / Windows local inference</span></h1><p>모델 다운로드부터 빌드, GPU 배치, API 서버 운영까지. 관리 화면은 이 PC의 localhost에서만 접근 가능.</p></header><main><nav id="nav"></nav><div class="grid" id="summary"></div>
<section id="env" class="tab"><div class="cols"><div class="panel"><h2>환경 진단</h2><p>먼저 GPU와 필수 도구를 확인하세요. 설치 직후 PATH가 갱신되지 않았다면 이 도구를 다시 실행하세요.</p><div class="buttons"><button class="primary" onclick="diagnose()">환경 다시 검사</button><button onclick="act('install_hf',{})">HF 다운로드 도구 설치</button></div><div id="diagnosis"></div></div><div class="panel"><h2>저장 경로</h2><p>기존 AI 폴더 또는 여유 공간이 큰 고정 디스크를 탐색합니다. 필요하면 직접 수정하세요.</p><div id="pathFields"></div><div class="buttons"><button onclick="save()">경로/세팅 저장</button></div><p>모델, 빌드 결과는 작업 폴더에 저장합니다. 개인 설정과 API 키는 사용자 전용 데이터 폴더에 저장합니다.</p></div></div></section>
<section id="download" class="tab hidden"><div class="cols"><div class="panel"><h2>모델 다운로드</h2><div id="modelFields"></div><div id="modelSpec"></div><div class="buttons"><button onclick="act('inspect')">원격 파일/실제 용량 조회</button><button class="primary" onclick="downloadModel()">다운로드 / 이어받기</button><button onclick="stop('download')">중지</button></div><p>같은 repo·revision·폴더로 다시 실행하면 완료 파일 및 HF가 보존한 부분 캐시를 재사용합니다. 파일을 임의로 이동하지 마세요.</p></div><div class="panel"><h2>다운로드 로그</h2><pre id="downloadLog"></pre><p>서버 구동 전 첫 shard와 모든 분할 파일의 존재/헤더를 검사합니다.</p></div></div></section>
<section id="build" class="tab hidden"><div class="cols"><div class="panel"><h2>llama.cpp CUDA 빌드</h2><p>기존 소스를 재사용하고 없다면 공식 Git 저장소에서 clone합니다. Visual Studio C++ 빌드 도구와 CUDA Toolkit이 필요합니다.</p><div id="buildFields"></div><label class="check"><input type="checkbox" id="update_source">기존 소스 업데이트 (변경사항이 없을 때만)</label><div class="buttons"><button class="primary" onclick="act('build')">컴파일 시작</button><button onclick="stop('build')">중지</button></div><p>H100은 CUDA arch 90. Git ref를 지정하면 해당 버전으로 빌드합니다. 최초 빌드는 오래 걸릴 수 있습니다.</p></div><div class="panel"><h2>빌드 로그</h2><pre id="buildLog"></pre><p>오류 시 마지막 로그와 환경 진단을 확인하세요. 기존 CMake 생성기와 충돌한다면 소스를 보존하고 build 폴더만 이름을 변경한 후 다시 빌드하세요.</p></div></div></section>
<section id="run" class="tab hidden"><div class="cols"><div class="panel"><h2>추론 서버 설정</h2><div class="buttons"><button onclick="preset('safe')">첫 구동 (2 GPU)</button><button onclick="preset('long')">1M / GPU KV</button><button onclick="preset('host')">1M / RAM KV</button><button onclick="preset('team')">15인 공유 KV</button></div><p>프리셋은 출발점일 뿐이며 GPU 장수와 선택 모델에 따라 조정하세요. 먼저 예상치를 계산하세요.</p><h3>모델 · 배치</h3><div id="runModel"></div><div class="grid" id="placement"></div><h3>컨텍스트 · 캐시</h3><div class="grid" id="contexts"></div><h3>엔진 · 네트워크</h3><div class="grid" id="advanced"></div><div class="buttons"><button onclick="save()">세팅 저장</button><button onclick="preview()">예상 자원 / 명령어</button><button class="primary" onclick="startServer()">서버 시작</button><button onclick="stop('server')">서버 중지</button></div></div><div><div class="panel"><h2>실행 전 메모리 예측</h2><div id="estimate"><p>설정을 선택한 후 예상 자원을 계산하세요.</p></div><p>GiB 단위. 실제 llama.cpp가 할당하는 메모리는 다를 수 있습니다.</p><h3>생성 명령어</h3><pre id="command"></pre></div><div class="panel"><h2>접속</h2><div id="connection"></div><p>로컬 브라우저에서 먼저 테스트한 후 사내 IP를 사용하세요. Claude Code 확장은 Anthropic Messages API 호환 여부를 별도 확인하세요.</p></div></div></div></section>
<section id="monitor" class="tab hidden"><div class="panel"><h2>실시간 GPU · 서버 상태</h2><div id="gpuCards" class="grid"></div><div id="taskCards" class="grid"></div><div class="buttons"><button onclick="stop('server')">관리 중인 서버 중지</button><button onclick="poll()">즉시 갱신</button></div><div id="health"></div></div><div class="cols"><div class="panel"><h2>서버 로그 (최근 900줄)</h2><label class="check"><input type="checkbox" id="follow" checked>자동 스크롤</label><pre id="serverLog" style="max-height:560px"></pre></div><div class="panel"><h2>API /metrics · /slots</h2><div id="rates" class="grid"></div><pre id="metrics"></pre><pre id="slots"></pre></div></div></section>
<section id="auth" class="tab hidden"><div class="panel"><h2>API 접근 키</h2><p>기본값은 인증 없음 + localhost. 사내 네트워크에서 서비스할 때만 승인된 범위에 노출하고 API 키 사용을 권장합니다. 키 활성화는 실행 설정에서 별도로 선택하세요. 키 추가/회수 후 재시작이 필요합니다.</p><div class="buttons"><button class="primary" onclick="key('add')">새 API 키 생성</button></div><div id="newKey"></div><div id="keyList"></div><p>키는 사용자 전용 디렉터리의 api-keys.txt에 저장되며 Git에 포함되지 않습니다. 사용자별 요청량 할당/회계는 제공하지 않습니다.</p></div></section></main><div id="toast" role="status"></div>
<script>
const csrf=__TOKEN__; let config, form={}, prev=null, polling=false, current='env'; const $=x=>document.getElementById(x); const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); const fmt=x=>x==null?'—':Number(x).toFixed(1); let toastTimer;
function toast(x){$('toast').textContent=x;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').textContent='',10000)}
async function api(path,body){const opt={headers:{'X-Manager-Token':csrf}};if(body!==undefined){opt.method='POST';opt.headers['Content-Type']='application/json';opt.body=JSON.stringify(body)} const r=await fetch('/api/'+path,opt);let j=await r.json();if(!r.ok)throw Error(j.error||r.status);return j}
const tabs=[['env','1 환경'],['download','2 다운로드'],['build','3 컴파일'],['run','4 서버 실행'],['monitor','5 모니터링'],['auth','6 API 키']]; $('nav').innerHTML=tabs.map(([k,t])=>`<button id="nav_${k}" onclick="tab('${k}')">${t}</button>`).join('');function tab(k){current=k;for(const [x] of tabs){$(x).classList.toggle('hidden',x!==k);$('nav_'+x).classList.toggle('active',x===k)}if(k==='auth')keys();if(k==='run')connection()}
function field(k,label,type='text',choices=null,help=''){let v=form[k]??'';let inner=type==='check'?`<label class="check"><input data-k="${k}" type="checkbox" ${v?'checked':''}>${esc(label)}</label>`:type==='select'?`<label class="field"><span>${esc(label)}</span><select data-k="${k}">${choices.map(([x,t])=>`<option value="${esc(x)}" ${String(x)===String(v)?'selected':''}>${esc(t)}</option>`).join('')}</select>`:`<label class="field"><span>${esc(label)}</span><input data-k="${k}" type="${type}" ${type==='number'?'step="any"':''} value="${esc(v)}">`;return inner+(type==='check'?'':(help?`<small>${esc(help)}</small>`:'')+'</label>')}
function wire(){document.querySelectorAll('[data-k]').forEach(el=>el.onchange=()=>{const k=el.dataset.k;form[k]=el.type==='checkbox'?el.checked:el.type==='number'?Number(el.value):el.value;if(k==='profile'){form.quant=Object.keys(config.model_profiles[form.profile].quants)[0];form.model_path='';render()}else if(k==='quant'){form.model_path='';render()}else{document.querySelectorAll(`[data-k="${k}"]`).forEach(x=>{if(x!==el){if(x.type==='checkbox')x.checked=form[k];else x.value=form[k]}});connection()}})}
function models(){let p=config.model_profiles[form.profile];return field('profile','모델 계열','select',Object.entries(config.model_profiles).map(([k,v])=>[k,v.label]))+field('quant','양자화','select',Object.entries(p.quants).map(([k,v])=>[k,`${k} · ${v.gb} GB`]))+field('vision','Vision projector 포함','check')}
function render(){let p=config.model_profiles[form.profile],q=p.quants[form.quant];$('pathFields').innerHTML=field('workspace','작업 폴더')+field('llama_dir','llama.cpp 소스 폴더','text',null,'비워 두면 작업폴더/llama.cpp')+field('model_dir','모델 폴더','text',null,'비워 두면 작업폴더/models')+field('server_exe','기존 llama-server.exe 직접 지정','text',null,'비워 두면 자동 탐색');$('modelFields').innerHTML=models();$('runModel').innerHTML=models()+field('model_path','첫 GGUF 파일 직접 지정 (선택)','text',null,'자동 감지에 실패했거나 기존 다운로드를 사용하려면 지정');$('modelSpec').innerHTML=`<div class="grid"><div class="metric">파일 용량<b>${q.gb} GB</b><small>${fmt(q.gb*1e9/1073741824)} GiB</small></div><div class="metric">원본 top-token 일치율<b>${q.agreement==null?'—':q.agreement+'%'}</b><small>품질 벤치마크 점수 아님</small></div></div><p>최대 ${p.max_context.toLocaleString()} 토큰 · <a target="_blank" rel="noreferrer" href="https://huggingface.co/${esc(p.repo)}">모델 카드</a></p>`;$('buildFields').innerHTML=field('cuda_arch','CUDA Architecture')+field('build_jobs','병렬 빌드 작업 수','number')+field('llama_ref','Git ref / 태그 / 커밋 (선택)','text',null,'비워 두면 현재 소스 유지, 최초 clone은 기본 브랜치');$('placement').innerHTML=field('mode','가중치 모드','select',[['gpu','GPU 전체 상주'],['hybrid','GPU + CPU MoE']])+field('cpu_layers','CPU MoE 레이어 수','number',null,'앞 N개 레이어 Expert만 RAM에 저장')+field('expert_cache','동적 GPU Expert cache (MiB)','number',null,'0 = 끔 / 최신 빌드 기능')+field('cpu_share','캐시 miss CPU 연산 비율 (0~1)','number')+field('gpu_ids','GPU 번호 (예: 0,1)','text',null,'비워 두면 모든 GPU')+field('split','GPU 배치 비율 (예: 1,1)','text',null,'비워 두면 여유 VRAM 비례');$('contexts').innerHTML=field('context','슬롯당 최대 컨텍스트','number')+field('parallel','동시 슬롯 수','number')+field('unified','공유 KV 풀 사용','check')+field('pool','전체 공유 KV 풀 토큰','number',null,'공유 모드에서만 사용')+field('kv_location','활성 KV 위치','select',[['gpu','GPU (권장)'],['cpu','RAM / CPU (느릴 수 있음)']])+field('kv_type','KV 정밀도','select',[['f16','FP16 (호환성 우선)'],['q8_0','Q8_0'],['q4_0','Q4_0 (실험)']])+field('ram_cache','RAM prompt cache 상한 (MiB)','number',null,'0 = 끔; 유휴 슬롯/프롬프트 보관')+field('runtime_reserve','GPU 기타 여유 공간 (GiB)','number')+field('ram_reserve','RAM 기타 여유 공간 (GiB)','number');$('advanced').innerHTML=field('batch','logical batch','number')+field('ubatch','physical ubatch','number')+field('threads','CPU 연산 스레드','number')+field('reasoning','Reasoning effort','select',['none','minimal','low','medium','high','xhigh','max'].map(x=>[x,x]))+field('alias','API 모델 이름')+field('host','접속 범위','select',[['127.0.0.1','이 서버에서만'],['0.0.0.0','사내망 접속 (승인 필요)']])+field('port','API 포트','number')+field('auth','API 키 인증 사용','check');wire();connection()}
function preset(k){if(k==='safe')Object.assign(form,{mode:'hybrid',cpu_layers:8,expert_cache:0,cpu_share:0,context:32768,pool:32768,parallel:1,unified:false,kv_location:'gpu',kv_type:'f16',ram_cache:0,batch:512,ubatch:128});if(k==='long')Object.assign(form,{context:1048576,pool:1048576,parallel:1,unified:false,kv_location:'gpu',kv_type:'q8_0',ram_cache:0});if(k==='host')Object.assign(form,{mode:'gpu',context:1048576,pool:1048576,parallel:1,unified:false,kv_location:'cpu',kv_type:'f16',ram_cache:0});if(k==='team')Object.assign(form,{mode:'gpu',parallel:15,context:1048576,pool:1572864,unified:true,kv_location:'gpu',kv_type:'q8_0',ram_cache:65536});render();preview()}
async function guard(fn){try{return await fn()}catch(e){toast(e.message);return null}}
async function act(name,data=form){return guard(async()=>{let r=await api(name,{...data,update_source:$('update_source').checked});toast('요청 접수: '+name);poll();return r})}
async function stop(kind){if(confirm(kind+' 작업을 중지할까요?'))await act('stop',{kind})}
async function save(){await guard(async()=>{await api('save',form);toast('사용자 설정 저장 완료')})}
async function downloadModel(){let q=config.model_profiles[form.profile].quants[form.quant];if(confirm(`${form.quant} 약 ${q.gb} GB를 다운로드/이어받기 할까요? 실제 크기는 로그에서 확인됩니다.`))await act('download')}
async function diagnose(){await guard(async()=>{let r=await api('diagnose',form);$('diagnosis').innerHTML=`<p>Python ${esc(r.python)} · 모델 디스크 여유 ${fmt(r.disk.free)} GiB</p><table><tr><th>항목</th><th>상태 / 경로</th></tr>${r.items.map(x=>`<tr><td>${esc(x.name)}</td><td><span class="${x.ok?'good':'bad'}">${x.ok?'정상':'확인 필요'}</span> <a href="${esc(x.link)}" target="_blank" rel="noreferrer">설치/안내</a><div class="muted">${esc(x.detail)}</div></td></tr>`).join('')}</table><pre>${esc(JSON.stringify(r.paths,null,2))}</pre>`;toast('환경 검사 완료')})}
function metric(k,v,h=''){return `<div class="metric"><small>${esc(k)}</small><b>${esc(v)}</b><small>${esc(h)}</small></div>`}
async function preview(){return guard(async()=>{let r=await api('preview',form),e=r.estimate;$('estimate').innerHTML=`<div class="grid">${metric('예상 VRAM',fmt(e.gpu)+' GiB','여유 '+fmt(e.vfree))}${metric('예상 RAM 예산',fmt(e.host_ram)+' GiB','여유 '+fmt(e.ram_free))}${metric('GPU 상주 가중치',fmt(e.gpu_weight)+' GiB','CPU '+fmt(e.cpu_weight))}${metric('활성 KV',fmt(e.kv)+' GiB','SWA 미절약 시 '+fmt(e.kv_worst))}${metric('KV pool',e.pool.toLocaleString(),'tokens')}${metric('전체 가중치',fmt(e.weight)+' GiB','디스크 여유 '+fmt(e.archive_free))}</div><p>GPU별 추정: ${e.gpu_allocation.map((x,i)=>`GPU ${e.gpu_ids[i]}: ${fmt(x)} GiB`).join(' / ')||'—'}</p><p>모델: ${esc(e.model||'아직 다운로드 안 됨 / 경로 미지정')}</p>${e.danger.map(x=>`<p class="bad">⚠ ${esc(x)}</p>`).join('')}${e.warnings.map(x=>`<p class="warn">• ${esc(x)}</p>`).join('')}`;$('command').textContent=r.command;return r})}
async function startServer(){let pr=await preview();if(!pr)return;const dangers=pr.estimate.danger;if(dangers.length&&!confirm('위험 설정입니다. 그래도 진행할까요?\n\n'+dangers.join('\n')))return;await guard(async()=>{let r=await api('start',{...form,force:!!dangers.length});if(r.confirm){if(!confirm(r.danger.join('\n')+'\n그래도 진행할까요?'))return;await api('start',{...form,force:true})}toast('서버 실행 요청 완료');tab('monitor');poll()})}
function connection(meta=null){if(!config)return;let p=meta?.port||form.port,host=meta?.host||form.host,alias=meta?.alias||form.alias;let base=`http://127.0.0.1:${p}`;$('connection').innerHTML=`<a href="${base}" target="_blank" rel="noreferrer">llama.cpp 채팅 UI 열기</a><pre>OpenAI: ${base}/v1\nAnthropic: ${base}\nModel: ${esc(alias)}\n${host==='0.0.0.0'?'원격 노트북은 127.0.0.1 대신 실제 서버 IP 사용':'현재 로컬 전용'}</pre>`}
function logs(id,items,follow=true){const x=$(id),v=items.join('\n');if(x.textContent!==v){x.textContent=v;if(follow)x.scrollTop=x.scrollHeight}}
function rateMetrics(m){let out=[];for(const ln of m.split('\n')){if(ln.startsWith('#'))continue;const z=ln.match(/^([^\s{]+)(?:\{[^}]*\})?\s+([\d.eE+\-]+)$/);if(z&&/tokens_per_second|requests_processing|requests_deferred|kv_cache_usage|predicted_tokens_total|prompt_tokens_total/.test(z[1]))out.push(metric(z[1].replace(/^llamacpp:/,''),fmt(Number(z[2]))))}return out.slice(0,12).join('')}
async function poll(){if(polling||!config)return;polling=true;try{let s=await api('status');prev=s;let st=s.tasks.server;let total=s.gpu.reduce((a,g)=>a+g.total,0),used=s.gpu.reduce((a,g)=>a+g.used,0);$('summary').innerHTML=metric('GPU VRAM',fmt(used)+' / '+fmt(total)+' GiB',`${s.gpu.length} GPU`)+metric('RAM 여유',fmt(s.ram_free)+' GiB','전체 '+fmt(s.ram_total)+' GiB')+metric('추론 서버',st?.active?(s.health?'응답 중':'시작/로딩 중'):(st?.state||'미실행'),st?.pid?'PID '+st.pid:'');$('gpuCards').innerHTML=s.gpu.map(g=>`<div class="metric"><small>GPU ${g.index} · ${esc(g.name)}</small><b>${fmt(g.used)} / ${fmt(g.total)} GiB</b><div class="bar"><i style="width:${Math.min(100,g.used/g.total*100)}%"></i></div><small>GPU ${fmt(g.util)}% · ${fmt(g.temp)}°C · ${fmt(g.power)}W</small></div>`).join('')||'<p>GPU 미감지</p>';$('taskCards').innerHTML=['download','build','server'].map(k=>metric(k,s.tasks[k]?.state||'idle',s.tasks[k]?.pid?'PID '+s.tasks[k].pid:'')).join('');$('health').textContent=s.health||'서버 health 응답 없음';logs('downloadLog',s.logs.download);logs('buildLog',s.logs.build);logs('serverLog',s.logs.server,$('follow').checked);$('metrics').textContent=s.metrics||'metrics 응답 없음';$('slots').textContent=JSON.stringify(s.slots,null,2);$('rates').innerHTML=rateMetrics(s.metrics);if(st?.active)connection(st.meta)}catch(e){$('health').textContent=e.message}finally{polling=false}}
async function keys(){await guard(async()=>drawKeys((await api('keys')).keys))}function drawKeys(a){$('keyList').innerHTML=a.length?`<table><tr><th>키</th><th>관리</th></tr>${a.map(x=>`<tr><td>${esc(x.mask)}</td><td><button onclick="key('remove',${x.index})">회수</button></td></tr>`).join('')}</table>`:'<p>등록된 키 없음</p>'}async function key(action,index){if(action==='remove'&&!confirm('이 키를 회수할까요?'))return;await guard(async()=>{let r=await api('key',{action,index});drawKeys(r.keys);if(r.new){$('newKey').innerHTML='<p class="warn">새 키 (한 번만 표시 · 안전하게 보관)</p><pre></pre><button onclick="navigator.clipboard.writeText(this.previousElementSibling.textContent)">복사</button>';$('newKey').querySelector('pre').textContent=r.new}else $('newKey').innerHTML='';toast('키 변경 완료; 서버 재시작 후 적용')})}
(async()=>{try{config=await api('config');form=config.form;render();tab('env');diagnose();poll();setInterval(poll,3000)}catch(e){toast(e.message)}})();
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
                w,l,m,x=paths();form={**BASE['defaults'],'workspace':str(w),'llama_dir':BASE['llama_dir'],'model_dir':BASE['model_dir'],'server_exe':BASE['server_exe'],'build_jobs':BASE['build_jobs'],'cuda_arch':BASE['cuda_arch'],'llama_ref':BASE['llama_ref'],**SETTINGS};return self.send(200,{'form':form,'model_profiles':BASE['model_profiles']})
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
            elif action=='save':result=save_prefs(d)
            elif action=='preview':
                _,e,args=command_preview(d);result={'estimate':e,'command':subprocess.list2cmdline(args)}
            elif action=='download':download(d)
            elif action=='inspect':download(d,True)
            elif action=='install_hf':install_hf()
            elif action=='build':build(d)
            elif action=='start':result=launch(d)
            elif action=='stop':
                if raw.get('kind') not in ('download','server','build'):raise ValueError('invalid task')
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
        for k in ('server','download','build'):stop_task(k)
        srv.server_close()

if __name__=='__main__':main()
