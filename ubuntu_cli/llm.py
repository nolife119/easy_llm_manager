#!/usr/bin/env python3
"""Offline-first Linux llama.cpp CLI. Python 3.12+, HF Hub only for downloads."""
import argparse, datetime, json, os, pathlib, re, shlex, shutil, subprocess, sys, tarfile, time, urllib.request, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
STATE=ROOT/"state"
CONFIG=ROOT/"config.json"
DEFAULT={"engine":"","archive":"","model_dir":"models","repo":"unsloth/Inkling-2-Mini-GGUF","revision":"main","layout":"unsloth","quant":"UD-Q4_K_XL","vision":False,"dspark":False,"context":8192,"parallel":1,"gpu_layers":99,"cache_k":"f16","cache_v":"f16","batch":512,"ubatch":128,"threads":26,"draft_tokens":2,"host":"127.0.0.1","port":8888,"alias":"inkling-2-mini","extra_args":[]}
DEFAULT.update({"model_path":"","projector_path":"","draft_path":"","scan_dirs":[],"download_files":[],"download_projector":"","download_draft":""})
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+".tmp"); t.write_text(json.dumps(d,indent=2,ensure_ascii=False)+"\n"); t.replace(p)
def read(p): return json.loads(p.read_text())
def config(): 
    if not CONFIG.exists(): save(CONFIG,DEFAULT)
    return {**DEFAULT,**read(CONFIG)}
def path(s):
    p=Path(os.path.expandvars(s)).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT/p).resolve()
def fail(s): raise RuntimeError(s)
def engine(c):
    if c["engine"]: return path(c["engine"])
    x=shutil.which("llama-server")
    if x: return Path(x).resolve()
    fail("Run setup first or set engine in config.json")
def env(c):
    e=os.environ.copy(); p=engine(c).parent
    libs=[str(p),str(p/"lib"),str(p.parent/"lib")]
    e["PATH"]=str(p)+":"+e.get("PATH","")
    e["LD_LIBRARY_PATH"]=":".join(libs+([e["LD_LIBRARY_PATH"]] if e.get("LD_LIBRARY_PATH") else []))
    return e
def probe(c,flag):
    p=subprocess.run([str(engine(c)),flag],env=env(c),capture_output=True,text=True,timeout=60)
    out=p.stdout+p.stderr
    if p.returncode: fail(out)
    return out
def doctor(c):
    print(probe(c,"--version"))
    d=probe(c,"--list-devices"); print(d)
    if not re.search(r"CUDA\d|NVIDIA",d,re.I): fail("No CUDA GPU detected")
    if shutil.which("nvidia-smi"): subprocess.run(["nvidia-smi"],check=False)
    print("Engine/device probe OK; model kernels still require smoke test.")
def setup(c,archive=None):
    if not archive and c["archive"]: archive=c["archive"]
    if not archive:
        candidates=sorted(set(ROOT.glob("*.tar.gz"))|set(ROOT.parent.glob("*cuda*.tar.gz"))|set((ROOT/"archives").glob("*.tar.gz")))
        candidates=[p for p in candidates if "linux" in p.name and "cuda" in p.name]
        if len(candidates)!=1: fail("Specify archive: setup /path/to/linux-cuda.tar.gz; found "+str(candidates))
        archive=str(candidates[0])
    a=path(archive); dest=ROOT/"runtime"/a.name.removesuffix(".tar.gz")
    if not dest.exists():
        dest.parent.mkdir(parents=True,exist_ok=True)
        import tempfile
        with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
            t=Path(tmp)
            with tarfile.open(a,"r:*") as tar: tar.extractall(t,filter="data")
            xs=list(t.rglob("llama-server"))
            if len(xs)!=1: fail("Archive must contain one llama-server")
            t.rename(dest)
    xs=list(dest.rglob("llama-server"))
    if len(xs)!=1: fail("Cannot uniquely locate llama-server")
    c["engine"]=str(xs[0]); c["archive"]=str(a)
    doctor(c)
    save(CONFIG,c)
    (ROOT/"env.sh").write_text("export PATH="+shlex.quote(str(xs[0].parent))+':"$PATH"\nexport LD_LIBRARY_PATH='+shlex.quote(":".join([str(xs[0].parent),str(xs[0].parent/"lib"),str(xs[0].parent.parent/"lib")]))+':"${LD_LIBRARY_PATH:-}"\n')
    print("Saved engine path; optional: . ./env.sh")
def files(c,all_assets=False):
    if c["download_files"]:
        return c["download_files"]+([c["download_projector"]] if (c["vision"] or all_assets) and c["download_projector"] else [])+([c["download_draft"]] if (c["dspark"] or all_assets) and c["download_draft"] else [])
    prefix="Mainline_Support/" if c["layout"]=="mainline" else ""
    if c["layout"] not in ("mainline","unsloth"): fail("Unknown layout")
    return [prefix+"Inkling-2-Mini-"+c["quant"]+".gguf"]+([prefix+"mmproj-F16.gguf"] if c["vision"] or all_assets else [])+([prefix+"dspark-Inkling-2-Mini-Q8_0.gguf"] if c["dspark"] or all_assets else [])
def manifest_path(c): return path(c["model_dir"])/".verified.json"
def manifest(c):
    p=manifest_path(c); return read(p) if p.exists() else {}
def verified(p,record):
    if not p.is_file() or not record: return False
    st=p.stat()
    with p.open("rb") as f: magic=f.read(4)
    return magic==b"GGUF" and st.st_size==record["size"] and st.st_mtime_ns==record["mtime_ns"]
def key(c,f): return c["repo"]+"@"+c["revision"]+":"+f
def status(c,all_assets=False):
    m=manifest(c); ok=True
    for f in files(c,all_assets):
        p=path(c["model_dir"])/f
        valid=verified(p,m.get(key(c,f)))
        print(("OK" if valid else "MISSING/UNVERIFIED"),f, p.stat().st_size if p.exists() else 0)
        ok &= valid
    return ok
def download(c,all_assets=False):
    from huggingface_hub import HfApi,hf_hub_download
    api=HfApi(); root=path(c["model_dir"]); root.mkdir(parents=True,exist_ok=True)
    m=manifest(c); info=api.model_info(c["repo"],revision=c["revision"],files_metadata=True)
    entries={s.rfilename:s for s in info.siblings}
    for f in files(c,all_assets):
        if f not in entries: fail("Not in repository: "+f+"; check layout/quant")
        s=entries[f]; p=root/f; k=key(c,f)
        if verified(p,m.get(k)) and m[k].get("commit")==info.sha: print("Already verified",f); continue
        for attempt in range(3):
            try:
                p=Path(hf_hub_download(c["repo"],f,revision=info.sha,local_dir=root))
                if s.size is None or p.stat().st_size!=s.size: fail("Size mismatch: "+f)
                with p.open("rb") as stream:
                    if stream.read(4)!=b"GGUF": fail("Invalid GGUF magic")
                sha=getattr(s.lfs,"sha256",None) if s.lfs else None
                if sha:
                    print("Checking SHA256",f,flush=True)
                    with p.open("rb") as stream: actual=hashlib.file_digest(stream,"sha256").hexdigest()
                    if actual!=sha: fail("SHA256 mismatch; move corrupt file aside and retry")
                m[k]={"size":s.size,"mtime_ns":p.stat().st_mtime_ns,"commit":info.sha,"sha256":sha}
                save(manifest_path(c),m); break
            except Exception as e:
                print("Download attempt",attempt+1,str(e),file=sys.stderr)
                if attempt==2: raise
                time.sleep(3*(attempt+1))
    if not status(c,all_assets): fail("Download verification failed")
def gguf_ok(p):
    try:
        with p.open("rb") as f:
            return p.stat().st_size>=24 and f.read(4)==b"GGUF"
    except OSError: return False
def auxiliary(p):
    n=p.name.lower()
    return "mmproj" in n or "dspark" in n or "dflash" in n or "eagle3" in n or n.startswith("draft")
def shards(p):
    m=re.fullmatch(r"(.*)-(\d{5})-of-(\d{5})\.gguf",p.name,re.I)
    if not m: return [p]
    prefix,idx,total=m.groups()
    if not 1<=int(total)<=1000 or idx!="00001": fail("Select first shard: "+str(p))
    return [p.with_name(f"{prefix}-{i:05d}-of-{int(total):05d}.gguf") for i in range(1,int(total)+1)]
def scan(c):
    roots=[path(c["model_dir"]),ROOT.parent/"models",Path.home()/"models",Path.home()/"AI"/"models",Path(os.environ.get("HF_HUB_CACHE",str(Path(os.environ.get("HF_HOME",str(Path.home()/".cache/huggingface")))/"hub")))]
    roots.extend(path(x) for x in c["scan_dirs"])
    found={}
    for root in roots:
        if not root.is_dir(): continue
        for p in root.rglob("*.gguf"):
            if auxiliary(p): continue
            try:
                if all(gguf_ok(x) for x in shards(p)): found.setdefault(str(p.absolute()),p)
            except RuntimeError: pass
    return sorted(found.values(),key=lambda p:str(p))
def list_models(c):
    xs=scan(c)
    for i,p in enumerate(xs,1):
        print(f"{i:3d} {sum(x.stat().st_size for x in shards(p))/1e9:7.2f} GB  {p}")
    if not xs: print("No local GGUF found; add scan_dirs or download.")
    return xs
def select_model(c,value):
    if not value or value.isdecimal():
        xs=list_models(c)
        if not xs: fail("No model found")
        n=int(value or input("Model number: "))
        if not 1<=n<=len(xs): fail("Invalid model number")
        p=xs[n-1]
    else: p=path(value)
    if auxiliary(p) or not all(gguf_ok(x) for x in shards(p)): fail("Invalid or incomplete model/shards")
    c.update(model_path=str(p),projector_path="",draft_path="",vision=False,dspark=False,alias=p.stem)
    save(CONFIG,c)
    print("Selected",p)
    print("Local header/shard check only; full integrity not verified. Run smoke.")
def local_assets(c):
    if c["model_path"]: p=path(c["model_path"])
    else:
        f=files(c)[0]; p=path(c["model_dir"])/f
        if not gguf_ok(p):
            xs=scan(c)
            if len(xs)==1: p=xs[0]
            elif len(xs)>1: fail("Multiple local models; run models then select NUMBER")
    if auxiliary(p) or not all(gguf_ok(x) for x in shards(p)): fail("Missing or invalid GGUF/shards; download or select a model")
    def asset(field,pattern):
        if c[field]: q=path(c[field])
        else:
            xs=[x for x in p.parent.glob("*.gguf") if pattern in x.name.lower() and gguf_ok(x)]
            if len(xs)!=1: fail("Set "+field+" explicitly; no unique companion in model directory")
            q=xs[0]
        if not gguf_ok(q): fail("Invalid companion "+str(q))
        return q
    return p,asset("projector_path","mmproj") if c["vision"] else None,asset("draft_path","dspark") if c["dspark"] else None
def configure(c,key,value):
    if key not in DEFAULT: fail("Unknown config key: "+key)
    old=DEFAULT[key]
    if isinstance(old,bool):
        if value.lower() not in ("true","false","on","off"): fail("Use true/false")
        value=value.lower() in ("true","on")
    elif isinstance(old,int):
        if key=="context" and re.fullmatch(r"\d+[kKmM]",value): value=int(value[:-1])*(1024 if value[-1].lower()=="k" else 1048576)
        else: value=int(value)
        if value<1 and key!="gpu_layers": fail("Expected positive integer")
    elif isinstance(old,list): value=json.loads(value)
    c[key]=value; save(CONFIG,c); print(key,"=",value)

def command(c):
    model,projector,draft=local_assets(c)
    print("Local GGUF:",model,"(header/shard check; use download/status for remote verification)")
    opts={"--model":str(model),"--ctx-size":c["context"],"--parallel":c["parallel"],"--n-gpu-layers":c["gpu_layers"],"--cache-type-k":c["cache_k"],"--cache-type-v":c["cache_v"],"--batch-size":c["batch"],"--ubatch-size":c["ubatch"],"--threads":c["threads"],"--flash-attn":"on","--host":c["host"],"--port":c["port"],"--alias":c["alias"],"--temp":1.0,"--top-p":1.0,"--top-k":0,"--min-p":0.0}
    if c["vision"]: opts["--mmproj"]=str(projector)
    if c["dspark"]: opts.update({"--model-draft":str(draft),"--spec-type":"draft-dspark","--spec-draft-n-max":c["draft_tokens"],"--n-gpu-layers-draft":99})
    if c["host"] not in ("127.0.0.1","localhost"): fail("Use approved port forwarding; default CLI only binds loopback")
    helptext=probe(c,"--help")
    flags=list(opts)+["--jinja","--metrics"]
    if not c["vision"] and "--no-mmproj" in helptext: flags.append("--no-mmproj")
    for flag in flags:
        if not re.search(r"(?<![\w-])"+re.escape(flag)+r"(?![\w-])",helptext): fail("Engine missing option "+flag)
    if c["dspark"] and "draft-dspark" not in helptext: fail("Engine does not advertise DSpark")
    extra=c["extra_args"]
    if not isinstance(extra,list) or not all(isinstance(x,str) for x in extra): fail("extra_args must be a list of strings")
    return [str(engine(c))]+[str(x) for pair in opts.items() for x in pair]+["--jinja","--metrics"]+(["--no-mmproj"] if "--no-mmproj" in flags else [])+extra
def now(): return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
def request(c,route,data=None,timeout=300):
    url="http://127.0.0.1:"+str(c["port"])+route
    req=urllib.request.Request(url,data=json.dumps(data).encode() if data is not None else None,headers={"Content-Type":"application/json"})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=timeout) as r: return json.load(r)
def run(c,smoke=False):
    if smoke: c={**c,"context":8192,"parallel":1,"vision":False,"dspark":False,"cache_k":"f16","cache_v":"f16","batch":512,"ubatch":128,"extra_args":[]}
    cmd=command(c); d=STATE/"runs"/now(); d.mkdir(parents=True)
    save(d/"config.json",c); save(d/"command.json",cmd)
    save(d/"engine.json",{"version":probe(c,"--version"),"devices":probe(c,"--list-devices")})
    print(shlex.join(cmd)); print("Logs:",d,flush=True)
    try:
        request(c,"/health",timeout=1)
    except Exception: pass
    else: fail("Server already responding on this port")
    with (d/"server.log").open("w") as log:
        p=subprocess.Popen(cmd,env=env(c),stdout=log,stderr=subprocess.STDOUT)
        ready=False; result={}
        try:
            deadline=time.monotonic()+1800
            while p.poll() is None and time.monotonic()<deadline:
                try:
                    if request(c,"/health",timeout=2).get("status")=="ok": ready=True; break
                except Exception: pass
                time.sleep(2)
            if not ready: fail("Health check failed; see "+str(d/"server.log"))
            if not smoke: save(STATE/"last-ready.json",c)
            save(STATE/"active.json",{"run":str(d),"pid":p.pid})
            print("Health OK. Ctrl+C stops server. tail -f",d/"server.log",flush=True)
            if smoke:
                result=request(c,"/completion",{"prompt":"Count from one to ten:","n_predict":32,"temperature":0,"cache_prompt":False})
                save(d/"smoke.json",result); print(result.get("timings"))
            else: p.wait()
        finally:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:p.kill();p.wait()
            save(d/"result.json",{"ready":ready,"exit_code":p.returncode,"smoke":bool(result)})
            active=STATE/"active.json"
            if active.exists() and read(active).get("run")==str(d): active.unlink()
    if not smoke and p.returncode: fail("Server exited: "+str(p.returncode))
def bench(c,n):
    active=read(STATE/"active.json"); d=Path(active["run"]); c=read(d/"config.json")
    t=time.monotonic()
    out=request(c,"/completion",{"prompt":"Explain how a PID controller works with an example in Python.","n_predict":n,"temperature":0,"cache_prompt":False})
    data={"seconds":time.monotonic()-t,"timings":out.get("timings"),"requested_tokens":n,"config":c}
    save(d/("bench-"+now()+".json"),data); print(json.dumps(data,indent=2))
def preset(name):
    if not re.fullmatch(r"[A-Za-z0-9_-]+",name): fail("Invalid preset name")
    return STATE/"presets"/(name+".json")
def main():
    p=argparse.ArgumentParser(); p.add_argument("action",nargs="?",default="menu",choices=["menu","setup","doctor","download","status","run","smoke","bench","save","load","restore","presets","history","show","init","models","select","set"]); p.add_argument("value",nargs="?"); p.add_argument("setting",nargs="?"); p.add_argument("--all",action="store_true"); a=p.parse_args(); c=config()
    if a.action=="menu":
        while True:
            print("\n1 Setup  2 Doctor  3 Download all  4 Status  5 Smoke  6 Run  7 Save preset  8 Load preset  9 Restore last ready  10 Models/select  11 Set config  0 Exit")
            x=input("> ").strip()
            if x=="0": return
            try:
                if x=="1": setup(config(),input("Archive path (blank=auto): ").strip() or None)
                elif x=="2": doctor(config())
                elif x=="3": download(config(),True)
                elif x=="4": status(config(),True)
                elif x=="5": run(config(),True)
                elif x=="6": run(config())
                elif x=="7": save(preset(input("Name: ")),config())
                elif x=="8": save(CONFIG,read(preset(input("Name: "))))
                elif x=="9": save(CONFIG,read(STATE/"last-ready.json"))
                elif x=="10": select_model(config(),None)
                elif x=="11": configure(config(),input("Key: "),input("Value: "))
            except (Exception,KeyboardInterrupt) as e: print("Stopped:",e,file=sys.stderr)
    elif a.action=="models": list_models(c)
    elif a.action=="select": select_model(c,a.value)
    elif a.action=="set": configure(c,a.value or fail("Key required"),a.setting or fail("Value required"))
    elif a.action=="setup": setup(c,a.value)
    elif a.action=="doctor": doctor(c)
    elif a.action=="download": download(c,a.all)
    elif a.action=="status": return 0 if status(c,a.all) else 1
    elif a.action in ("run","smoke"): run(c,a.action=="smoke")
    elif a.action=="bench": bench(c,int(a.value or 128))
    elif a.action=="save": save(preset(a.value or fail("Name required")),c)
    elif a.action=="load":
        name=a.value or fail("Name required")
        builtins={"safe": {"context":8192,"cache_k":"f16","cache_v":"f16","vision":False,"dspark":False,"batch":512,"ubatch":128}, "128k":{"context":131072,"cache_k":"q8_0","cache_v":"q8_0","vision":False,"dspark":False,"batch":2048,"ubatch":512},"512k":{"context":524288,"cache_k":"q8_0","cache_v":"q8_0","vision":True,"dspark":False,"batch":2048,"ubatch":512},"512k-dspark":{"context":524288,"cache_k":"q8_0","cache_v":"q8_0","vision":True,"dspark":True,"batch":2048,"ubatch":512}}
        save(CONFIG,{**c,**builtins[name]} if name in builtins else read(preset(name)))
    elif a.action=="restore": save(CONFIG,read(STATE/"last-ready.json"))
    elif a.action=="presets": print("Built-in: safe, 128k, 512k, 512k-dspark, max-context\n"+"\n".join(x.stem for x in sorted((STATE/"presets").glob("*.json"))))
    elif a.action=="history": print("\n".join(str(x) for x in sorted((STATE/"runs").glob("*"))))
    elif a.action in ("show","init"): print(json.dumps(c,indent=2))
    return 0
if __name__=="__main__":
    try: sys.exit(main())
    except KeyboardInterrupt: sys.exit(130)
    except Exception as e: print("ERROR:",e,file=sys.stderr); sys.exit(1)
