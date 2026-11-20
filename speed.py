"""v0.5 inference acceleration policy. No GPU or network side effects."""
import math
import re

DEFAULTS = {"spec_mode":"none","backend_sampling":False,"cuda_p2p":False,"split_mode":"layer","cuda_graphs":"auto","draft_path":"","draft_tokens":7}
KEYS = tuple(DEFAULTS)
LABELS = {"none":"기본", "ngram-mod":"반복 패턴 (Mod)", "ngram-simple":"반복 패턴 (Simple)", "draft-dspark":"DSpark"}
def has(help_text, flag):
    return bool(re.search(r"(?<![\w-])"+re.escape(flag)+r"(?![\w-])",help_text))

def capabilities(h):
    def enum(flag, value):
        # Some old help lists values on following lines; still require the flag itself.
        return has(h,flag) and has(h,value)
    return {"ngram_mod":enum("--spec-type","ngram-mod"),"ngram_simple":enum("--spec-type","ngram-simple"),
            "dspark":enum("--spec-type","draft-dspark") and any(has(h,x) for x in ("--spec-draft-model","--model-draft")),
            "sampling":has(h,"--backend-sampling"),"row":enum("--split-mode","row"),"tensor":enum("--split-mode","tensor"),
            "p2p":"experimental","graphs":"environment-only"}

def validated(d):
    x={**DEFAULTS,**{k:d[k] for k in KEYS if k in d}}
    if x["spec_mode"] not in LABELS or x["split_mode"] not in ("layer","row","tensor"):
        raise ValueError("가속 설정 값 오류")
    if x["cuda_graphs"] not in ("auto","off"):raise ValueError("CUDA Graph 설정 오류")
    for k in ("backend_sampling","cuda_p2p"):
        if type(x[k]) is not bool:raise ValueError("가속 스위치 오류: "+k)
    if not isinstance(x["draft_path"],str):raise ValueError("DSpark 파일 경로 오류")
    if type(x["draft_tokens"]) is not int or not 1<=x["draft_tokens"]<=16:raise ValueError("DSpark 예측 길이는 1~16")
    if x["split_mode"]=="tensor" and x["backend_sampling"]:raise ValueError("Tensor split과 Backend sampling은 함께 사용할 수 없습니다.")
    if x["spec_mode"]=="draft-dspark" and not x["draft_path"].strip():raise ValueError("DSpark GGUF 경로를 지정하세요.")
    return x

def options(d,h):
    x=validated(d); c=capabilities(h); args=[]
    if x["split_mode"]!="layer" and not c[x["split_mode"]]:raise ValueError("현재 엔진에 "+x["split_mode"]+" 분배가 없습니다.")
    if x["backend_sampling"]:
        if not c["sampling"]:raise ValueError("현재 엔진에 Backend sampling 옵션이 없습니다.")
        args+=["--backend-sampling"]
    mode=x["spec_mode"]
    if mode!="none":
        cap={"ngram-mod":"ngram_mod","ngram-simple":"ngram_simple","draft-dspark":"dspark"}[mode]
        if not c[cap]:raise ValueError("현재 엔진에 "+mode+" 지원이 확인되지 않습니다.")
        args+=["--spec-type",mode]
        if mode=="draft-dspark":
            flag="--spec-draft-model" if has(h,"--spec-draft-model") else "--model-draft"
            args += [flag,x["draft_path"]]
            if has(h,"--spec-draft-n-max"):args+=["--spec-draft-n-max",str(x["draft_tokens"])]
            elif has(h,"--draft-max"):args+=["--draft-max",str(x["draft_tokens"])]
    return args

def environment(env,d):
    x=validated(d)
    # Presence, not value, may enable/disable these switches.
    env={k:v for k,v in env.items() if k.upper() not in ("GGML_CUDA_P2P","GGML_CUDA_DISABLE_GRAPHS")}
    if x["cuda_p2p"]:env["GGML_CUDA_P2P"]="1"
    if x["cuda_graphs"]=="off":env["GGML_CUDA_DISABLE_GRAPHS"]="1"
    return env

def variants(d,h,experimental=False, include_draft=False):
    """Compare against an explicit baseline; never alter context, quant, KV, or batch."""
    c=capabilities(h)
    neutral={**d,**DEFAULTS,"draft_path":d.get("draft_path",""),"draft_tokens":d.get("draft_tokens",7)}
    out=[("기준 설정",neutral)]
    for mode,cap in (("ngram-mod","ngram_mod"),("ngram-simple","ngram_simple")):
        if c[cap]:out.append((LABELS[mode],{**neutral,"spec_mode":mode}))
    if c["sampling"]:out.append(("GPU 샘플링",{**neutral,"backend_sampling":True}))
    if c["ngram_mod"] and c["sampling"]:out.append(("Mod + GPU 샘플링",{**neutral,"spec_mode":"ngram-mod","backend_sampling":True}))
    if c["ngram_simple"] and c["sampling"]:out.append(("Simple + GPU 샘플링",{**neutral,"spec_mode":"ngram-simple","backend_sampling":True}))
    if include_draft and c["dspark"] and neutral["draft_path"]:out.append(("DSpark",{**neutral,"spec_mode":"draft-dspark"}))
    if experimental:
        out.append(("PCIe P2P (실험)",{**neutral,"cuda_p2p":True}))
        for mode in ("row","tensor"):
            if c[mode]:out.append((mode+" 분배 (실험)",{**neutral,"split_mode":mode}))
        out.append(("CUDA Graph 비활성 비교",{**neutral,"cuda_graphs":"off"}))
    return out

def assess(samples,baseline,target):
    """Two workload rates; compare deterministic output hashes to the baseline.
    This is a regression smoke test, not a general model-quality certification."""
    if set(samples)!={"code","general"}:raise ValueError("두 입력 유형의 측정이 필요합니다.")
    rates={k:float(v["predicted_per_second"]) for k,v in samples.items()}
    if any(not math.isfinite(v) or v<=0 for v in rates.values()):raise ValueError("잘못된 속도 측정")
    if any(not v.get("output_hash") or v.get("predicted_n",0)<v["requested_output"] for v in samples.values()):
        raise ValueError("출력 부족 또는 검증 데이터 없음")
    same=baseline is None or all(samples[k]["output_hash"]==baseline[k]["output_hash"] for k in samples)
    score=2/sum(1/v for v in rates.values())
    return {"score":score,"rates":rates,"match":same,"goal_pass":same and min(rates.values())>=target}
