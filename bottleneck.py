"""Local synthetic diagnosis; no source code, real prompts, tokens or secrets are logged."""
import json
import math
import statistics
import time
import urllib.request

TOOL = {"type":"function","function":{"name":"lookup_test","description":"Lookup a hypothetical test measurement.","parameters":{"type":"object","properties":{"test_id":{"type":"integer"}},"required":["test_id"],"additionalProperties":False}}}

def cases(max_context, long_tokens=32768, long_output=1024, full=False):
    # One-factor changes relative to B0; chat tools are compared with chat, not raw.
    n=4096
    if max_context < n+128+2048:raise ValueError("4K 진단을 위해 컨텍스트 여유가 필요합니다.")
    big=min(long_tokens,max_context-128-2048)
    out=min(long_output,max_context-n-2048)
    base={"id":"B0","label":"기준 (4K / greedy / raw)","n":n,"out":128,"temp":0.0,"stream":False,"chat":False,"tools":False,"parent":None}
    def case(id,label,parent=base,**kw):return {**parent,"id":id,"label":label,"parent":parent["id"],**kw}
    sample=case("S","확률 샘플링",temp=1.0)
    stream=case("R","스트리밍",stream=True)
    length=case("L","긴 출력",out=out)
    context=case("C","긴 입력",n=big)
    chat=case("H","채팅 템플릿",chat=True)
    tools=case("T","툴 스키마 포함",chat,tools=True)
    combined=case("A","Claude Code 유사 (합성)",tools,n=big,out=out,temp=1.0,stream=True)
    suite=[base,sample,stream,length,context,chat,tools,combined]
    if full:
        for i in (8192,16384,65536,131072,262144):
            if i not in (n,big) and i+128+2048<=max_context and i<=long_tokens:
                suite.append(case("C"+str(i),f"입력 {i//1024}K",n=i))
    return suite

def synthetic_text(n):
    # Stable task, padding independent of the request knobs; no user content.
    lines=[f"test {i}: offset={i%19}; gain={i%13+1}; residual={i*i%113}; analyze noise, calibration, and reproducibility.\n" for i in range(max(256,n//12))]
    return "".join(lines)

TASK="\nWrite a detailed technical explanation of how to validate a sensor calibration experiment. Include pseudocode, mathematical justification, failure modes, and independent follow-up tests. Do not summarize early."

def request(case, text, ids, alias):
    common={"temperature":case["temp"],"top_k":0,"top_p":1.0,"min_p":0.0,"seed":123,
            "cache_prompt":False,"ignore_eos":True,"stream":case["stream"]}
    if case["chat"]:
        body={**common,"model":alias,"messages":[{"role":"user","content":text+TASK}],
              "max_tokens":case["out"]}
        if case["tools"]:body.update(tools=[TOOL],tool_choice="auto")
        if case["stream"]:body["stream_options"]={"include_usage":True}
        return "/v1/chat/completions",body
    return "/completion",{**common,"prompt":ids,"n_predict":case["out"]}

def read_response(response, stream, stop=lambda:False):
    last={};timings={};usage={};chunks=0;first=None;begin=time.monotonic()
    def ingest(x):
        nonlocal chunks,first
        if x.get("error"):raise RuntimeError("서버가 생성 도중 오류를 반환했습니다.")
        if isinstance(x.get("timings"),dict):timings.update(x["timings"])
        if isinstance(x.get("usage"),dict):usage.update(x["usage"])
        delta=x.get("content")
        if not delta:
            choices=x.get("choices") or []
            if choices:
                d=choices[0].get("delta") or choices[0].get("message") or {}
                delta=d.get("content") or d.get("reasoning_content") or d.get("tool_calls")
        if delta:
            chunks+=1
            if first is None:first=time.monotonic()
    if stream:
        for line in response:
            if stop():raise InterruptedError("시험 중지")
            if not line.startswith(b"data:"):continue
            raw=line[5:].strip()
            if raw==b"[DONE]":break
            if not raw:continue
            ingest(json.loads(raw))
    else:
        x=json.loads(response.read(16*1024**2))
        ingest(x)
    return {"timings":timings,"usage":usage,"chunks":chunks,
            "first_event_ms":round((first-begin)*1000,1) if stream and first is not None else None}

def run_http(port,path,body,key=None,stop=lambda:False,timeout=600):
    headers={"Content-Type":"application/json"}
    if key:headers["Authorization"]="Bearer "+key
    req=urllib.request.Request(f"http://127.0.0.1:{int(port)}"+path,data=json.dumps(body).encode(),headers=headers)
    start=time.monotonic()
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=timeout) as r:
        header_ms=(time.monotonic()-start)*1000
        data=read_response(r,body["stream"],stop)
    if data["first_event_ms"] is not None:data["first_event_ms"]+=header_ms
    data["wall_s"]=round(time.monotonic()-start,3)
    return data

def measurements(data, log_timing, requested):
    # Prefer request-scoped engine timing; chat APIs sometimes omit timings.
    # No estimated t/s from SSE chunk counts: one chunk != one token.
    t=data.get("timings") or {}; source="API"
    if not isinstance(t.get("predicted_per_second"),(int,float)):
        t=log_timing or {};source="server log"
    rate=t.get("predicted_per_second")
    count=t.get("predicted_n",data.get("usage",{}).get("completion_tokens"))
    if not isinstance(rate,(float,int)) or not math.isfinite(rate) or rate<=0:
        raise ValueError("엔진 timing 미반환 (속도 판정 불가)")
    if not isinstance(count,(int,float)) or count<=0:raise ValueError("생성 토큰 수 미반환")
    return {"decode":float(rate),"prefill":t.get("prompt_per_second"),
            "prompt_n":t.get("prompt_n",data.get("usage",{}).get("prompt_tokens")),
            "output_n":count,"wall_s":data["wall_s"],"first_event_ms":data.get("first_event_ms"),
            "source":source,"complete":count>=requested,"chunks":data.get("chunks",0)}

def summarize(samples):
    valid=[x for x in samples if x.get("complete")]
    if not valid:raise ValueError("요청한 출력 길이를 충분히 채우지 못해 비교할 수 없습니다.")
    return {"decode":statistics.median(x["decode"] for x in valid),"min":min(x["decode"] for x in valid),
            "max":max(x["decode"] for x in valid),"valid":len(valid),"samples":samples}

def compare(rows):
    indexed={x["id"]:x for x in rows if x.get("status")=="ok"};out=[]
    for row in rows:
        p=indexed.get(row.get("parent"))
        if not p or row.get("status")!="ok":continue
        # Negative delta means slower. Parent for tools is chat, not raw.
        delta=100*(row["decode"]/p["decode"]-1)
        reliable=row["valid"]>=2 and p["valid"]>=2 and row["max"]<p["min"]
        out.append({"id":row["id"],"label":row["label"],"parent":p["id"],"delta_pct":round(delta,2),
                    "nonoverlap":reliable,"not_isolated":row["id"]=="A"})
    return sorted(out,key=lambda x:x["delta_pct"])
