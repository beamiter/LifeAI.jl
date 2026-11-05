#!/usr/bin/env python3
"""Measure repeatable end-to-end TP8 HTTP throughput without exposing its key."""
import argparse,json,time,statistics
from pathlib import Path
import urllib.request
from query_ripple_tp8 import discover_endpoint
p=argparse.ArgumentParser(description=__doc__);p.add_argument('run_dir',type=Path);p.add_argument('--repeats',type=int,default=3);p.add_argument('--tokens',type=int,default=96);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert a.repeats>0 and a.tokens>1
ready=json.loads((a.run_dir/'results/ready.json').read_text());key=(a.run_dir/'credentials/api_key').read_text().strip();opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));endpoint,_=discover_endpoint(ready,opener)
payload=dict(model=ready['model_id'],messages=[dict(role='user',content='Explain step by step how binary search works and provide a Python implementation.')],temperature=0,max_tokens=a.tokens,stream=False)
records=[]
for repeat in range(a.repeats):
 request=urllib.request.Request(endpoint+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+key});start=time.monotonic()
 with opener.open(request,timeout=600) as response:body=json.load(response)
 elapsed=time.monotonic()-start;count=body['usage']['completion_tokens'];assert count==a.tokens,(count,body)
 record=dict(repeat=repeat+1,prompt_tokens=body['usage']['prompt_tokens'],completion_tokens=count,seconds=elapsed,tokens_per_second=count/elapsed,server=body['lifeai'],text=body['choices'][0]['message']['content']);records.append(record)
 result=dict(endpoint=endpoint,scope='End-to-end non-streaming HTTP; includes prefill',tokens=a.tokens,records=records,median_tokens_per_second=statistics.median(r['tokens_per_second'] for r in records));a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record,ensure_ascii=False),flush=True)
