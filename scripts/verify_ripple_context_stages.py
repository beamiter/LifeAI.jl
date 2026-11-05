#!/usr/bin/env python3
"""Exercise staged long context, long-distance lookup and exact prefix reuse.

Synthetic public records only. Each stage appends to one conversation, probes
three different positions, then fills the input+output budget to the tier limit.
"""
import argparse,datetime,hashlib,json,re,secrets,time,urllib.request
from pathlib import Path
from tokenizers import Tokenizer
from query_ripple_tp8 import discover_endpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    parser.add_argument('--model',type=Path,default=Path('/jfs/application/user/bgl/models/ricursiveAI/Ripple-65B-Preview'))
    parser.add_argument('--stages',default='131072,262144,524288,1048576')
    parser.add_argument('--timeout',type=float,default=21600)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();stages=[int(x) for x in args.stages.split(',')]
    output=args.output or args.run_dir/'results/context-stages.json'
    ready=json.loads((args.run_dir/'results/ready.json').read_text());key=(args.run_dir/'credentials/api_key').read_text().strip()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));endpoint,_=discover_endpoint(ready,opener)
    tokenizer=Tokenizer.from_file(str(args.model/'tokenizer.json'))
    state=dict(started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),endpoint=endpoint,configured_context=ready['max_context'],code_revision=ready['code_revision'],stages=[],passed_limits=[],all_passed=False)
    messages=[dict(role='system',content='You are reading an archive of records. Answer lookup questions by copying the exact verification value associated with the specified key. Ignore unrelated filler. Never infer a value from the key name. Reply with only the requested value.')]
    previous_ids=[];current_job=None
    def save():
        temp=output.with_suffix('.tmp');temp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n');temp.replace(output)
    def render(items):return ''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in items)+'<|im_start|>assistant\n'
    def encode(items):return tokenizer.encode(render(items),add_special_tokens=False).ids
    def request(method,path,payload=None,timeout=30):
        req=urllib.request.Request(endpoint+path,method=method,data=None if payload is None else json.dumps(payload,ensure_ascii=False).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
        with opener.open(req,timeout=timeout) as response:return json.load(response)
    def fit_message(target,builder):
        base=encode(messages+[dict(role='user',content=builder(0,0))]);remaining=target-len(base)
        assert remaining>=0,(target,len(base))
        lo,hi=0,max(2,remaining//unit_tokens+128)
        while lo+1<hi:
            mid=(lo+hi)//2
            count=len(encode(messages+[dict(role='user',content=builder(mid,0))]))
            if count<=target:lo=mid
            else:hi=mid
        content=builder(lo,0);padding=0
        for _ in range(8):
            ids=encode(messages+[dict(role='user',content=content)])
            delta=target-len(ids)
            if delta==0:return content,ids
            padding+=delta;assert padding>=0
            content=builder(lo,padding)
        raise AssertionError('Cannot construct an exact token budget')
    def query(content,label,expected,budget,record,known_ids=None):
        nonlocal previous_ids,current_job
        proposed=messages+[dict(role='user',content=content)]
        ids=known_ids if known_ids is not None else encode(proposed)
        assert len(ids)+budget<=record['context_limit']
        expected_reuse=len(previous_ids) if previous_ids and ids[:len(previous_ids)]==previous_ids else 0
        assert not previous_ids or expected_reuse==len(previous_ids), "Chat template did not preserve the exact cached token prefix"
        start=time.monotonic();payload=dict(model=ready['model_id'],messages=proposed,temperature=0,max_tokens=budget,stream=False,context_limit=record['context_limit'])
        created=request('POST','/v1/chat/completions/jobs',payload,timeout=120);current_job=created['id']
        record.setdefault('requests',[]).append(dict(label=label,prompt_tokens=len(ids),reserved_output=budget,expected_reused_tokens=expected_reuse,job_id=current_job,status='running'))
        sample=record['requests'][-1];save();last_print=0
        while True:
            job=request('GET','/v1/chat/completions/jobs/'+current_job)
            if job['status']=='completed':body=job['result'];break
            if job['status'] in ('failed','cancelled'):raise RuntimeError(job.get('error') or job['status'])
            if time.monotonic()-start>args.timeout:raise TimeoutError(label)
            now=time.monotonic()
            if now-last_print>=15:
                sample.update(phase=job.get('phase'),processed=job.get('prefill_processed'),total=job.get('prompt_eval_count'),elapsed_seconds=now-start)
                save();print(json.dumps(dict(stage=record['context_limit'],label=label,phase=sample['phase'],processed=sample['processed'],total=sample['total'],seconds=sample['elapsed_seconds'])),flush=True);last_print=now
            time.sleep(2)
        current_job=None;elapsed=time.monotonic()-start
        text=body['choices'][0]['message']['content'];answer=text.rsplit('</think>',1)[-1].strip()
        assert body['usage']['prompt_tokens']==len(ids),(len(ids),body['usage'])
        assert body['choices'][0]['finish_reason']=='stop',text
        correct=answer=='OK' if expected=='OK' else expected in answer
        prefix=body['lifeai'].get('prefix_cache') or {}
        assert prefix.get('reused_tokens')==expected_reuse,(label,prefix,expected_reuse)
        sample.update(status='passed' if correct else 'failed',seconds=elapsed,completion_tokens=body['usage']['completion_tokens'],text=text,answer=answer,expected=expected,lookup_correct=correct,lifeai=body['lifeai'])
        save();print(json.dumps(dict(stage=record['context_limit'],label=label,correct=correct,seconds=elapsed,prompt_tokens=len(ids),reused_tokens=prefix.get('reused_tokens'),completion_tokens=body['usage']['completion_tokens'])),flush=True)
        assert correct,(label,expected,answer)
        messages[:]=proposed+[dict(role='assistant',content=text)];previous_ids=ids
    unit='Routine archive text describes ordinary inventory records, dates, labels, and maintenance notes. These filler records contain no verification keys.\n'
    unit_tokens=len(tokenizer.encode(unit,add_special_tokens=False).ids)
    anchor_key='LC_GLOBAL_ANCHOR'
    anchor_value='V'+secrets.token_hex(8).upper()
    state['global_anchor']=dict(key=anchor_key,value=anchor_value)
    try:
        for index,limit in enumerate(stages):
            assert limit<=ready['max_context']
            record=dict(context_limit=limit,status='running',started_at=datetime.datetime.now(datetime.timezone.utc).isoformat());state['stages'].append(record);save()
            keys=[f'LC_{limit}_{part}' for part in ['BEGIN','MIDDLE','END']]
            values=['V'+secrets.token_hex(8).upper() for _ in keys]
            record['needles']=[dict(key=k,value=v,relative_placement=p) for k,v,p in zip(keys,values,[.05,.50,.95])]
            def archive(count,padding):
                a=count//20;b=count//2;c=count-count//20
                needle=lambda i:f'\nARCHIVE VERIFICATION RECORD: key={keys[i]}; verification_value={values[i]}.\n'
                return (f'Archive segment {index+1}.\n'+(f'ARCHIVE VERIFICATION RECORD: key={anchor_key}; verification_value={anchor_value}.\n' if index==0 else '')+unit*a+needle(0)+unit*(b-a)+needle(1)+unit*(c-b)+needle(2)+unit*(count-c)+' x'*padding+f'\nEnd of archive segment. Look up key {keys[0]}. Reply with only its exact verification_value.')
            content,ids=fit_message(limit-4096,archive)
            query(content,'begin',values[0],256,record,ids)
            for position in (1,2):
                query(f'Look up archive key {keys[position]}. Reply with only its exact verification_value.', ['begin','middle','end'][position],values[position],256,record)
            if limit==max(stages):
                query(f'Look up archive key {anchor_key} from the very first archive segment. Reply with only its exact verification_value.', 'global_oldest_anchor',anchor_value,256,record)
            def boundary(count,padding):return 'Additional ordinary archive padding.\n'+unit*count+' x'*padding+'\nReply with exactly OK.'
            content,ids=fit_message(limit-256,boundary)
            query(content,'exact_context_budget','OK',256,record,ids)
            assert record['requests'][-1]['prompt_tokens']+256==limit
            record.update(status='passed',completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat());state['passed_limits'].append(limit);save()
            print(json.dumps(dict(phase='stage_passed',context_limit=limit,lookups=len(record['requests'])-1)),flush=True)
        state['all_passed']=True;state['completed_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
    except BaseException as error:
        if current_job is not None:
            try:request('DELETE','/v1/chat/completions/jobs/'+current_job)
            except OSError:pass
        state['error']=str(error);save();raise


if __name__=='__main__':main()
