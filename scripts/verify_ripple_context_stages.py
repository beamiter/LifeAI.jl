#!/usr/bin/env python3
"""Staged long-context lookup, exact budgets and resumable prefix-cache checks."""
import argparse
from array import array
import datetime
import hashlib
import json
import secrets
import sys
import time
import urllib.request
from pathlib import Path

SYSTEM = ('You are reading an archive of records. Answer lookup questions by copying the exact verification value associated with the specified key. '
          'Ignore unrelated filler. Never infer a value from the key name. Reply with only the requested value.')
UNIT = ('Routine archive text describes ordinary inventory records, dates, labels, and maintenance notes. '
        'These filler records contain no verification keys.\n')
DEFAULT_STAGES = [131072,262144,524288,1048576]


def render(items):
    return ''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in items)+'<|im_start|>assistant\n'


def ids_sha256(ids):
    words = array('I', ids)
    if sys.byteorder != 'little':
        words.byteswap()
    return hashlib.sha256(words.tobytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


class FitError(AssertionError):
    def __init__(self, message, trace):
        super().__init__(message)
        self.trace = trace


def legacy_fit_message(target,builder,encode,unit_tokens):
    """Reproduce the original successful corpus byte-for-byte, not for new input."""
    base = encode(builder(0,0))
    remaining = target-len(base)
    assert remaining >= 0, (target,len(base))
    lo,hi = 0,max(2,remaining//unit_tokens+128)
    while lo+1 < hi:
        mid = (lo+hi)//2
        count = len(encode(builder(mid,0)))
        if count <= target:
            lo = mid
        else:
            hi = mid
    content = builder(lo,0)
    padding = 0
    trace = [dict(kind='legacy_repeat',repeat=lo,padding=0,tokens=len(encode(content)),target=target)]
    for iteration in range(8):
        ids = encode(content)
        delta = target-len(ids)
        trace.append(dict(kind='legacy_padding',iteration=iteration,repeat=lo,padding=padding,tokens=len(ids),delta=delta))
        if delta == 0:
            return content,ids,dict(method='legacy_v1',repeat=lo,padding=padding,trace=trace)
        padding += delta
        assert padding >= 0
        content = builder(lo,padding)
    raise FitError('Cannot construct an exact token budget',trace)


def fit_message(target,builder,encode,unit_tokens):
    """Bracket actual encoded lengths, then find an exactly representable tail.

    Adding the first ' x' can break a newline merge and cost two tokens. If a
    one-token gap is unreachable, use one fewer24-token UNIT then fill the tail.
    """
    counts = {}
    def evaluate(repeat,padding):
        key = repeat,padding
        if key not in counts:
            content = builder(repeat,padding)
            ids = encode(content)
            counts[key] = len(ids)
            return content,ids
        # Keep only integer counts: a1Mi list per trial would waste host memory.
        content = builder(repeat,padding)
        return content,encode(content)
    trace = []
    base_content,base_ids = evaluate(0,0)
    if len(base_ids)>target:
        raise FitError('Fixed message already exceeds target',trace)
    lo,hi = 0,max(2,(target-len(base_ids))//max(1,unit_tokens)+128)
    for _ in range(64):
        _,ids = evaluate(hi,0)
        trace.append(dict(kind='repeat_bracket',repeat=hi,tokens=len(ids)))
        if len(ids)>target:
            break
        lo,hi = hi,hi*2
    else:
        raise FitError('Unable to bracket repeat count',trace)
    while lo+1<hi:
        mid=(lo+hi)//2
        _,ids=evaluate(mid,0)
        if len(ids)<=target:
            lo=mid
        else:
            hi=mid
    for repeat in range(lo,max(-1,lo-16),-1):
        content,ids=evaluate(repeat,0)
        trace.append(dict(kind='repeat_candidate',repeat=repeat,padding=0,tokens=len(ids),gap=target-len(ids)))
        if len(ids)==target:
            return content,ids,dict(method='bracketed_v2',repeat=repeat,padding=0,trace=trace)
        if len(ids)>target:
            continue
        pad_lo,pad_hi=0,max(1,target-len(ids))
        for _ in range(64):
            _,upper=evaluate(repeat,pad_hi)
            if len(upper)>=target:
                break
            pad_lo,pad_hi=pad_hi,pad_hi*2
        else:
            continue
        while pad_lo<=pad_hi:
            padding=(pad_lo+pad_hi)//2
            content,ids=evaluate(repeat,padding)
            trace.append(dict(kind='padding_candidate',repeat=repeat,padding=padding,tokens=len(ids),delta=target-len(ids)))
            if len(ids)==target:
                return content,ids,dict(method='bracketed_v2',repeat=repeat,padding=padding,trace=trace)
            if len(ids)<target:
                pad_lo=padding+1
            else:
                pad_hi=padding-1
    raise FitError('No exact token budget found with bounded padding search',trace)


def builders(index,record,anchor):
    keys=[n['key'] for n in record['needles']]
    values=[n['value'] for n in record['needles']]
    def archive(count,padding):
        a=count//20;b=count//2;c=count-count//20
        needle=lambda i:f'\nARCHIVE VERIFICATION RECORD: key={keys[i]}; verification_value={values[i]}.\n'
        return (f'Archive segment {index+1}.\n'+(f'ARCHIVE VERIFICATION RECORD: key={anchor["key"]}; verification_value={anchor["value"]}.\n' if index==0 else '')+
                UNIT*a+needle(0)+UNIT*(b-a)+needle(1)+UNIT*(c-b)+needle(2)+UNIT*(count-c)+' x'*padding+
                f'\nEnd of archive segment. Look up key {keys[0]}. Reply with only its exact verification_value.')
    def boundary(count,padding):
        return 'Additional ordinary archive padding.\n'+UNIT*count+' x'*padding+'\nReply with exactly OK.'
    return archive,boundary


def lookup_content(label,record,anchor):
    if label in ('middle','end'):
        index=1 if label=='middle' else 2
        return f'Look up archive key {record["needles"][index]["key"]}. Reply with only its exact verification_value.'
    if label=='global_oldest_anchor':
        return f'Look up archive key {anchor["key"]} from the very first archive segment. Reply with only its exact verification_value.'
    raise ValueError('Unsupported recorded request label: '+label)


def rebuild_history(state,stages,tokenizer):
    messages=[dict(role='system',content=SYSTEM)]
    previous=[]
    proof=[]
    unit_tokens=len(tokenizer.encode(UNIT,add_special_tokens=False).ids)
    for index,record in enumerate(state['stages']):
        archive,boundary=builders(index,record,state['global_anchor'])
        for sample in record.get('requests',[]):
            if sample['status']!='passed':
                raise ValueError('Cannot reconstruct an incomplete inference request: '+sample.get('job_id','unknown'))
            encode=lambda content:tokenizer.encode(render(messages+[dict(role='user',content=content)]),add_special_tokens=False).ids
            label=sample['label']
            if label in ('begin','exact_context_budget'):
                method=sample.get('fit',{}).get('method','legacy_v1')
                fitter=legacy_fit_message if method=='legacy_v1' else fit_message
                content,ids,fit=fitter(sample['prompt_tokens'],archive if label=='begin' else boundary,encode,unit_tokens)
            else:
                content=lookup_content(label,record,state['global_anchor']);ids=encode(content);fit=None
            assert len(ids)==sample['prompt_tokens'],(label,len(ids),sample['prompt_tokens'])
            reused=len(previous) if previous and ids[:len(previous)]==previous else 0
            assert reused==sample['expected_reused_tokens'],(label,reused,sample['expected_reused_tokens'])
            digest=ids_sha256(ids)
            if sample.get('prompt_ids_sha256'):
                assert digest==sample['prompt_ids_sha256'],('saved prompt hash differs',label)
            answer=sample['text'].rsplit('</think>',1)[-1].strip()
            expected=sample['expected']
            assert answer=='OK' if expected=='OK' else expected in answer
            proof.append(dict(stage=record['context_limit'],label=label,prompt_tokens=len(ids),expected_reused_tokens=reused,
                              prompt_ids_sha256=digest,render_sha256=hashlib.sha256(render(messages+[dict(role='user',content=content)]).encode()).hexdigest(),fit=fit))
            messages += [dict(role='user',content=content),dict(role='assistant',content=sample['text'])]
            previous=ids
    return messages,previous,proof


def main():
    from tokenizers import Tokenizer
    from query_ripple_tp8 import discover_endpoint
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    parser.add_argument('--model',type=Path,default=Path('/jfs/application/user/bgl/models/ricursiveAI/Ripple-65B-Preview'))
    parser.add_argument('--stages')
    parser.add_argument('--timeout',type=float,default=21600)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resume',type=Path,nargs='?',const=Path('__default__'))
    parser.add_argument('--dry-run-resume',action='store_true')
    args=parser.parse_args()
    output=args.output or args.run_dir/'results/context-stages.json'
    checkpoint_path=output.with_name(output.stem+'.messages.json')
    proof_path=output.with_name(output.stem+'.resume-proof.json')
    ready=json.loads((args.run_dir/'results/ready.json').read_text())
    tokenizer=Tokenizer.from_file(str(args.model/'tokenizer.json'))
    if args.resume is not None:
        resume_path=output if args.resume==Path('__default__') else args.resume
        state=json.loads(resume_path.read_text())
        assert state['code_revision']==ready['code_revision'],'Resume belongs to a different loaded model service'
        if args.stages:
            stages=[int(x) for x in args.stages.split(',')]
        else:
            stages=state.get('requested_stages') or [r['context_limit'] for r in state['stages']]
        assert [r['context_limit'] for r in state['stages']]==stages[:len(state['stages'])]
        if checkpoint_path.exists():
            saved=json.loads(checkpoint_path.read_text());messages=saved['messages']
            previous_ids=tokenizer.encode(render(messages[:-1]),add_special_tokens=False).ids
            assert len(previous_ids)==saved['previous_prompt_tokens'] and ids_sha256(previous_ids)==saved['previous_prompt_ids_sha256']
            complete=sum(len(r.get('requests',[])) for r in state['stages'])
            assert complete==saved['completed_requests'],'Message checkpoint does not cover the recorded results'
            proof=saved.get('reconstruction_proof',[])
        else:
            messages,previous_ids,proof=rebuild_history(state,stages,tokenizer)
        report=dict(code_revision=state['code_revision'],completed_requests=sum(len(r.get('requests',[])) for r in state['stages']),
                    last_prompt_tokens=len(previous_ids),last_prompt_ids_sha256=ids_sha256(previous_ids),
                    original_error=state.get('error'),reconstruction_proof=proof,stage_fits=[])
    else:
        if args.dry_run_resume:
            parser.error('--dry-run-resume requires --resume')
        stages=[int(x) for x in args.stages.split(',')] if args.stages else DEFAULT_STAGES
        state=dict(started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),endpoint=ready['endpoint'],configured_context=ready['max_context'],code_revision=ready['code_revision'],stages=[],passed_limits=[],all_passed=False)
        state['global_anchor']=dict(key='LC_GLOBAL_ANCHOR',value='V'+secrets.token_hex(8).upper())
        messages=[dict(role='system',content=SYSTEM)];previous_ids=[];proof=[];report=None
    state['requested_stages']=stages
    unit_tokens=len(tokenizer.encode(UNIT,add_special_tokens=False).ids)
    def encode(content):return tokenizer.encode(render(messages+[dict(role='user',content=content)]),add_special_tokens=False).ids
    def save():atomic_json(output,state)
    def save_checkpoint():
        atomic_json(checkpoint_path,dict(code_revision=state['code_revision'],messages=messages,
            previous_prompt_tokens=len(previous_ids),previous_prompt_ids_sha256=ids_sha256(previous_ids),
            completed_requests=sum(len(r.get('requests',[])) for r in state['stages']),reconstruction_proof=proof))
    if args.dry_run_resume:
        pending=next((r for r in state['stages'] if r['status']!='passed'),None)
        if pending is not None:
            assert not pending.get('requests'),'Dry-run resume only supports a pre-submission harness failure'
            index=stages.index(pending['context_limit']);archive,_=builders(index,pending,state['global_anchor'])
            try:
                legacy_fit_message(pending['context_limit']-4096,archive,encode,unit_tokens)
            except FitError as error:
                report['legacy_failure_trace']=error.trace
            content,ids,fit=fit_message(pending['context_limit']-4096,archive,encode,unit_tokens)
            assert ids[:len(previous_ids)]==previous_ids,'Rebuilt pending request changed the live cached prefix'
            report['pending_fit']=dict(stage=pending['context_limit'],prompt_tokens=len(ids),prompt_ids_sha256=ids_sha256(ids),expected_reused_tokens=len(previous_ids),fit=fit)
        save_checkpoint();atomic_json(proof_path,report)
        print(json.dumps({k:v for k,v in report.items() if k not in ('reconstruction_proof','legacy_failure_trace')},ensure_ascii=False),flush=True)
        return
    if args.resume is not None:
        state.setdefault('resume_history',[]).append(dict(resumed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            original_started_at=state['started_at'],original_error=state.pop('error',None),original_fit_error_trace=state.pop('fit_error_trace',None),last_prompt_tokens=len(previous_ids),last_prompt_ids_sha256=ids_sha256(previous_ids),proof_file=str(proof_path)))
        save();save_checkpoint()
    key=(args.run_dir/'credentials/api_key').read_text().strip()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));endpoint,_=discover_endpoint(ready,opener);state['endpoint']=endpoint
    current_job=None
    def request(method,path,payload=None,timeout=30):
        req=urllib.request.Request(endpoint+path,method=method,data=None if payload is None else json.dumps(payload,ensure_ascii=False).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
        with opener.open(req,timeout=timeout) as response:return json.load(response)
    def query(content,label,expected,budget,record,known_ids=None,fit=None):
        nonlocal previous_ids,current_job
        proposed=messages+[dict(role='user',content=content)]
        ids=known_ids if known_ids is not None else encode(content)
        assert len(ids)+budget<=record['context_limit']
        expected_reuse=len(previous_ids) if previous_ids and ids[:len(previous_ids)]==previous_ids else 0
        assert not previous_ids or expected_reuse==len(previous_ids),'Chat template changed the cached token prefix'
        start=time.monotonic();payload=dict(model=ready['model_id'],messages=proposed,temperature=0,max_tokens=budget,stream=False,context_limit=record['context_limit'])
        created=request('POST','/v1/chat/completions/jobs',payload,timeout=120);current_job=created['id']
        record.setdefault('requests',[]).append(dict(label=label,prompt_tokens=len(ids),prompt_ids_sha256=ids_sha256(ids),reserved_output=budget,expected_reused_tokens=expected_reuse,job_id=current_job,status='running',fit=fit))
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
        messages[:]=proposed+[dict(role='assistant',content=text)];previous_ids=ids;save_checkpoint()
    try:
        for index,limit in enumerate(stages):
            assert limit<=ready['max_context']
            existing=next((r for r in state['stages'] if r['context_limit']==limit),None)
            if existing is not None and existing['status']=='passed':continue
            record=existing or dict(context_limit=limit,status='running',started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
            if existing is None:state['stages'].append(record)
            if not record.get('needles'):
                keys=[f'LC_{limit}_{part}' for part in ('BEGIN','MIDDLE','END')]
                record['needles']=[dict(key=k,value='V'+secrets.token_hex(8).upper(),relative_placement=p) for k,p in zip(keys,(.05,.50,.95))]
            save();archive,boundary=builders(index,record,state['global_anchor'])
            done={q['label'] for q in record.get('requests',[]) if q['status']=='passed'}
            assert len(done)==len(record.get('requests',[])),'Inspect incomplete inference before resuming'
            if 'begin' not in done:
                content,ids,fit=fit_message(limit-4096,archive,encode,unit_tokens)
                query(content,'begin',record['needles'][0]['value'],256,record,ids,fit)
            for label,position in [('middle',1),('end',2)]:
                if label not in done:query(lookup_content(label,record,state['global_anchor']),label,record['needles'][position]['value'],256,record)
            if limit==max(stages) and 'global_oldest_anchor' not in done:
                query(lookup_content('global_oldest_anchor',record,state['global_anchor']),'global_oldest_anchor',state['global_anchor']['value'],256,record)
            if 'exact_context_budget' not in done:
                content,ids,fit=fit_message(limit-256,boundary,encode,unit_tokens)
                query(content,'exact_context_budget','OK',256,record,ids,fit)
            assert record['requests'][-1]['prompt_tokens']+256==limit
            record.update(status='passed',completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
            if limit not in state['passed_limits']:state['passed_limits'].append(limit)
            save();print(json.dumps(dict(phase='stage_passed',context_limit=limit,lookups=len(record['requests'])-1)),flush=True)
        state['all_passed']=True;state['completed_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
    except BaseException as error:
        if current_job is not None:
            try:request('DELETE','/v1/chat/completions/jobs/'+current_job)
            except OSError:pass
        state['error']=str(error)
        if isinstance(error,FitError):state['fit_error_trace']=error.trace
        save();raise


if __name__=='__main__':main()
