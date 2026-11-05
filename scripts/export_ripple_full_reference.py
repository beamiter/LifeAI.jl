#!/usr/bin/env python3
"""Run the pinned official 65B forward with per-layer CUDA weight loading.

Uses the unmodified official RicursiveForCausalLM.forward. Forward hooks only
materialize/dematerialize parameters, keeping the full BF16 checkpoint on disk.
Run with /isaac-sim/python.sh on the DSW (torch 2.7, transformers 4.57.6).
"""
import argparse,hashlib,importlib,json,pathlib,sys,time,types
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer

HASHES={
 'configuration_ricursive.py':'002cb3915be9685eb962d617623bec14d45b5d6b5d46e689fac98a3b38cb6cf6',
 'modeling_ricursive.py':'f83865ef643e273a6c9aecf1575e85c95ad01fd995b1e43c3c7a892765006bb3',
}
REVISION='e27e171792f303d74c048184e8259ba663cfa0d5'

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('checkpoint',type=pathlib.Path)
 parser.add_argument('output',type=pathlib.Path)
 parser.add_argument('--long-tokens',type=int,default=1057)
 parser.add_argument('--greedy-tokens',type=int,default=4)
 parser.add_argument('--case',default=None)
 parser.add_argument('--save-hidden',action='store_true')
 parser.add_argument('--attention-backend',choices=['math','auto'],default='math')
 args=parser.parse_args();root=args.checkpoint;args.output.mkdir(parents=True,exist_ok=True)
 verification=json.loads((root/'verification.json').read_text())
 assert verification['all_valid'] and verification['revision']==REVISION
 for name,digest in HASHES.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
 package=types.ModuleType('ripple_official');package.__path__=[str(root.resolve())];sys.modules[package.__name__]=package
 Config=importlib.import_module('ripple_official.configuration_ricursive').RicursiveConfig
 Model=importlib.import_module('ripple_official.modeling_ricursive').RicursiveForCausalLM
 config=Config(**json.loads((root/'config.json').read_text()))
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 if args.attention_backend=='math':
  # The DSW's torch 2.7 efficient SDPA differs from the FP32 mathematical
  # result for some sequence lengths on this Blackwell GPU. Pin a reliable
  # reference backend while leaving the published model forward unchanged.
  torch.backends.cuda.enable_flash_sdp(False)
  torch.backends.cuda.enable_mem_efficient_sdp(False)
  torch.backends.cuda.enable_cudnn_sdp(False)
  torch.backends.cuda.enable_math_sdp(True)
 with torch.device('meta'):model=Model(config)
 model.eval()
 index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
 transfers={'bytes':0,'calls':0};started=time.time()
 def load(prefix,module):
  for relative,parameter in list(module.named_parameters()):
   name=prefix+'.'+relative if prefix else relative
   with safe_open(root/index[name],framework='pt',device='cpu') as shard:
    tensor=shard.get_tensor(name).to(device='cuda',dtype=torch.float32)
   parent,_,leaf=relative.rpartition('.')
   owner=module.get_submodule(parent) if parent else module
   owner._parameters[leaf]=torch.nn.Parameter(tensor,requires_grad=False)
   transfers['bytes']+=tensor.numel()*tensor.element_size();transfers['calls']+=1
  if prefix.startswith('model.layers.'):
   layer=int(prefix.rsplit('.',1)[1])
   if layer%10==0:print(json.dumps({'phase':'layer','layer':layer,'elapsed':time.time()-started}),flush=True)
 def unload(module):
  for relative,parameter in list(module.named_parameters()):
   parent,_,leaf=relative.rpartition('.')
   owner=module.get_submodule(parent) if parent else module
   owner._parameters[leaf]=torch.nn.Parameter(torch.empty(parameter.shape,device='meta'),requires_grad=False)
 def hook_module(prefix,module):
  module.register_forward_pre_hook(lambda m,inputs,p=prefix:load(p,m))
  module.register_forward_hook(lambda m,inputs,output:(unload(m),None)[1])
 hook_module('model.embed_tokens',model.model.embed_tokens)
 for number,layer in enumerate(model.model.layers):hook_module(f'model.layers.{number}',layer)
 hook_module('model.norm',model.model.norm);hook_module('lm_head',model.lm_head)
 tokenizer=AutoTokenizer.from_pretrained(root)
 def prompt(text):return tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True)+'<think>\n\n</think>\n'
 cases=[{'name':'exact_ok','prompt':prompt('Reply with exactly OK.'),'greedy':args.greedy_tokens},
        {'name':'math','prompt':prompt('计算 17 × 23，只回答结果。'),'greedy':0}]
 long=tokenizer.encode('hello world\n'*args.long_tokens,add_special_tokens=False)[:args.long_tokens]
 cases.append({'name':'long_window','input_ids':long,'greedy':0})
 tensors={};records=[];torch.cuda.reset_peak_memory_stats()
 if args.save_hidden:
  def save_embedding(module,inputs,value):
   if config.embed_scale is not None:value=value*config.embed_scale
   tensors['hidden0']=value[0].transpose(0,1).float().cpu().contiguous()
  model.model.embed_tokens.register_forward_hook(save_embedding)
  for number,layer in enumerate(model.model.layers):
   layer.register_forward_hook(lambda module,inputs,value,n=number:tensors.__setitem__(f'hidden{n+1}',value[0][0].transpose(0,1).float().cpu().contiguous()))
 with torch.inference_mode():
  for number,case in enumerate(cases):
   if args.case is not None and case['name']!=args.case:continue
   ids=case.get('input_ids') or tokenizer.encode(case['prompt'],add_special_tokens=False)
   before=time.perf_counter();output=model(torch.tensor([ids],device='cuda'),use_cache=True)
   torch.cuda.synchronize();prefill=time.perf_counter()-before
   tensors[f'case{number}.prefill_logits']=output.logits[0,-1].float().cpu().contiguous()
   generated=[];cache=output.past_key_values;decode_seconds=[]
   for step in range(case['greedy']):
    token=int(output.logits[0,-1].argmax());generated.append(token)
    tensors[f'case{number}.step{step}_logits']=output.logits[0,-1].float().cpu().contiguous()
    if token in config.eos_token_id or step==case['greedy']-1:break
    before=time.perf_counter();output=model(torch.tensor([[token]],device='cuda'),past_key_values=cache,use_cache=True)
    torch.cuda.synchronize();decode_seconds.append(time.perf_counter()-before);cache=output.past_key_values
   record={**case,'input_ids':ids,'prompt_tokens':len(ids),'greedy_tokens':generated,
           'text':tokenizer.decode(generated,skip_special_tokens=True),'prefill_seconds':prefill,'decode_seconds':decode_seconds}
   records.append(record);del output,cache
   save_file(tensors,str(args.output/'reference.safetensors'))
   (args.output/'reference.json').write_text(json.dumps({'repository':'ricursiveAI/Ripple-65B-Preview','revision':REVISION,
    'torch':torch.__version__,'activation_dtype':'float32','attention_backend':args.attention_backend,
    'synthetic':verification.get('synthetic',False),
    'model_parameters':sum(p.numel() for p in model.parameters()),'cases':records,'transfer':transfers,
    'peak_cuda_allocated':torch.cuda.max_memory_allocated(),'elapsed':time.time()-started},ensure_ascii=False,indent=2)+'\n')
   print(json.dumps({'phase':'case_complete','name':case['name'],'tokens':len(ids),'seconds':prefill,'generated':generated,'text':record['text']},ensure_ascii=False),flush=True)
 print('REFERENCE_COMPLETE',flush=True)

if __name__=='__main__':main()
