#!/usr/bin/env python3
"""Generate a TP8-compatible synthetic fixture with the pinned official model."""
import argparse,hashlib,importlib,json,pathlib,sys,types
import torch
from safetensors.torch import save_file
from export_ripple_reference import SOURCE_HASHES,REVISION

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--upstream',type=pathlib.Path,required=True)
 parser.add_argument('--output',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1]/'test/ripple/fixtures/tp8')
 args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 for name,expected in SOURCE_HASHES.items():assert hashlib.sha256((args.upstream/name).read_bytes()).hexdigest()==expected
 package=types.ModuleType('ripple_tp_official');package.__path__=[str(args.upstream.resolve())];sys.modules[package.__name__]=package
 Config=importlib.import_module('ripple_tp_official.configuration_ricursive').RicursiveConfig
 Model=importlib.import_module('ripple_tp_official.modeling_ricursive').RicursiveForCausalLM
 torch.set_num_threads(1);torch.manual_seed(891)
 config=Config(vocab_size=288,hidden_size=32,intermediate_size=64,num_hidden_layers=5,
  num_attention_heads=16,num_key_value_heads=8,head_dim=4,max_position_embeddings=128,sliding_window=3,
  layer_types=['sliding_attention','full_attention','sliding_attention','full_attention','sliding_attention'],
  use_valres=True,ffn_clamp_threshold=0.4,embed_scale=32**0.5,
  bos_token_id=257,eos_token_id=[265,257],pad_token_id=277)
 model=Model(config).float().eval()
 with torch.no_grad():
  for name,p in model.named_parameters():
   if 'layernorm' in name or '_norm.weight' in name or name=='model.norm.weight':p.copy_(0.7+torch.rand_like(p)*0.6)
   elif name.endswith('value_res_lambda'):p.copy_(torch.randn_like(p))
   else:p.copy_(torch.randn_like(p)*0.2)
 weights={k:v.detach().contiguous() for k,v in model.state_dict().items()}
 save_file(weights,str(args.output/'model.safetensors'))
 (args.output/'config.json').write_text(json.dumps(config.to_dict(),indent=2)+'\n')
 tokens=torch.tensor([[3,8,4,7,9,2,12,16,6,1,15,5,21]])
 with torch.inference_mode():
  logits=model(tokens,use_cache=False).logits[0];growing=tokens.clone();greedy=[]
  for _ in range(4):
   token=int(model(growing,use_cache=False).logits[0,-1].argmax());greedy.append(token)
   growing=torch.cat([growing,torch.tensor([[token]])],dim=1)
 reference={'repository':'ricursiveAI/Ripple-65B-Preview','revision':REVISION,'synthetic':True,
  'input_ids':tokens[0].tolist(),'logits':logits.tolist(),'greedy_tokens':greedy}
 (args.output/'reference.json').write_text(json.dumps(reference,indent=2)+'\n')
 print(args.output)

if __name__=='__main__':main()
