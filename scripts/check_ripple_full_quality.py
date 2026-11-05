#!/usr/bin/env python3
"""Check the deliberately small real-model smoke tasks; not a benchmark suite."""
import argparse,ast,json,pathlib,re
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('generation',type=pathlib.Path)
parser.add_argument('output',type=pathlib.Path)
parser.add_argument('--allow-math-equation',action='store_true')
args=parser.parse_args();data=json.loads(args.generation.read_text());cases={c['name']:c for c in data['cases']}
results={}
results['exact_ok']=cases['exact_ok']['text'].strip()=='OK'
math_text=cases['math']['text'].strip()
numbers=re.findall(r'(?<!\d)[-+]?\d+(?:\.\d+)?(?!\d)',math_text)
results['math_17_times_23']=(bool(numbers) and float(numbers[-1])==391) if args.allow_math_equation else math_text=='391'
text=cases['code']['text'];match=re.search(r'```(?:python)?\s*\n(.*?)```',text,re.S)
code=match.group(1) if match else text.strip()
allowed=(ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Return,ast.BinOp,ast.Add,ast.Name,ast.Load,ast.Expr,ast.Constant)
try:
 tree=ast.parse(code)
 assert all(isinstance(node,allowed) for node in ast.walk(tree))
 assert len(tree.body)==1 and isinstance(tree.body[0],ast.FunctionDef)
 function=tree.body[0];assert function.name=='add' and not function.decorator_list
 scope={'__builtins__':{'int':int,'float':float,'str':str}};exec(compile(tree,'<ripple-generated-add>','exec'),scope)
 results['python_add']=all(scope['add'](a,b)==a+b for a,b in [(2,3),(-7,11),(0,0),(1.5,2.5)])
except (AssertionError,SyntaxError,TypeError,ValueError,NameError,KeyError):results['python_add']=False
results['chinese_output']=len(re.findall(r'[\u3400-\u9fff]',cases['chinese']['text']))>=5
results['steady_decode_at_least_64_tokens']=cases['throughput']['generated_tokens']>=64
output={'checks':results,'all_passed':all(results.values()),'scope':'Five smoke checks, not a general quality benchmark','code':code,
        'math_mode':'last_numeric_result' if args.allow_math_equation else 'exact_result_only',
        'throughput_tokens_per_second':cases['throughput']['decode_tokens_per_second']}
args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n');print(json.dumps(output,ensure_ascii=False))
raise SystemExit(0 if output['all_passed'] else 1)
