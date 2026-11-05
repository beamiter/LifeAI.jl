#!/usr/bin/env python3
"""CPU-only synthetic transcript flattening for exact Lumen cached-prefix acceptance.
No inference, network calls, credentials, or checkpoint writes. Requires tokenizers,
transformers (for the official Jinja renderer), and a local Ripple tokenizer.
"""
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import sys

NEXT_USER = 'Reply with exactly OK.'
LIMIT = 1048576

def sha(data):
    return hashlib.sha256(data).hexdigest()

def ids_sha(ids):
    words = array('I', ids)
    assert words.itemsize == 4
    if sys.byteorder != 'little':
        words.byteswap()
    return sha(words.tobytes())

def service_render(messages):
    # Exact no-tools/no-reasoning branch of apply_ripple_chat_template.
    return ''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in messages)+'<|im_start|>assistant\n'

def main():
    from tokenizers import Tokenizer
    from transformers.utils.chat_template_utils import _compile_jinja_template
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--overbudget-file', type=Path, help='Also write a synthetic oversized prompt expected to receive HTTP 400; no inference is sent')
    args = p.parse_args()
    checkpoint_bytes = args.checkpoint.read_bytes()
    saved = json.loads(checkpoint_bytes)
    messages = saved['messages']
    assert len(messages) >= 3 and messages[0]['role'] == 'system'
    assert messages[-1]['role'] == 'assistant'
    for i, message in enumerate(messages):
        assert set(message) == {'role', 'content'}, 'Only plain synthetic messages supported'
        assert isinstance(message['content'], str)
        assert message['role'] == ('system' if i == 0 else 'user' if i % 2 else 'assistant')
    tok_path = args.model/'tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tok_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    template_path = args.model/'chat_template.jinja'
    template = template_path.read_text() if template_path.exists() else json.loads((args.model/'tokenizer_config.json').read_text())['chat_template']
    assert isinstance(template, str)
    assert sha(template.encode()) in {'877c2dbf365243dda14738ca250b31398059b35ffd91d4ed0ec68be45ca6ebd3', '5958bf5c3ae31d278e5e65f37ea7119df04b5d56564d8fd802a8c6692567db17'}, 'Unsupported service template'
    official_template = _compile_jinja_template(template)
    def render(items):
        text = service_render(items)
        assert official_template.render(messages=items, tools=None, add_generation_prompt=True, clear_thinking=False) == text
        return text
    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False).ids
    previous = encode(render(messages[:-1]))
    assert len(previous) == saved['previous_prompt_tokens']
    assert ids_sha(previous) == saved['previous_prompt_ids_sha256'], 'Checkpoint old-prompt hash mismatch'
    extended = messages+[{'role':'user', 'content':NEXT_USER}]
    full_text = render(extended)
    system = messages[0]['content']
    # Strip only the formatter-owned first-user opener and final-user closer.
    transcript = ''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in extended[1:])
    opener, closer = '<|im_start|>user\n', '<|im_end|>\n'
    assert transcript.startswith(opener) and transcript.endswith(closer)
    flat = transcript[len(opener):-len(closer)]
    flat_text = render([{'role':'system','content':system}, {'role':'user','content':flat}])
    assert full_text == flat_text, 'Flattened formatter bytes differ'
    full_ids, flat_ids = encode(full_text), encode(flat_text)
    assert full_ids == flat_ids, 'Flattened token IDs differ'
    assert full_ids[:len(previous)] == previous, 'Prior validated prompt is not an exact prefix'
    assert ids_sha(full_ids[:len(previous)]) == saved['previous_prompt_ids_sha256']
    assert len(full_ids)+8 <= LIMIT, 'Prompt plus eight output tokens exceeds 1Mi'
    output = args.output_dir.resolve()
    assert output != args.checkpoint.parent.resolve(), 'Use a distinct output directory'
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        'purpose':'Synthetic transcript flattening for Lumen --system/--file cached-prefix verification only; embedded role markers deliberately reproduce the validated multi-turn token sequence.',
        'inference_performed':False, 'checkpoint':str(args.checkpoint.resolve()),
        'checkpoint_sha256':sha(checkpoint_bytes), 'code_revision':saved.get('code_revision'),
        'completed_requests':saved['completed_requests'], 'message_count':len(messages),
        'tokenizer_sha256':sha(tok_path.read_bytes()), 'chat_template_sha256':sha(template.encode()),
        'token_id_hash_encoding':'SHA256 of zero-based UInt32 token IDs, little endian',
        'previous_prompt_tokens':len(previous), 'previous_prompt_ids_sha256':ids_sha(previous),
        'expected_prefix_tokens_at_least':len(previous),
        'expected_prefix_note':'Guaranteed prior prompt only; final generated token/cache coverage depends on native generation. No stronger cache hit is assumed.',
        'prompt_tokens':len(full_ids), 'prompt_ids_sha256':ids_sha(full_ids),
        'max_tokens':8, 'context_limit':LIMIT, 'total_budget_tokens':len(full_ids)+8,
        'remaining_context_tokens':LIMIT-len(full_ids)-8,
        'prompt_file_bytes':len(flat.encode()), 'prompt_file_sha256':sha(flat.encode()),
        'system_file_bytes':len(system.encode()), 'system_file_sha256':sha(system.encode()),
        'formatted_prompt_bytes':len(full_text.encode()), 'formatted_prompt_sha256':sha(full_text.encode()),
        'assertions':['official formatter equals service formatter','flattened bytes equal multi-turn bytes','complete token IDs equal','checkpoint previous prompt hash matches','previous prompt is exact token prefix','prompt plus 8 output tokens <= 1048576'],
    }
    (output/'prompt.txt').write_bytes(flat.encode())
    (output/'system.txt').write_bytes(system.encode())
    if args.overbudget_file is not None:
        over = flat + ' x'*max(512, LIMIT+1-len(full_ids))
        over_text = render([{'role':'system','content':system}, {'role':'user','content':over}])
        over_ids = encode(over_text)
        assert len(over_ids)+8 > LIMIT, 'Oversized fixture did not exceed context budget'
        over_bytes = over.encode()
        # Mirror the CLI non-stream chat payload; JSON escaping can change wire size.
        payload = {'model':'ripple', 'messages':[{'role':'system','content':system}, {'role':'user','content':over}], 'max_tokens':8, 'stream':False}
        json_bytes = json.dumps(payload, ensure_ascii=False, separators=(',',':')).encode()
        assert len(over_bytes) < 64*1024*1024 and len(json_bytes) < 64*1024*1024
        over_path = args.overbudget_file.resolve()
        assert over_path not in {args.checkpoint.resolve(), (output/'prompt.txt').resolve(), (output/'system.txt').resolve(), (output/'manifest.json').resolve()}
        over_path.parent.mkdir(parents=True, exist_ok=True)
        over_path.write_bytes(over_bytes)
        manifest['overbudget_fixture'] = {
            'purpose':'Expected HTTP 400 context rejection through Lumen --no-stream; this is not a passing 1Mi generation fixture.',
            'file':str(over_path), 'expected_http_status':400,
            'prompt_tokens':len(over_ids), 'max_tokens':8,
            'total_budget_tokens':len(over_ids)+8, 'context_limit':LIMIT,
            'file_bytes':len(over_bytes), 'file_sha256':sha(over_bytes),
            'formatted_prompt_bytes':len(over_text.encode()), 'prompt_ids_sha256':ids_sha(over_ids),
            'representative_json_bytes':len(json_bytes), 'representative_json_sha256':sha(json_bytes),
            'json_note':'Representative compact UTF-8 chat payload; actual CLI serialization may vary slightly.',
            'within_64_mib':True, 'within_128_mib':True,
            'inference_performed':False,
        }
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
