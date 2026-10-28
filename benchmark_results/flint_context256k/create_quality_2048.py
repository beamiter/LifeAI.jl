#!/usr/bin/env python3
"""Add a separate varied 260096+2048 case without changing earlier corpora."""
import argparse
import hashlib
import json
import struct
from pathlib import Path
from create_quality_corpus import MARKERS, TAIL, render, build_varied, Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer-dir', type=Path, required=True)
    parser.add_argument('--corpus-dir', type=Path, required=True,
                        help='Existing output of create_quality_corpus.py; prior manifest is preserved.')
    args = parser.parse_args()
    root = args.corpus_dir
    tokenizer_path = args.tokenizer_dir / 'tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    target, reserve = 260096, 2048
    prefix, record_count, first, last = build_varied(tokenizer, target)
    base = len(tokenizer.encode(render(prefix + TAIL)).ids)
    pad = target - base
    assert 0 <= pad < 256
    for _ in range(8):
        text = prefix + ' filler' * pad + TAIL
        rendered = render(text)
        encoded = tokenizer.encode(rendered)
        difference = target - len(encoded.ids)
        if not difference:
            break
        pad += difference
    assert len(encoded.ids) == target
    raw = text.encode()
    assert len(raw) <= 4 * 1024 * 1024
    name = 'quality-262144-varied-prompt-260096-reserve-2048'
    (root / (name + '.txt')).write_bytes(raw)
    ids_bytes = struct.pack('<' + 'I' * len(encoded.ids), *encoded.ids)
    (root / (name + '.ids.u32')).write_bytes(ids_bytes)
    spans = {}
    for key, value in MARKERS.items():
        character = rendered.index(value)
        overlapping = [i for i, (a, b) in enumerate(encoded.offsets)
                       if a < character + len(value) and b > character]
        spans[key] = {'value': value, 'token_start_zero_based': min(overlapping),
                      'token_end_exclusive': max(overlapping) + 1,
                      'prompt_character_fraction': character / len(rendered)}
    case = {'file': name + '.txt', 'ids_file': name + '.ids.u32', 'style': 'varied',
            'prompt_tokens_official_tokenizers': target, 'requested_output_tokens': reserve,
            'reserved_output_tokens': reserve, 'total_context_budget': 262144,
            'prompt_utf8_bytes': len(raw), 'record_count': record_count,
            'all_record_indices_unique': True, 'padding_words': pad,
            'sha256': hashlib.sha256(raw).hexdigest(),
            'token_ids_zero_based_u32le_sha256': hashlib.sha256(ids_bytes).hexdigest(),
            'marker_spans': spans, 'first_record_samples': first, 'last_record_samples': last}
    original = root / 'quality-manifest.json'
    manifest = {'model': 'RecursiveSI/flint', 'context_tokens': 262144,
                'tokenizer_sha256': hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
                'renderer': '<|user|>{UTF8_FILE}<|assistant|>', 'expected_answer': MARKERS,
                'cases': [case], 'prior_manifest_file': original.name,
                'prior_manifest_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
                'note': 'Separate expanded output budget; all earlier 512-budget files and manifest are unchanged. EOS may stop early.'}
    (root / 'quality-long-2048-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
