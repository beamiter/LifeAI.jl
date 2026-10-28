#!/usr/bin/env python3
"""Build exact Flint chat-token budgets with the released tokenizer."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
MARKERS = {'begin': 'AMBER-4827-MICA', 'middle': 'COBALT-9156-RIVER', 'end': 'JADE-6732-ORBIT'}
ROW = 'This is a neutral filler record.\n'
HEAD = ('Read the document and remember the three named identifiers. '
        'The repeated neutral filler records contain no relevant facts.\n'
        f'BEGIN_ID={MARKERS["begin"]}\n'
        'DOCUMENT SECTION ONE\n')
MIDDLE = f'MIDDLE_ID={MARKERS["middle"]}\nDOCUMENT SECTION TWO\n'
TAIL = (f'\nEND_ID={MARKERS["end"]}\n'
        'QUESTION: What are the three identifiers? Return only a JSON object '
        'with keys begin, middle, end and their exact values from the document. '
        'Ignore the filler records.\n')


def render(text):
    return '<|user|>' + text + '<|assistant|>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer-dir', required=True, type=Path,
                        help='Local directory containing the released Flint tokenizer.json.')
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.tokenizer_dir / 'tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    cases = []
    for prompt_tokens, output_tokens in ((261632, 512), (262143, 1)):
        prefix = HEAD + ROW * 18000 + MIDDLE + ROW * 18000
        base_tokens = len(tokenizer.encode(render(prefix + TAIL)).ids)
        padding = prompt_tokens - base_tokens
        assert padding >= 0
        for _ in range(8):
            text = prefix + ' filler' * padding + TAIL
            rendered = render(text)
            encoded = tokenizer.encode(rendered)
            difference = prompt_tokens - len(encoded.ids)
            if difference == 0:
                break
            padding += difference
        assert len(encoded.ids) == prompt_tokens
        data = text.encode()
        assert len(data) <= 4 * 1024 * 1024
        name = f'flint-{prompt_tokens}-prompt-{output_tokens}-output'
        (root / (name + '.txt')).write_bytes(data)
        (root / (name + '.ids.u32')).write_bytes(struct.pack('<' + 'I' * len(encoded.ids), *encoded.ids))
        marker_spans = {}
        for key, value in MARKERS.items():
            character = rendered.index(value)
            overlapping = [i for i, (a, b) in enumerate(encoded.offsets)
                           if a < character + len(value) and b > character]
            marker_spans[key] = {'value': value, 'character_offset': character,
                                 'prompt_character_fraction': character / len(rendered),
                                 'token_start_zero_based': min(overlapping),
                                 'token_end_exclusive': max(overlapping) + 1}
        cases.append({'file': name + '.txt', 'ids_file': name + '.ids.u32',
                      'prompt_tokens_official_tokenizers': len(encoded.ids),
                      'requested_output_tokens': output_tokens,
                      'total_context_budget': prompt_tokens + output_tokens,
                      'prompt_utf8_bytes': len(data), 'padding_words': padding,
                      'sha256': hashlib.sha256(data).hexdigest(),
                      'token_ids_zero_based_u32le_sha256': hashlib.sha256((root / (name + '.ids.u32')).read_bytes()).hexdigest(),
                      'marker_spans': marker_spans})
    manifest = {'model': 'RecursiveSI/flint', 'context_tokens': 262144,
                'tokenizer_sha256': hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
                'renderer': '<|user|>{UTF8_FILE}<|assistant|>',
                'expected_answer': MARKERS, 'cases': cases,
                'note': 'Exact capacity is prompt tokens plus requested output budget; generation may stop early.'}
    (root / 'corpus-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
