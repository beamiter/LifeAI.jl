#!/usr/bin/env python3
"""Calibrate indexed ordinary-text retrieval controls at 8K, 32K and 256K."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import struct
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
MARKERS = {'begin': 'AMBER-4827-MICA', 'middle': 'COBALT-9156-RIVER', 'end': 'JADE-6732-ORBIT'}
ROW = 'This is a neutral filler record.\n'
HEAD = ('Read the document and remember the three named identifiers. '
        'The repeated neutral filler records contain no relevant facts.\n'
        f'BEGIN_ID={MARKERS["begin"]}\nDOCUMENT SECTION ONE\n')
MIDDLE = f'MIDDLE_ID={MARKERS["middle"]}\nDOCUMENT SECTION TWO\n'
TAIL = (f'\nEND_ID={MARKERS["end"]}\nQUESTION: What are the three identifiers? '
        'Return only a JSON object with keys begin, middle, end and their exact values '
        'from the document. Ignore the filler records.\n')


def render(text):
    return '<|user|>' + text + '<|assistant|>'


def records():
    rng = random.Random(20261028)
    names = 'Mira Owen Lina Felix Nora Theo Clara Hugo Iris Leon Ada Milo'.split()
    places = ['the public library', 'the city museum', 'a quiet garden', 'the harbor office',
              'the railway station', 'the community workshop', 'the neighborhood school',
              'the old observatory', 'the riverside market', 'a small bakery',
              'the botanical center', 'the town archive']
    objects = ['a blue notebook', 'a wooden picture frame', 'a folded paper map',
               'a ceramic flower pot', 'a clean cotton cloth', 'a brass measuring ruler',
               'a stack of postcards', 'a green canvas bag', 'a box of colored pencils',
               'a repaired desk lamp', 'a wool scarf', 'a simple glass bottle']
    weather = ['clear skies', 'light rain', 'a cool breeze', 'warm sunshine',
               'thin clouds', 'early morning fog', 'a calm afternoon', 'a mild evening']
    templates = [
        '{name} carried {object} to {place} after breakfast; the note mentioned {weather}.',
        'At {place}, {name} checked {number} items and set {object} beside the window.',
        'The visitor log for {place} says {name} arrived on day {number} during {weather}.',
        '{name} spent {number} minutes organizing a shelf at {place} before returning {object}.',
        'A routine inventory at {place} listed {object}; {name} signed the page marked {number}.',
        'Under {weather}, {name} photographed {place} and wrote a brief description of {object}.',
        'The afternoon schedule assigned {name} to {place}; the delivery contained {object}.',
        'A maintenance card numbered {number} described {object} that {name} examined at {place}.',
        '{name} left {object} near the entrance of {place} while the group discussed {weather}.',
        'At the end of a quiet day, {name} counted {number} labels for {place} and packed {object}.',
        'A postcard from {place} described {weather}; {name} placed it next to {object}.',
        'The weekly report records {number} ordinary visits to {place}, where {name} displayed {object}.',
    ]
    index = 0
    while True:
        index += 1
        values = dict(name=rng.choice(names), place=rng.choice(places), object=rng.choice(objects),
                      weather=rng.choice(weather), number=rng.randrange(3, 980))
        yield f'Record {index:06d}: ' + rng.choice(templates).format(**values) + '\n'


def build_varied(tokenizer, target):
    overhead = len(tokenizer.encode(render(HEAD + MIDDLE + TAIL)).ids)
    budget = target - overhead
    iterator = records()
    before, after = [], []
    used = 0
    pending = None
    for collection, limit in ((before, budget // 2), (after, budget - 32)):
        while True:
            record = pending if pending is not None else next(iterator)
            cost = len(tokenizer.encode(record).ids)
            if used + cost > limit:
                pending = record
                break
            collection.append(record)
            used += cost
            pending = None
    return HEAD + ''.join(before) + MIDDLE + ''.join(after), len(before) + len(after), before[:2], after[-2:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.tokenizer_dir / 'tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    cases = []
    for cap, style in ((8192, 'repeated'), (8192, 'varied'), (32768, 'varied'), (262144, 'varied')):
        target, reserve = cap - 512, 512
        if style == 'repeated':
            overhead = len(tokenizer.encode(render(HEAD + MIDDLE + TAIL)).ids)
            count = max(0, (target - overhead) // 14 - 1)
            prefix = HEAD + ROW * count + MIDDLE + ROW * count
            record_count, first, last = count * 2, [ROW.strip()], [ROW.strip()]
        else:
            prefix, record_count, first, last = build_varied(tokenizer, target)
        base = len(tokenizer.encode(render(prefix + TAIL)).ids)
        pad = target - base
        assert 0 <= pad < 256, (style, cap, pad)
        for _ in range(8):
            text = prefix + ' filler' * pad + TAIL
            rendered = render(text)
            encoded = tokenizer.encode(rendered)
            delta = target - len(encoded.ids)
            if not delta:
                break
            pad += delta
        assert len(encoded.ids) == target
        raw = text.encode()
        assert len(raw) <= 4 * 1024 * 1024
        name = f'quality-{cap}-{style}-prompt-{target}-reserve-{reserve}'
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
        cases.append({'file': name + '.txt', 'ids_file': name + '.ids.u32', 'style': style,
                      'prompt_tokens_official_tokenizers': target, 'requested_output_tokens': reserve,
                      'reserved_output_tokens': reserve, 'total_context_budget': cap,
                      'prompt_utf8_bytes': len(raw), 'record_count': record_count,
                      'all_record_indices_unique': style == 'varied', 'padding_words': pad,
                      'sha256': hashlib.sha256(raw).hexdigest(),
                      'token_ids_zero_based_u32le_sha256': hashlib.sha256(ids_bytes).hexdigest(),
                      'marker_spans': spans, 'first_record_samples': first, 'last_record_samples': last})
    manifest = {'model': 'RecursiveSI/flint', 'context_tokens': 262144,
                'tokenizer_sha256': hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
                'renderer': '<|user|>{UTF8_FILE}<|assistant|>', 'expected_answer': MARKERS,
                'cases': cases, 'note': 'Output tokens are reserved budgets; EOS may stop generation earlier.'}
    (root / 'quality-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'cases': [{k: case[k] for k in ('file', 'style', 'prompt_tokens_official_tokenizers',
        'total_context_budget', 'prompt_utf8_bytes', 'record_count', 'padding_words')} for case in cases]}, indent=2))


if __name__ == '__main__':
    main()
