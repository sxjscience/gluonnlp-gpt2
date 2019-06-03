import os
import io
import json
from gluonnlp.vocab import Vocab

def convert_vocab_bpe(src_path, dst_path):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    with io.open(os.path.join(src_path, 'encoder.json'), 'r', encoding='utf-8') as f:
        token_to_idx = json.load(f)
        token_to_idx = {k : int(v) for k, v in token_to_idx.items()}
    idx_to_token = {v: k for k, v in token_to_idx.items()}
    vocab = Vocab(unknown_token=None)
    vocab._idx_to_token = idx_to_token
    vocab._token_to_idx = token_to_idx
    vocab._reserved_tokens = None
    vocab._padding_token = None
    vocab._bos_token = None
    vocab._eos_token = '<|endoftext|>'
    with io.open(os.path.join(dst_path, 'vocab.json'), 'w', encoding='utf-8') as of:
        of.write(vocab.to_json())
    with io.open(os.path.join(src_path, 'vocab.bpe'), 'r', encoding='utf-8') as f:
        of = io.open(os.path.join(dst_path, 'bpe_ranks.json'), 'w', encoding='utf-8')
        of.write(f.read())

