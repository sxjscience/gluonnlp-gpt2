import os
import io
import json
import argparse
from gluonnlp.vocab import Vocab

def convert_vocab_bpe(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    with io.open(os.path.join(src_dir, 'encoder.json'), 'r', encoding='utf-8') as f:
        token_to_idx = json.load(f)
        token_to_idx = {k : int(v) for k, v in token_to_idx.items()}
    idx_to_token = {int(v): k for k, v in token_to_idx.items()}
    vocab = Vocab(unknown_token=None)
    vocab._idx_to_token = idx_to_token
    vocab._token_to_idx = token_to_idx
    vocab._reserved_tokens = None
    vocab._padding_token = None
    vocab._bos_token = None
    vocab._eos_token = '<|endoftext|>'
    with io.open(os.path.join(dst_dir, 'vocab.json'), 'w', encoding='utf-8') as of:
        of.write(vocab.to_json())
    with io.open(os.path.join(src_dir, 'vocab.bpe'), 'r', encoding='utf-8') as f:
        of = io.open(os.path.join(dst_dir, 'bpe_ranks.json'), 'w', encoding='utf-8')
        of.write(f.read())


parser = argparse.ArgumentParser()
parser.add_argument("--src_dir", help="Source path of the model directory in openai/gpt-2", type=str, required=True)
parser.add_argument("--dst_dir", help="Destination path of the model directory of gluonnlp", type=str, required=True)

args = parser.parse_args()
print('Convert {} to {}'.format(args.src_dir, args.dst_dir))
convert_vocab_bpe(args.src_dir, args.dst_dir)
