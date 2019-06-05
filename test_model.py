import mxnet as mx
import io
from model import GPT2_117M
from transforms import GPT2Tokenizer, GPT2Detokenizer
from gluonnlp.vocab import Vocab

def test_pretrained_gpt2(ctx=None):
    model = GPT2_117M()
    model.initialize(ctx=ctx)
    model.hybridize()
    model.load_parameters(filename='models/117M/model.params')

    tokenizer = GPT2Tokenizer(bpe_ranks_path='models/117M/bpe_ranks.json')
    detokenizer = GPT2Detokenizer(tokenizer)
    with io.open('models/117M/vocab.json', 'r', encoding='utf-8') as f:
        vocab = Vocab.from_json(f.read())
    sentence = ' natural language processing tools such as gluonnlp and torchtext'
    indices = vocab[tokenizer(sentence)]
    nd_indices = mx.nd.expand_dims(mx.nd.array(indices, ctx=ctx), axis=0)
    logits, new_states = model(nd_indices, None)
    print(logits)

test_pretrained_gpt2(ctx=mx.gpu())

