import mxnet as mx
import numpy as np
import numpy.testing as npt
import io
from model import GPT2_117M, GPT2_345M
from transforms import GPT2Tokenizer, GPT2Detokenizer
from gluonnlp.vocab import Vocab

def test_pretrained_gpt2(ctx=None):
    sentence = ' natural language processing tools such as gluonnlp and torchtext'
    for model_name in ['117M', '345M']:
        if model_name == '117M':
            model = GPT2_117M()
            model.load_parameters(filename='models/117M/model.params', ctx=ctx)
            gt_logits = np.load('117M_gt_logits.npy')
            tokenizer = GPT2Tokenizer(bpe_ranks_path='models/117M/bpe_ranks.json')
            with io.open('models/117M/vocab.json', 'r', encoding='utf-8') as f:
                vocab = Vocab.from_json(f.read())
        elif model_name == '345M':
            model = GPT2_345M()
            model.load_parameters(filename='models/345M/model.params', ctx=ctx)
            gt_logits = np.load('345M_gt_logits.npy')
            tokenizer = GPT2Tokenizer(bpe_ranks_path='models/345M/bpe_ranks.json')
            with io.open('models/345M/vocab.json', 'r', encoding='utf-8') as f:
                vocab = Vocab.from_json(f.read())
    model.hybridize()
    indices = vocab[tokenizer(sentence)]
    nd_indices = mx.nd.expand_dims(mx.nd.array(indices, ctx=ctx), axis=0)
    logits, new_states = model(nd_indices, None)
    npt.assert_allclose(logits.asnumpy(), gt_logits, 1E-2, 1E-2)

test_pretrained_gpt2(ctx=mx.gpu())

