import mxnet as mx
from model import GPT2_117M



def test_pretrained_gpt2():
    model = GPT2_117M()
    model.initialize(ctx=mx.gpu())
    model.hybridize()
    print(model)
